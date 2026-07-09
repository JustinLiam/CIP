from __future__ import annotations

import importlib
import fcntl
import random
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import yaml

from .calibration_registry import default_epi_abm_root, project_root, get_registry_entry


def _set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def capture_rng_state() -> Dict[str, Any]:
    state: Dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.random.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: Dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.random.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available() and "torch_cuda" in state:
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def deep_clone(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.clone()
    if isinstance(value, dict):
        return {k: deep_clone(v) for k, v in value.items()}
    if isinstance(value, list):
        return [deep_clone(v) for v in value]
    if isinstance(value, tuple):
        return tuple(deep_clone(v) for v in value)
    return value


class EpiABMWeeklyEnv:
    """Online wrapper around epi-diff-abm with day-level dynamics and held actions."""

    def __init__(
        self,
        *,
        county: str = "01045",
        date_tag: str = "202010-202104",
        epi_root: Optional[str] = None,
        device: str = "cuda",
        action_hold_days: int = 7,
        num_steps: Optional[int] = None,
        num_weeks: Optional[int] = None,
        num_weeks_to_eval: Optional[int] = None,
        intervention_mode: str = "binary_threshold",
    ) -> None:
        self.county = str(county)
        self.date_tag = str(date_tag)
        if epi_root is None:
            self.epi_root = default_epi_abm_root()
        else:
            epi_path = Path(epi_root)
            self.epi_root = epi_path if epi_path.is_absolute() else project_root() / epi_path
        self.device = str(device)
        self.action_hold_days = int(action_hold_days)
        self.entry = get_registry_entry(self.county, date_tag=self.date_tag)
        self.num_steps = int(num_steps or self.entry.num_steps)
        self.num_weeks = int(num_weeks or self.entry.num_weeks)
        self.num_weeks_to_eval = int(num_weeks_to_eval or self.entry.num_weeks_to_eval)
        self.intervention_mode = str(intervention_mode or "binary_threshold")

        self._external_loaded = False
        self.sim = None
        self.runner = None
        self.initial_state = None
        self.calibrated_params_path: Optional[Path] = None
        self.current_day = 0
        self.last_info: Dict[str, Any] = {}

    def _ensure_external_imports(self):
        if str(self.epi_root) not in sys.path:
            sys.path.insert(0, str(self.epi_root))
        from agent_torch.core.dataloader import LoadPopulation
        from agent_torch.core.executor import Executor
        from agent_torch.core.helpers import to_cpu
        from abm_nets import map_and_replace_tensor
        import covid_abm

        self.LoadPopulation = LoadPopulation
        self.Executor = Executor
        self.to_cpu = to_cpu
        self.map_and_replace_tensor = map_and_replace_tensor
        self.covid_abm = covid_abm
        self._external_loaded = True

    @contextmanager
    def _temporary_config(self):
        config_path = self.epi_root / "covid_abm" / "yamls" / "config.yaml"
        lock_path = config_path.with_name(f"{config_path.name}.lock")
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        with lock_path.open("w") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            backup_text = config_path.read_text()
            cfg = yaml.safe_load(backup_text)
            meta = cfg["simulation_metadata"]
            meta["POPULATION"] = self.county
            meta["DATE"] = self.date_tag
            meta["num_steps_per_episode"] = self.num_steps
            meta["NUM_WEEKS"] = self.num_weeks
            meta["NUM_WEEKS_TO_EVAL"] = self.num_weeks_to_eval
            meta["device"] = self.device
            meta["GENERATING_COUNTERFACTUAL"] = False
            meta["calibration"] = True
            config_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
            try:
                yield
            finally:
                config_path.write_text(backup_text)
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

    def _new_runner(self):
        if not self._external_loaded:
            self._ensure_external_imports()
        population_module = importlib.import_module(f"populations.pop{self.county}")
        with self._temporary_config():
            sim = self.Executor(self.covid_abm, pop_loader=self.LoadPopulation(population_module))
            runner = sim._get_runner(sim.config)
            runner.init()
        return sim, runner

    def _load_calibrated_params(self) -> Path:
        param_path = self.entry.calibrated_params_path(self.epi_root)
        if not param_path.exists():
            raise FileNotFoundError(
                f"Missing calibrated params for county {self.county}: {param_path}. "
                "Run the upstream epi-diff-abm calibration first."
            )
        values = np.loadtxt(param_path)
        param_tensor = torch.tensor(values, dtype=torch.float, device=self.device)
        if param_tensor.ndim == 1:
            param_tensor = param_tensor[:, None]

        named = [(name, param) for name, param in self.runner.named_parameters()]
        r2_name = self._find_param_name(named, "R2", 1)
        infected_name = self._find_param_name(named, "infected_proportion", 3)
        k_name = self._find_param_name(named, "k", 4)

        self.map_and_replace_tensor(r2_name)(
            self.runner, True, param_tensor[: self.num_weeks], mode_calibrate=True
        )
        self.map_and_replace_tensor(infected_name)(
            self.runner, True, param_tensor[-2], mode_calibrate=True
        )
        self.map_and_replace_tensor(k_name)(self.runner, True, param_tensor[-1], mode_calibrate=True)
        return param_path

    @staticmethod
    def _find_param_name(named_parameters, suffix: str, fallback_idx: int) -> str:
        matches = [name for name, _ in named_parameters if name.split(".")[-1] == suffix]
        if matches:
            return matches[0]
        return named_parameters[fallback_idx][0]

    def _ensure_runner(self) -> Path:
        if self.runner is None:
            self.sim, self.runner = self._new_runner()
            self.calibrated_params_path = self._load_calibrated_params()
            self.initial_state = deep_clone(self.runner.state)
        if self.initial_state is None or self.calibrated_params_path is None:
            raise RuntimeError("EpiABM runner was initialized without a reusable initial state.")
        return self.calibrated_params_path

    def reset(
        self,
        *,
        decision_day: int = 0,
        seed: int = 0,
        burn_in_actions: Optional[np.ndarray] = None,
    ) -> Dict[str, np.ndarray]:
        """Initialize ABM, load calibrated params, and burn in to decision_day."""
        _set_seed(seed)
        param_path = self._ensure_runner()
        self.runner.state = deep_clone(self.initial_state)
        self.runner.state_trajectory = []
        self.current_day = 0
        self.last_info = {
            "calibrated_params": str(param_path),
            "seed": int(seed),
            "county": self.county,
        }

        if decision_day > 0:
            for day in range(int(decision_day)):
                action = None
                if burn_in_actions is not None:
                    action = np.asarray(burn_in_actions[day], dtype=np.float32)
                self.step_day(action=action)
        return self.aggregate_state(max(self.current_day - 1, 0))

    def _set_online_action(self, action: Optional[Iterable[float]]) -> None:
        if self.runner is None:
            raise RuntimeError("EpiABMWeeklyEnv.reset must be called before stepping.")
        env = self.runner.state.setdefault("environment", {})
        if action is None:
            env.pop("online_intervention", None)
            return
        arr = np.asarray(action, dtype=np.float32).reshape(-1)
        if arr.size < 2:
            raise ValueError(f"Expected action with at least two entries, got shape {arr.shape}.")
        values = torch.tensor(arr[:2], dtype=torch.float, device=self.device)
        if self.intervention_mode in {"continuous_freeze", "continuous", "freezing_interval"}:
            env["online_intervention"] = {"mode": "continuous_freeze", "values": values}
        else:
            env["online_intervention"] = values

    def _step_global_range(self, start_day: int, end_day: int) -> None:
        for time_step in range(int(start_day), int(end_day)):
            self.runner.state["current_step"] = time_step
            self.runner.state_trajectory.append([])

            for substep in self.runner.config["substeps"].keys():
                observation_profile, action_profile = {}, {}
                for agent_type in self.runner.config["substeps"][substep]["active_agents"]:
                    observation_profile[agent_type] = self.runner.controller.observe(
                        self.runner.state,
                        self.runner.initializer.observation_function,
                        agent_type,
                    )
                    action_profile[agent_type] = self.runner.controller.act(
                        self.runner.state,
                        observation_profile[agent_type],
                        self.runner.initializer.policy_function,
                        agent_type,
                    )

                next_state = self.runner.controller.progress(
                    self.runner.state,
                    action_profile,
                    self.runner.initializer.transition_function,
                )
                self.runner.state = next_state
                self.runner.state_trajectory[-1].append(self.to_cpu(self.runner.state))

    def step_day(self, action: Optional[Iterable[float]] = None) -> Tuple[float, Dict[str, np.ndarray]]:
        if self.current_day >= self.num_steps:
            raise RuntimeError(f"Cannot step beyond configured num_steps={self.num_steps}.")
        self._set_online_action(action)
        day = int(self.current_day)
        self._step_global_range(day, day + 1)
        self.current_day += 1
        y = self.daily_cases(day)
        return y, self.aggregate_state(day)

    def step_week(self, action: Iterable[float]) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        ys = []
        for _ in range(self.action_hold_days):
            if self.current_day >= self.num_steps:
                break
            y, _ = self.step_day(action=action)
            ys.append(y)
        return np.asarray(ys, dtype=np.float32), self.aggregate_state(max(self.current_day - 1, 0))

    def rollout(self, actions: np.ndarray) -> np.ndarray:
        actions = np.asarray(actions, dtype=np.float32)
        ys = []
        for row in actions:
            y, _ = self.step_day(row)
            ys.append(y)
        return np.asarray(ys, dtype=np.float32).reshape(-1, 1)

    def snapshot(self) -> Dict[str, Any]:
        if self.runner is None:
            raise RuntimeError("EpiABMWeeklyEnv.reset must be called before taking a snapshot.")
        return {
            "state": deep_clone(self.runner.state),
            "current_day": int(self.current_day),
            "rng_state": capture_rng_state(),
        }

    def restore_snapshot(self, snapshot: Dict[str, Any]) -> Dict[str, np.ndarray]:
        if self.runner is None:
            self._ensure_runner()
        self.runner.state = deep_clone(snapshot["state"])
        self.runner.state_trajectory = []
        self.current_day = int(snapshot["current_day"])
        restore_rng_state(snapshot["rng_state"])
        return self.aggregate_state(max(self.current_day - 1, 0))

    def daily_cases(self, day: int) -> float:
        daily = self.runner.state["environment"]["daily_infected"].detach().cpu().numpy().reshape(-1)
        return float(daily[int(day)])

    def daily_deaths(self, day: int) -> float:
        daily = self.runner.state["environment"]["daily_deaths"].detach().cpu().numpy().reshape(-1)
        return float(daily[int(day)])

    def aggregate_state(self, day: int) -> Dict[str, np.ndarray]:
        stages = self.runner.state["agents"]["citizens"]["disease_stage"].detach().cpu().numpy().reshape(-1)
        n = max(float(stages.shape[0]), 1.0)
        proportions = np.asarray([(stages == k).sum() / n for k in range(5)], dtype=np.float32)
        day = int(max(day, 0))
        return {
            "stage_proportions": proportions,
            "daily_cases": np.asarray([self.daily_cases(day)], dtype=np.float32),
            "daily_deaths": np.asarray([self.daily_deaths(day)], dtype=np.float32),
            "day": np.asarray([day], dtype=np.float32),
        }

    def read_factual_interventions(self) -> np.ndarray:
        path = self.entry.population_dir(self.epi_root) / "intervention.csv"
        df = pd.read_csv(path).sort_values("t")
        values = df[["school_intervention", "occ_intervention"]].values.astype(np.float32)
        return values[: self.num_steps]

    def static_features(self) -> np.ndarray:
        if self.runner is None:
            self.reset(decision_day=0, seed=0)
        ages = self.runner.state["agents"]["citizens"]["age"].detach().cpu().numpy().reshape(-1)
        pop = max(float(ages.shape[0]), 1.0)
        child = float((ages < 1).sum()) / pop
        adult = float(((ages >= 1) & (ages <= 4)).sum()) / pop
        elderly = float((ages > 4).sum()) / pop
        return np.asarray([pop / 100000.0, child, adult, elderly], dtype=np.float32)
