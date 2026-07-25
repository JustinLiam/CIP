from __future__ import annotations

import logging
import json
import pickle
import csv
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from src.data.dataset_collection import SyntheticDatasetCollection

from .calibration_registry import default_epi_abm_root, project_root
from .simulator_cache import EpiABMSimulatorCache, normalize_county_id
from .weekly_env import EpiABMWeeklyEnv


logger = logging.getLogger(__name__)


def _counties_from_epicf_csv(path: str) -> List[str]:
    csv_path = Path(path)
    if not csv_path.is_absolute():
        csv_path = project_root() / csv_path
    counties = set()
    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        if "county" not in (reader.fieldnames or []):
            raise ValueError(f"{csv_path} does not contain a 'county' column.")
        for row in reader:
            counties.add(normalize_county_id(row["county"]))
    return sorted(counties)


def _to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _normalize_outcome_transform(value: str) -> str:
    value = str(value or "raw_cases_zscore").strip().lower()
    aliases = {
        "raw": "raw_cases_zscore",
        "raw_cases": "raw_cases_zscore",
        "raw_cases_zscore": "raw_cases_zscore",
        "cases": "raw_cases_zscore",
        "per10k": "per10k_cases_zscore",
        "per_10k": "per10k_cases_zscore",
        "per10k_cases": "per10k_cases_zscore",
        "per_10k_cases": "per10k_cases_zscore",
        "per10k_cases_zscore": "per10k_cases_zscore",
        "per_10k_cases_zscore": "per10k_cases_zscore",
    }
    if value not in aliases:
        raise ValueError(
            f"Unknown EpiABM outcome_transform={value!r}; expected raw_cases_zscore "
            "or per10k_cases_zscore."
        )
    return aliases[value]


def _population_from_static(static_features: Any, *, n_rows: int) -> np.ndarray:
    static = _to_numpy(static_features).astype(np.float32)
    if static.ndim == 3:
        pop_scaled = static[:, 0, 0]
    elif static.ndim == 2:
        pop_scaled = static[:, 0]
    elif static.ndim == 1:
        pop_scaled = static[:1]
    else:
        raise ValueError(f"static_features must have shape [B, F] or [B, T, F], got {static.shape}")
    population = np.asarray(pop_scaled, dtype=np.float32).reshape(-1, 1, 1) * 100000.0
    if population.shape[0] == 1 and n_rows != 1:
        population = np.repeat(population, n_rows, axis=0)
    if population.shape[0] != n_rows:
        raise ValueError(f"population rows {population.shape[0]} do not match outputs rows {n_rows}")
    population = np.where(population > 0, population, 1.0)
    return population.astype(np.float32)


class EpiABMDataset(Dataset):
    """Daily EpiABM trajectories with a MIMIC-style simulator oracle."""

    def __init__(
        self,
        data: Dict[str, np.ndarray],
        *,
        subset_name: str,
        epi_root: str,
        county: str,
        date_tag: str,
        action_hold_days: int,
        episode_actions: Dict[int, np.ndarray],
        simulator_cache: Optional[EpiABMSimulatorCache] = None,
        intervention_mode: str = "binary_threshold",
        outcome_transform: str = "raw_cases_zscore",
        base_seed: int = 0,
        device: str = "cuda",
    ) -> None:
        self.data = data
        self.subset_name = subset_name
        self.epi_root = str(epi_root)
        self.county = str(county)
        self.date_tag = str(date_tag)
        self.action_hold_days = int(action_hold_days)
        self.episode_actions = {int(k): np.asarray(v, dtype=np.float32) for k, v in episode_actions.items()}
        self.intervention_mode = str(intervention_mode or "binary_threshold")
        self.outcome_transform = _normalize_outcome_transform(outcome_transform)
        self.base_seed = int(base_seed)
        self.device = str(device)
        self.processed = False
        self.norm_const = 1.0
        self.scaling_params = None
        self.max_seq_length = int(self.data["active_entries"].shape[1])
        self.simulator_cache = simulator_cache or EpiABMSimulatorCache(
            epi_root=self.epi_root,
            date_tag=self.date_tag,
            device=self.device,
            action_hold_days=self.action_hold_days,
            max_seq_length=self.max_seq_length,
            intervention_mode=self.intervention_mode,
        )

    def __len__(self):
        return len(self.data["active_entries"])

    def __getitem__(self, index) -> dict:
        return {k: v[index] for k, v in self.data.items()}

    def _to_model_output_scale(
        self,
        raw_outputs: Any,
        static_features: Optional[Any] = None,
    ) -> np.ndarray:
        raw = _to_numpy(raw_outputs).astype(np.float32)
        if self.outcome_transform == "raw_cases_zscore":
            return raw
        if static_features is None:
            static_features = self.data["static_features"]
        population = _population_from_static(static_features, n_rows=raw.shape[0])
        return (raw / population * 10000.0).astype(np.float32)

    def get_scaling_params(self) -> Dict[str, np.ndarray]:
        outputs = self._to_model_output_scale(self.data["unscaled_outputs"])
        mask = self.data["active_entries"].astype(bool)
        valid = outputs[mask.reshape(outputs.shape[0], outputs.shape[1])]
        if valid.size == 0:
            valid = outputs.reshape(-1, outputs.shape[-1])
        mean = valid.reshape(-1, outputs.shape[-1]).mean(axis=0)
        std = valid.reshape(-1, outputs.shape[-1]).std(axis=0)
        std = np.where(std < 1e-6, 1.0, std)
        return {
            "output_means": mean.astype(np.float32),
            "output_stds": std.astype(np.float32),
            "outcome_transform": self.outcome_transform,
            "output_space": "raw_cases" if self.outcome_transform == "raw_cases_zscore" else "cases_per_10k",
        }

    def process_data(self, scaling_params):
        if self.processed:
            return self.data
        self.scaling_params = scaling_params
        mean = np.asarray(scaling_params["output_means"], dtype=np.float32)
        std = np.asarray(scaling_params["output_stds"], dtype=np.float32)
        outputs = self._to_model_output_scale(self.data["unscaled_outputs"])
        prev_outputs = self._to_model_output_scale(self.data["prev_unscaled_outputs"])
        self.data["outputs"] = ((outputs - mean) / std).astype(np.float32)
        self.data["prev_outputs"] = ((prev_outputs - mean) / std).astype(np.float32)
        self.data["model_unscaled_outputs"] = outputs.astype(np.float32)
        self.data["prev_model_unscaled_outputs"] = prev_outputs.astype(np.float32)
        self.processed = True
        return self.data

    def _covariates_from_agg(self, day: int, agg: Dict[str, np.ndarray]) -> np.ndarray:
        day_norm = float(day) / max(float(self.max_seq_length - 1), 1.0)
        return np.concatenate(
            [
                np.asarray(agg["stage_proportions"], dtype=np.float32),
                np.asarray(
                    [
                        float(np.asarray(agg["daily_deaths"]).reshape(-1)[0]),
                        day_norm,
                        float(int(day) % self.action_hold_days == 0),
                    ],
                    dtype=np.float32,
                ),
            ],
            axis=0,
        ).astype(np.float32)

    def _history_metadata(self, H: Dict[str, np.ndarray], row: int) -> Dict[str, Any]:
        active = H.get("active_entries")
        if active is not None:
            history_len = int(np.asarray(active[row]).reshape(-1).sum())
        else:
            history_len = int(H["current_treatments"].shape[1])
        history_len = max(history_len, 1)

        if "sim_day" in H:
            first_day = int(H["sim_day"][row, 0].reshape(-1)[0])
            last_day = int(H["sim_day"][row, history_len - 1].reshape(-1)[0])
            decision_day = last_day + 1
        else:
            first_day = 0
            decision_day = history_len
            last_day = decision_day - 1

        if "sim_episode_id" in H:
            episode_id = int(H["sim_episode_id"][row, history_len - 1].reshape(-1)[0])
        else:
            episode_id = 0
        if "sim_seed" in H:
            seed = int(H["sim_seed"][row, history_len - 1].reshape(-1)[0])
        else:
            seed = self.base_seed + episode_id
        if "sim_county_id" in H:
            county = normalize_county_id(H["sim_county_id"][row, history_len - 1].reshape(-1)[0])
        else:
            county = normalize_county_id(self.county)

        history_actions = np.asarray(
            H["current_treatments"][row, :history_len], dtype=np.float32
        )
        return {
            "history_len": history_len,
            "first_day": first_day,
            "last_day": last_day,
            "decision_day": decision_day,
            "episode_id": episode_id,
            "seed": seed,
            "county": county,
            "history_actions": history_actions,
        }

    def _burn_in_actions_from_history(self, meta: Dict[str, Any]) -> np.ndarray:
        history_actions = np.asarray(meta["history_actions"], dtype=np.float32)
        decision_day = int(meta["decision_day"])
        first_day = int(meta["first_day"])
        episode_id = int(meta["episode_id"])
        action_dim = int(history_actions.shape[-1])

        base = self.episode_actions.get(episode_id)
        if base is None:
            burn = np.zeros((max(decision_day, first_day + history_actions.shape[0]), action_dim), dtype=np.float32)
        else:
            burn = np.asarray(base, dtype=np.float32).copy()
            if burn.ndim != 2 or burn.shape[-1] != action_dim:
                raise ValueError(
                    f"Episode {episode_id} burn-in actions have shape {burn.shape}, "
                    f"expected [T, {action_dim}]."
                )
            if burn.shape[0] < decision_day:
                pad = np.zeros((decision_day - burn.shape[0], action_dim), dtype=np.float32)
                burn = np.concatenate([burn, pad], axis=0)

        start = max(first_day, 0)
        end = min(start + history_actions.shape[0], decision_day)
        if end > start:
            burn[start:end] = history_actions[: end - start]
        return burn[:decision_day].astype(np.float32)

    def simulate_trajectory_after_actions(
        self,
        H_t,
        actions,
        scaling_params=None,
    ) -> Dict[str, np.ndarray]:
        """Roll out calibrated ABM and return full daily simulated observations."""
        scaling_params = scaling_params or self.scaling_params or self.get_scaling_params()
        mean = np.asarray(scaling_params["output_means"], dtype=np.float32)
        std = np.asarray(scaling_params["output_stds"], dtype=np.float32)

        H = {k: _to_numpy(v) for k, v in H_t.items()}
        actions_np = _to_numpy(actions).astype(np.float32)
        if actions_np.ndim == 2:
            actions_np = actions_np[:, None, :]
        if actions_np.ndim != 3:
            raise ValueError(f"actions must have shape [B, tau, A], got {actions_np.shape}")

        batch_size = actions_np.shape[0]
        daily_outputs, daily_covariates = [], []
        daily_days, daily_episode_ids, daily_seeds, daily_counties = [], [], [], []
        for i in range(batch_size):
            meta = self._history_metadata(H, i)
            burn_in_actions = self._burn_in_actions_from_history(meta)
            behavior_actions = self.episode_actions.get(int(meta["episode_id"]))
            if behavior_actions is None:
                raise KeyError(f"No behavior action schedule for episode_id={meta['episode_id']}.")
            trajectory = self.simulator_cache.rollout_from_history(
                county=meta["county"],
                episode_id=int(meta["episode_id"]),
                seed=int(meta["seed"]),
                decision_day=int(meta["decision_day"]),
                burn_in_actions=burn_in_actions,
                behavior_actions=behavior_actions,
                planned_actions=actions_np[i],
            )
            ys, covs, days = [], [], []
            for y, agg in trajectory:
                day = int(np.asarray(agg["day"]).reshape(-1)[0])
                ys.append([y])
                covs.append(self._covariates_from_agg(day, agg))
                days.append([float(day)])
            daily = np.asarray(ys, dtype=np.float32)
            daily_outputs.append(daily)
            daily_covariates.append(np.asarray(covs, dtype=np.float32))
            daily_days.append(np.asarray(days, dtype=np.float32))
            daily_episode_ids.append(np.full((actions_np.shape[1], 1), float(meta["episode_id"]), dtype=np.float32))
            daily_seeds.append(np.full((actions_np.shape[1], 1), float(meta["seed"]), dtype=np.float32))
            daily_counties.append(np.full((actions_np.shape[1], 1), float(int(meta["county"])), dtype=np.float32))

        daily_arr = np.stack(daily_outputs, axis=0).astype(np.float32)
        model_daily = self._to_model_output_scale(daily_arr, H.get("static_features"))
        norm_daily = ((model_daily - mean) / std).astype(np.float32)
        treatments = actions_np.astype(np.float32)
        return {
            "outputs": norm_daily,
            "unscaled_outputs": daily_arr,
            "model_unscaled_outputs": model_daily,
            "current_treatments": treatments,
            "current_covariates": np.stack(daily_covariates, axis=0).astype(np.float32),
            "vitals": np.stack(daily_covariates, axis=0).astype(np.float32),
            "active_entries": np.ones((batch_size, actions_np.shape[1], 1), dtype=np.float32),
            "sim_day": np.stack(daily_days, axis=0).astype(np.float32),
            "sim_episode_id": np.stack(daily_episode_ids, axis=0).astype(np.float32),
            "sim_seed": np.stack(daily_seeds, axis=0).astype(np.float32),
            "sim_county_id": np.stack(daily_counties, axis=0).astype(np.float32),
        }

    def simulate_next_after_action(
        self,
        H_t,
        action,
        scaling_params=None,
    ) -> Dict[str, np.ndarray]:
        actions_np = _to_numpy(action).astype(np.float32)
        if actions_np.ndim == 2:
            actions_np = actions_np[:, None, :]
        traj = self.simulate_trajectory_after_actions(H_t, actions_np[:, :1], scaling_params)
        return {key: value[:, 0, :] for key, value in traj.items()}

    def start_simulation_session(self, H_t) -> Dict[str, Any]:
        """Restore one batch row once for an autoregressive online rollout."""
        H = {key: _to_numpy(value) for key, value in H_t.items()}
        if int(H["current_treatments"].shape[0]) != 1:
            raise ValueError("Persistent EpiABM sessions currently require batch size 1.")
        meta = self._history_metadata(H, 0)
        burn_in_actions = self._burn_in_actions_from_history(meta)
        behavior_actions = self.episode_actions.get(int(meta["episode_id"]))
        if behavior_actions is None:
            raise KeyError(
                f"No behavior action schedule for episode_id={meta['episode_id']}."
            )
        env = self.simulator_cache.restore_from_history(
            county=meta["county"],
            episode_id=int(meta["episode_id"]),
            seed=int(meta["seed"]),
            decision_day=int(meta["decision_day"]),
            burn_in_actions=burn_in_actions,
            behavior_actions=behavior_actions,
        )
        return {
            "env": env,
            "meta": meta,
            "static_features": H.get("static_features"),
        }

    def simulate_next_in_session(
        self,
        session: Dict[str, Any],
        action,
        scaling_params=None,
    ) -> Dict[str, np.ndarray]:
        """Advance a restored EpiABM session by one day without replaying history."""
        scaling_params = scaling_params or self.scaling_params or self.get_scaling_params()
        mean = np.asarray(scaling_params["output_means"], dtype=np.float32)
        std = np.asarray(scaling_params["output_stds"], dtype=np.float32)
        actions_np = _to_numpy(action).astype(np.float32)
        if actions_np.ndim == 1:
            actions_np = actions_np[None, :]
        if actions_np.shape[0] != 1:
            raise ValueError(
                f"Persistent EpiABM sessions require one action row, got {actions_np.shape}."
            )

        env = session["env"]
        meta = session["meta"]
        y, agg = env.step_day(actions_np[0])
        day = int(np.asarray(agg["day"]).reshape(-1)[0])
        raw_output = np.asarray([[[y]]], dtype=np.float32)
        model_output = self._to_model_output_scale(
            raw_output,
            session.get("static_features"),
        )
        norm_output = ((model_output - mean) / std).astype(np.float32)
        covariates = self._covariates_from_agg(day, agg)[None, None, :]
        treatments = actions_np[None, :, :]

        trajectory = {
            "outputs": norm_output,
            "unscaled_outputs": raw_output,
            "model_unscaled_outputs": model_output,
            "current_treatments": treatments.astype(np.float32),
            "current_covariates": covariates.astype(np.float32),
            "vitals": covariates.astype(np.float32),
            "active_entries": np.ones((1, 1, 1), dtype=np.float32),
            "sim_day": np.asarray([[[float(day)]]], dtype=np.float32),
            "sim_episode_id": np.asarray(
                [[[float(meta["episode_id"])]]], dtype=np.float32
            ),
            "sim_seed": np.asarray([[[float(meta["seed"])]]], dtype=np.float32),
            "sim_county_id": np.asarray(
                [[[float(int(meta["county"]))]]], dtype=np.float32
            ),
        }
        return {key: value[:, 0, :] for key, value in trajectory.items()}

    def simulate_output_after_actions(
        self,
        H_t,
        actions,
        scaling_params=None,
        *,
        return_daily: bool = False,
    ) -> np.ndarray:
        """Roll out calibrated ABM from the metadata encoded in H_t."""
        traj = self.simulate_trajectory_after_actions(H_t, actions, scaling_params)
        if return_daily:
            return traj["outputs"]
        return traj["outputs"][:, -1, :].astype(np.float32)


class EpiABMDatasetCollection(SyntheticDatasetCollection):
    """Dataset collection for calibrated epi-diff-abm daily trajectories."""

    def __init__(
        self,
        *,
        seed: int = 20260704,
        name: str = "epi_abm_01045",
        county: str = "01045",
        counties: Optional[Iterable[str]] = None,
        counties_from_epicf_csv: Optional[str] = None,
        date_tag: str = "202010-202104",
        epi_root: Optional[str] = None,
        processed_data_dir: str = "data/processed/epi_abm/01045",
        generate_if_missing: bool = True,
        force_regenerate: bool = False,
        device: str = "cuda",
        action_hold_days: int = 7,
        max_seq_length: int = 182,
        projection_horizon: int = 14,
        split: Optional[Dict[str, float]] = None,
        split_by: str = "county",
        num_random_policies: int = 1,
        behavior_policy_subset: str = "factual_only",
        treatment_mode: str = "binary",
        intervention_mode: Optional[str] = None,
        random_policy_mode: Optional[str] = None,
        outcome_transform: str = "raw_cases_zscore",
        cache_version: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.seed = int(seed)
        self.name = str(name)
        if counties is None and counties_from_epicf_csv:
            counties = _counties_from_epicf_csv(counties_from_epicf_csv)
        self.counties = [normalize_county_id(c) for c in (counties or [county])]
        self.county = self.counties[0]
        self.date_tag = str(date_tag)
        if epi_root is None:
            self.epi_root = str(default_epi_abm_root())
        else:
            epi_path = Path(epi_root)
            self.epi_root = str(epi_path if epi_path.is_absolute() else project_root() / epi_path)
        self.processed_data_dir = Path(processed_data_dir)
        self.generate_if_missing = bool(generate_if_missing)
        self.force_regenerate = bool(force_regenerate)
        self.device = str(device)
        self.action_hold_days = int(action_hold_days)
        self.max_seq_length = int(max_seq_length)
        self.projection_horizon = int(projection_horizon)
        self.split = split or {"val": 0.25, "test": 0.25}
        self.split_by = str(split_by or "county").strip().lower()
        self.num_random_policies = int(num_random_policies)
        self.behavior_policy_subset = str(behavior_policy_subset or "factual_only").strip().lower()
        self.treatment_mode = str(treatment_mode or "binary")
        self.outcome_transform = _normalize_outcome_transform(outcome_transform)
        if intervention_mode is None:
            intervention_mode = "continuous_freeze" if self.treatment_mode == "continuous" else "binary_threshold"
        self.intervention_mode = str(intervention_mode)
        if random_policy_mode is None:
            random_policy_mode = "continuous_weekly" if self.intervention_mode == "continuous_freeze" else "binary_weekly"
        self.random_policy_mode = str(random_policy_mode)
        if cache_version is None:
            cache_version = "daily_v2_continuous" if self.intervention_mode == "continuous_freeze" else "daily_v1"
        self.cache_version = str(cache_version)
        self.autoregressive = True
        self.has_vitals = True
        self.simulator_cache = EpiABMSimulatorCache(
            epi_root=self.epi_root,
            date_tag=self.date_tag,
            device=self.device,
            action_hold_days=self.action_hold_days,
            max_seq_length=self.max_seq_length,
            intervention_mode=self.intervention_mode,
        )

        bundle = self._load_or_generate()
        self.episode_actions = bundle["episode_actions"]
        self.static_dim = int(bundle["data"]["static_features"].shape[-1])
        self.vitals_dim = int(bundle["data"]["current_covariates"].shape[-1])
        self.split_indices = bundle["split_indices"]

        train_data, val_data, test_data = self._split_data(bundle["data"], self.split_indices)
        common = {
            "epi_root": self.epi_root,
            "county": self.county,
            "date_tag": self.date_tag,
            "action_hold_days": self.action_hold_days,
            "episode_actions": self.episode_actions,
            "simulator_cache": self.simulator_cache,
            "intervention_mode": self.intervention_mode,
            "outcome_transform": self.outcome_transform,
            "base_seed": self.seed,
            "device": self.device,
        }
        self.train_f = EpiABMDataset(train_data, subset_name="train", **common)
        self.val_f = EpiABMDataset(val_data, subset_name="val", **common)
        self.test_f = EpiABMDataset(test_data, subset_name="test", **common)
        self.train_scaling_params = self.train_f.get_scaling_params()

    def _cache_path(self) -> Path:
        county_label = self.county if len(self.counties) == 1 else "multi_" + "_".join(self.counties[:3])
        if len(self.counties) > 3:
            county_label = f"{county_label}_plus{len(self.counties) - 3}"
        return self.processed_data_dir / f"{county_label}_{self.date_tag}_{self.cache_version}.pkl"

    def _manifest_path(self) -> Path:
        return self._cache_path().with_suffix(".manifest.json")

    def _load_or_generate(self) -> Dict[str, Any]:
        path = self._cache_path()
        if path.exists() and not self.force_regenerate:
            with path.open("rb") as f:
                bundle = pickle.load(f)
            metadata = dict(bundle.get("metadata", {}))
            if metadata.get("behavior_policy_subset") != self.behavior_policy_subset:
                if not self.generate_if_missing:
                    raise ValueError(
                        f"EpiABM cache {path} was built with behavior_policy_subset="
                        f"{metadata.get('behavior_policy_subset')!r}, expected {self.behavior_policy_subset!r}."
                    )
                logger.warning(
                    "Regenerating EpiABM cache because behavior_policy_subset changed: cached=%s expected=%s",
                    metadata.get("behavior_policy_subset"),
                    self.behavior_policy_subset,
                )
                bundle = self._ensure_split_indices(self._generate_bundle())
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("wb") as f:
                    pickle.dump(bundle, f)
                self._write_manifest(bundle)
                return bundle
            bundle = self._ensure_split_indices(bundle)
            self._write_manifest(bundle)
            return bundle
        if not self.generate_if_missing:
            raise FileNotFoundError(f"EpiABM cache not found: {path}")
        bundle = self._ensure_split_indices(self._generate_bundle())
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump(bundle, f)
        self._write_manifest(bundle)
        return bundle

    def _write_manifest(self, bundle: Dict[str, Any]) -> None:
        metadata = dict(bundle.get("metadata", {}))
        metadata.update(
            {
                "county": self.county,
                "counties": self.counties,
                "date_tag": self.date_tag,
                "seed": self.seed,
                "max_seq_length": self.max_seq_length,
                "action_hold_days": self.action_hold_days,
                "num_random_policies": self.num_random_policies,
                "behavior_policy_subset": self.behavior_policy_subset,
                "split_by": self.split_by,
                "treatment_mode": self.treatment_mode,
                "intervention_mode": self.intervention_mode,
                "outcome_transform": self.outcome_transform,
                "random_policy_mode": self.random_policy_mode,
                "cache_version": self.cache_version,
                "cache_path": str(self._cache_path()),
                "split_indices": bundle.get("split_indices", {}),
                "split_counties": bundle.get("metadata", {}).get("split_counties", {}),
                "split_by_effective": bundle.get("metadata", {}).get("split_by_effective", self.split_by),
            }
        )
        path = self._manifest_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")

    def _policy_schedules(self, env: EpiABMWeeklyEnv, county: str) -> List[Tuple[str, np.ndarray]]:
        factual = env.read_factual_interventions().astype(np.float32)[: self.max_seq_length]
        if self.behavior_policy_subset in {"factual", "factual_only", "observed"}:
            return [("factual", factual)]
        if self.behavior_policy_subset not in {"all", "augmented"}:
            raise ValueError(
                f"Unknown behavior_policy_subset={self.behavior_policy_subset!r}; "
                "expected 'factual_only' or 'all'."
            )
        all_open = np.zeros_like(factual)
        all_closed = np.ones_like(factual)
        schedules = [("factual", factual), ("all_open", all_open), ("all_closed", all_closed)]

        rng = np.random.RandomState(self.seed + int(county))
        for idx in range(self.num_random_policies):
            actions = np.zeros_like(factual)
            for start in range(0, self.max_seq_length, self.action_hold_days):
                if self.random_policy_mode in {"continuous", "continuous_weekly"}:
                    action = rng.uniform(0.0, 1.0, size=(2,)).astype(np.float32)
                else:
                    action = rng.binomial(1, 0.5, size=(2,)).astype(np.float32)
                actions[start : start + self.action_hold_days] = action
            schedules.append((f"random_weekly_{idx}", actions))
        return schedules

    def _generate_bundle(self) -> Dict[str, Any]:
        rows = []
        episode_actions: Dict[int, np.ndarray] = {}
        static_features = []
        episode_id = 0
        for county in self.counties:
            print(json.dumps({
                "event": "epi_abm_generate_county_start",
                "county": county,
                "episode_id": int(episode_id),
            }), flush=True)
            try:
                env = self.simulator_cache.get_env(county)
                schedules = self._policy_schedules(env, county)
                for policy_name, schedule in schedules:
                    seed = self.seed + episode_id
                    trajectory = self.simulator_cache.precompute_behavior(
                        county=county,
                        episode_id=episode_id,
                        seed=seed,
                        actions=schedule,
                    )
                    static = env.static_features()
                    outputs, deaths, covariates = [], [], []
                    for day, (y, agg) in enumerate(trajectory):
                        outputs.append([y])
                        deaths.append(float(agg["daily_deaths"][0]))
                        day_norm = float(day) / max(float(self.max_seq_length - 1), 1.0)
                        cov = np.concatenate(
                            [
                                agg["stage_proportions"],
                                np.asarray([deaths[-1], day_norm, float(day % self.action_hold_days == 0)], dtype=np.float32),
                            ],
                            axis=0,
                        )
                        covariates.append(cov)
                    rows.append(
                        {
                            "episode_id": episode_id,
                            "county": county,
                            "policy_name": f"{county}:{policy_name}",
                            "seed": seed,
                            "outputs": np.asarray(outputs, dtype=np.float32),
                            "current_treatments": schedule.astype(np.float32),
                            "current_covariates": np.asarray(covariates, dtype=np.float32),
                        }
                    )
                    episode_actions[episode_id] = schedule.astype(np.float32)
                    static_features.append(static)
                    episode_id += 1
                print(json.dumps({
                    "event": "epi_abm_generate_county_done",
                    "county": county,
                    "episode_id_next": int(episode_id),
                }), flush=True)
            except Exception as exc:
                print(json.dumps({
                    "event": "epi_abm_generate_county_failed",
                    "county": county,
                    "episode_id": int(episode_id),
                    "error": repr(exc),
                }), flush=True)
                raise
            finally:
                self.simulator_cache.release_county(county)

        data = self._stack_rows(rows, np.asarray(static_features, dtype=np.float32))
        policy_names = [str(r["policy_name"]) for r in rows]
        return {
            "data": data,
            "episode_actions": episode_actions,
            "metadata": {
                "policy_names": policy_names,
                "num_episodes": len(policy_names),
                "policy_name_by_episode": {str(i): name for i, name in enumerate(policy_names)},
                "counties": self.counties,
                "behavior_policy_subset": self.behavior_policy_subset,
            },
        }

    def _stack_rows(self, rows: List[Dict[str, Any]], static_features: np.ndarray) -> Dict[str, np.ndarray]:
        outputs = np.stack([r["outputs"] for r in rows], axis=0).astype(np.float32)
        treatments = np.stack([r["current_treatments"] for r in rows], axis=0).astype(np.float32)
        covariates = np.stack([r["current_covariates"] for r in rows], axis=0).astype(np.float32)
        active = np.ones((len(rows), self.max_seq_length, 1), dtype=np.float32)
        prev_outputs = np.concatenate([np.zeros_like(outputs[:, :1]), outputs[:, :-1]], axis=1)
        prev_treatments = np.concatenate([np.zeros_like(treatments[:, :1]), treatments[:, :-1]], axis=1)
        next_covariates = np.concatenate([covariates[:, 1:], covariates[:, -1:]], axis=1)
        sim_day = np.tile(np.arange(self.max_seq_length, dtype=np.float32)[None, :, None], (len(rows), 1, 1))
        episode_ids = np.asarray([r["episode_id"] for r in rows], dtype=np.float32)[:, None, None]
        seeds = np.asarray([r["seed"] for r in rows], dtype=np.float32)[:, None, None]
        county_id = np.asarray([float(int(r["county"])) for r in rows], dtype=np.float32)[:, None, None]
        return {
            "sequence_lengths": np.full((len(rows),), self.max_seq_length, dtype=np.float32),
            "prev_treatments": prev_treatments,
            "current_treatments": treatments,
            "prev_unscaled_outputs": prev_outputs,
            "unscaled_outputs": outputs,
            "current_covariates": covariates,
            "next_covariates": next_covariates,
            "vitals": covariates,
            "next_vitals": next_covariates,
            "static_features": static_features.astype(np.float32),
            "active_entries": active,
            "sim_day": sim_day,
            "sim_episode_id": np.tile(episode_ids, (1, self.max_seq_length, 1)),
            "sim_seed": np.tile(seeds, (1, self.max_seq_length, 1)),
            "sim_county_id": np.tile(county_id, (1, self.max_seq_length, 1)),
        }

    def _ensure_split_indices(self, bundle: Dict[str, Any]) -> Dict[str, Any]:
        metadata = dict(bundle.get("metadata", {}))
        if (
            "split_indices" not in bundle
            or metadata.get("split_by_requested") != self.split_by
            or int(metadata.get("split_seed", -1)) != self.seed
        ):
            split_indices, split_metadata = self._make_split_indices(bundle["data"])
            bundle["split_indices"] = split_indices
            metadata.update(split_metadata)
            bundle["metadata"] = metadata
        return bundle

    def _make_split_indices(self, data_or_n) -> Tuple[Dict[str, List[int]], Dict[str, Any]]:
        if isinstance(data_or_n, dict) and self.split_by == "county":
            indices, metadata = self._make_county_split_indices(data_or_n)
            if indices is not None:
                return indices, metadata

        n = int(data_or_n["active_entries"].shape[0]) if isinstance(data_or_n, dict) else int(data_or_n)
        indices = self._make_episode_split_indices(n)
        metadata = {
            "split_by_requested": self.split_by,
            "split_by_effective": "episode",
            "split_seed": self.seed,
            "split_counties": {},
        }
        if self.split_by == "county":
            metadata["split_fallback_reason"] = "county split requires at least three distinct counties"
        return indices, metadata

    def _make_episode_split_indices(self, n: int) -> Dict[str, List[int]]:
        n = int(n)
        rng = np.random.RandomState(self.seed)
        indices = np.arange(n)
        rng.shuffle(indices)
        n_test = max(1, int(round(n * float(self.split.get("test", 0.25))))) if n >= 3 else 1
        n_val = max(1, int(round(n * float(self.split.get("val", 0.25))))) if n >= 3 else 1
        n_train = max(1, n - n_val - n_test)
        train_idx = indices[:n_train].astype(int).tolist()
        val_idx = indices[n_train : n_train + n_val].astype(int).tolist()
        test_idx = indices[n_train + n_val :].astype(int).tolist()
        if len(test_idx) == 0:
            test_idx = list(val_idx)
        return {"train": train_idx, "val": val_idx, "test": test_idx}

    def _make_county_split_indices(self, data: Dict[str, np.ndarray]) -> Tuple[Optional[Dict[str, List[int]]], Dict[str, Any]]:
        county_values = np.asarray(data["sim_county_id"])[:, 0, 0]
        row_counties = np.asarray([normalize_county_id(x) for x in county_values])
        counties = np.asarray(sorted(set(row_counties.tolist())))
        if counties.size < 3:
            return None, {}

        rng = np.random.RandomState(self.seed)
        shuffled = counties.copy()
        rng.shuffle(shuffled)
        n_counties = int(shuffled.size)
        n_test = max(1, int(round(n_counties * float(self.split.get("test", 0.25)))))
        n_val = max(1, int(round(n_counties * float(self.split.get("val", 0.25)))))
        if n_test + n_val >= n_counties:
            n_test = 1
            n_val = 1
        n_train = n_counties - n_val - n_test

        train_counties = set(shuffled[:n_train].tolist())
        val_counties = set(shuffled[n_train : n_train + n_val].tolist())
        test_counties = set(shuffled[n_train + n_val :].tolist())

        split_indices = {
            "train": np.where(np.isin(row_counties, list(train_counties)))[0].astype(int).tolist(),
            "val": np.where(np.isin(row_counties, list(val_counties)))[0].astype(int).tolist(),
            "test": np.where(np.isin(row_counties, list(test_counties)))[0].astype(int).tolist(),
        }
        if not split_indices["train"] or not split_indices["val"] or not split_indices["test"]:
            return None, {}

        split_counties = {
            "train": sorted(train_counties),
            "val": sorted(val_counties),
            "test": sorted(test_counties),
        }
        metadata = {
            "split_by_requested": self.split_by,
            "split_by_effective": "county",
            "split_seed": self.seed,
            "split_counties": split_counties,
        }
        return split_indices, metadata

    def _split_data(
        self,
        data: Dict[str, np.ndarray],
        split_indices: Dict[str, Iterable[int]],
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        train_idx = split_indices["train"]
        val_idx = split_indices["val"]
        test_idx = split_indices["test"]
        return self._take(data, train_idx), self._take(data, val_idx), self._take(data, test_idx)

    @staticmethod
    def _take(data: Dict[str, np.ndarray], indices: Iterable[int]) -> Dict[str, np.ndarray]:
        idx = np.asarray(list(indices), dtype=np.int64)
        return {
            k: (v[idx].copy() if hasattr(v, "__len__") and len(v) == len(data["active_entries"]) else v)
            for k, v in data.items()
        }

    def process_data_multi(self):
        self.train_f.process_data(self.train_scaling_params)
        self.val_f.process_data(self.train_scaling_params)
        self.test_f.process_data(self.train_scaling_params)
        self.processed_data_multi = True

    def process_data_encoder(self):
        self.process_data_multi()

    def process_data_decoder(self, encoder=None, save_encoder_r=False):
        self.process_data_multi()
