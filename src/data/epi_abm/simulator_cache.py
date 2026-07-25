from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
import gc

import numpy as np
import torch

from .calibration_registry import project_root
from .weekly_env import EpiABMWeeklyEnv, deep_clone


def normalize_county_id(county: Any) -> str:
    if isinstance(county, bytes):
        county = county.decode("utf-8")
    if isinstance(county, str):
        county = county.strip()
        return county.zfill(5) if county.isdigit() else county
    return f"{int(float(county)):05d}"


class EpiABMSimulatorCache:
    """Reusable EpiABM runners and behavior-policy day-start snapshots.

    A behavior snapshot at day d is the ABM state before stepping day d, with
    Python/NumPy/Torch RNG state captured. Online rollouts can therefore start
    from the exact behavior prefix and only replay days after the policy first
    diverges from that behavior trajectory.
    """

    def __init__(
        self,
        *,
        epi_root: str,
        date_tag: str,
        device: str,
        action_hold_days: int,
        max_seq_length: int,
        intervention_mode: str,
    ) -> None:
        epi_path = Path(epi_root)
        self.epi_root = str(epi_path if epi_path.is_absolute() else project_root() / epi_path)
        self.date_tag = str(date_tag)
        self.device = str(device)
        self.action_hold_days = int(action_hold_days)
        self.max_seq_length = int(max_seq_length)
        self.intervention_mode = str(intervention_mode or "binary_threshold")

        self._envs: Dict[str, EpiABMWeeklyEnv] = {}
        self._snapshots: Dict[
            Tuple[str, int, int], List[Optional[Dict[str, Any]]]
        ] = {}
        self._trajectory_cache: Dict[Tuple[str, int, int], List[Tuple[float, Dict[str, np.ndarray]]]] = {}
        self._snapshot_days: Optional[set] = None

    def configure_snapshot_days(self, days: Optional[Iterable[int]]) -> None:
        """Keep only requested day-start snapshots, plus day zero.

        Evaluation with a fixed decision day only needs that anchor. Retaining
        every daily GPU snapshot can otherwise consume most of a 24 GiB card.
        """
        normalized = None
        if days is not None:
            normalized = {0}
            normalized.update(int(day) for day in days)
            invalid = [
                day for day in normalized
                if day < 0 or day > self.max_seq_length
            ]
            if invalid:
                raise ValueError(
                    f"Snapshot days must be in 0..{self.max_seq_length}, got {invalid}"
                )
        if normalized != self._snapshot_days:
            self._snapshots.clear()
            self._trajectory_cache.clear()
            self._snapshot_days = normalized

    def get_env(self, county: Any) -> EpiABMWeeklyEnv:
        county = normalize_county_id(county)
        env = self._envs.get(county)
        if env is None:
            env = EpiABMWeeklyEnv(
                county=county,
                date_tag=self.date_tag,
                epi_root=self.epi_root,
                device=self.device,
                action_hold_days=self.action_hold_days,
                num_steps=self.max_seq_length,
                intervention_mode=self.intervention_mode,
            )
            self._envs[county] = env
        return env

    def release_county(self, county: Any) -> None:
        """Drop cached runner and snapshots for a county after offline materialization."""
        county = normalize_county_id(county)
        self._envs.pop(county, None)
        for key in [key for key in self._snapshots if key[0] == county]:
            self._snapshots.pop(key, None)
        for key in [key for key in self._trajectory_cache if key[0] == county]:
            self._trajectory_cache.pop(key, None)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def precompute_behavior(
        self,
        *,
        county: Any,
        episode_id: int,
        seed: int,
        actions: np.ndarray,
    ) -> List[Tuple[float, Dict[str, np.ndarray]]]:
        county = normalize_county_id(county)
        key = (county, int(episode_id), int(seed))
        if key in self._snapshots:
            return self._trajectory_cache[key]

        actions = np.asarray(actions, dtype=np.float32)
        if actions.ndim != 2:
            raise ValueError(f"behavior actions must have shape [T, A], got {actions.shape}.")
        if actions.shape[0] < self.max_seq_length:
            raise ValueError(
                f"behavior actions for county={county} episode={episode_id} have "
                f"{actions.shape[0]} days, expected at least {self.max_seq_length}."
            )

        env = self.get_env(county)
        env.reset(decision_day=0, seed=int(seed))

        snapshots: List[Optional[Dict[str, Any]]] = [
            env.snapshot()
            if self._snapshot_days is None or 0 in self._snapshot_days
            else None
        ]
        trajectory: List[Tuple[float, Dict[str, np.ndarray]]] = []
        for day in range(self.max_seq_length):
            y, agg = env.step_day(actions[day])
            trajectory.append((float(y), {k: np.asarray(v).copy() for k, v in agg.items()}))
            snapshot_day = day + 1
            snapshots.append(
                env.snapshot()
                if self._snapshot_days is None or snapshot_day in self._snapshot_days
                else None
            )

        self._snapshots[key] = snapshots
        self._trajectory_cache[key] = trajectory
        return trajectory

    def _common_behavior_prefix(
        self,
        *,
        decision_day: int,
        burn_in_actions: np.ndarray,
        behavior_actions: np.ndarray,
    ) -> int:
        max_day = min(int(decision_day), burn_in_actions.shape[0], behavior_actions.shape[0])
        anchor_day = 0
        for day in range(max_day):
            if not np.allclose(burn_in_actions[day], behavior_actions[day], atol=1e-6, rtol=1e-6):
                break
            anchor_day = day + 1
        return anchor_day

    def restore_from_history(
        self,
        *,
        county: Any,
        episode_id: int,
        seed: int,
        decision_day: int,
        burn_in_actions: np.ndarray,
        behavior_actions: np.ndarray,
    ) -> EpiABMWeeklyEnv:
        county = normalize_county_id(county)
        decision_day = int(decision_day)
        behavior_actions = np.asarray(behavior_actions, dtype=np.float32)
        burn_in_actions = np.asarray(burn_in_actions, dtype=np.float32)
        self.precompute_behavior(
            county=county,
            episode_id=int(episode_id),
            seed=int(seed),
            actions=behavior_actions,
        )

        key = (county, int(episode_id), int(seed))
        snapshots = self._snapshots[key]
        if decision_day >= len(snapshots):
            raise ValueError(
                f"decision_day={decision_day} is outside cached snapshot range "
                f"0..{len(snapshots) - 1} for county={county}, episode={episode_id}."
            )

        anchor_day = self._common_behavior_prefix(
            decision_day=decision_day,
            burn_in_actions=burn_in_actions,
            behavior_actions=behavior_actions,
        )
        snapshot_day = anchor_day
        while snapshot_day >= 0 and snapshots[snapshot_day] is None:
            snapshot_day -= 1
        if snapshot_day < 0:
            raise RuntimeError(
                f"No retained snapshot can restore county={county}, "
                f"episode={episode_id}, anchor_day={anchor_day}."
            )
        env = self.get_env(county)
        env.restore_snapshot(deep_clone(snapshots[snapshot_day]))
        for day in range(snapshot_day, decision_day):
            env.step_day(burn_in_actions[day])
        return env

    def rollout_from_history(
        self,
        *,
        county: Any,
        episode_id: int,
        seed: int,
        decision_day: int,
        burn_in_actions: np.ndarray,
        behavior_actions: np.ndarray,
        planned_actions: np.ndarray,
    ) -> List[Tuple[float, Dict[str, np.ndarray]]]:
        env = self.restore_from_history(
            county=county,
            episode_id=episode_id,
            seed=seed,
            decision_day=decision_day,
            burn_in_actions=burn_in_actions,
            behavior_actions=behavior_actions,
        )
        planned_actions = np.asarray(planned_actions, dtype=np.float32)
        trajectory: List[Tuple[float, Dict[str, np.ndarray]]] = []
        for action in planned_actions:
            y, agg = env.step_day(action)
            trajectory.append((float(y), {k: np.asarray(v).copy() for k, v in agg.items()}))
        return trajectory
