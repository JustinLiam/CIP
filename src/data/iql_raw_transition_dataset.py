"""
Raw IQL transitions for EM training: store H_t / H_{t+1} without precomputed states.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from src.data.ct_transition_dataset import _build_H_slice, _collate_pad_H
from src.data.iql_dataset_builder import (
    dataset_actions_to_tanh_policy_space,
    horizon_remaining_distance_reward_np,
    huber_loss_np,
)


def _coerce_target_horizons(target_horizons: Optional[Iterable[int]], max_tau: float) -> List[int]:
    if target_horizons is None:
        return list(range(1, int(max_tau) + 1))
    if isinstance(target_horizons, str):
        raw = target_horizons.strip()
        if raw.startswith("["):
            target_horizons = ast.literal_eval(raw)
        else:
            target_horizons = [x.strip() for x in raw.split(",") if x.strip()]
    out = sorted({int(h) for h in target_horizons if int(h) > 0})
    if not out:
        raise ValueError("target_horizons must contain at least one positive horizon.")
    return out


def _sample_target_indices(
    *,
    t: int,
    last_idx: int,
    max_tau: float,
    samples_per_transition: int,
    rng: np.random.RandomState,
    target_sampling: str,
    target_horizons: Optional[Iterable[int]],
) -> List[int]:
    hi = min(t + int(max_tau), last_idx)
    if hi <= t:
        return []

    mode = str(target_sampling).strip().lower()
    if mode in ("random", "random_future", "future"):
        return [int(rng.randint(low=t + 1, high=hi + 1)) for _ in range(samples_per_transition)]

    if mode in ("horizon_aligned", "aligned", "fixed_horizon"):
        horizons = [
            h
            for h in _coerce_target_horizons(target_horizons, max_tau)
            if h <= int(max_tau) and t + h <= last_idx
        ]
        if not horizons:
            return []
        replace = len(horizons) < samples_per_transition
        size = samples_per_transition if replace else min(samples_per_transition, len(horizons))
        chosen = rng.choice(horizons, size=size, replace=replace)
        return [t + int(h) for h in sorted(np.asarray(chosen, dtype=np.int64).tolist())]

    raise ValueError(
        f"Unknown target_sampling={target_sampling!r}; expected 'random_future' or 'horizon_aligned'."
    )


@dataclass
class IQLRawTransition:
    """One transition; tensors are 1-sample before collate."""

    patient_idx: int
    t: int
    t_target: int
    H_t: Dict[str, torch.Tensor]
    H_t_next: Dict[str, torch.Tensor]
    action: torch.Tensor
    reward: float
    done: float
    y_target: torch.Tensor
    delta_t_norm: float
    delta_t_next_norm: float
    a_prev_tanh: torch.Tensor


@dataclass
class IQLRawBatch:
    H_t: Dict[str, torch.Tensor]
    H_t_next: Dict[str, torch.Tensor]
    action: torch.Tensor
    reward: torch.Tensor
    done: torch.Tensor
    y_target: torch.Tensor
    delta_t_norm: torch.Tensor
    delta_t_next_norm: torch.Tensor
    a_prev_tanh: torch.Tensor


def build_iql_raw_transitions(
    data: Dict[str, np.ndarray],
    *,
    reward_type: str = "progress",
    max_patients: Optional[int] = None,
    max_action: float = 1.0,
    dataset_actions_unit_interval: bool = True,
    max_tau: float = 12.0,
    reward_clip: float = 3.0,
    reward_scale: str = "auto",
    reward_huber_delta: float = 1.0,
    samples_per_transition: int = 1,
    target_sampling: str = "horizon_aligned",
    target_horizons: Optional[Iterable[int]] = None,
    horizon_terminal_done: bool = True,
    decision_interval_days: int = 1,
    seed: Optional[int] = None,
) -> List[IQLRawTransition]:
    """
    Same sampling as ``build_iql_transitions_from_ct`` but without encoder;
    stores history dicts for online encoding in M-step.
    """
    samples_per_transition = int(samples_per_transition)
    if samples_per_transition < 1:
        raise ValueError("samples_per_transition must be >= 1.")
    rng = np.random.RandomState(seed) if seed is not None else np.random

    n_patients = data["current_treatments"].shape[0]
    if max_patients is not None:
        n_patients = min(n_patients, max_patients)
    if max_tau <= 0:
        raise ValueError("max_tau must be positive.")
    decision_interval_days = max(1, int(decision_interval_days))

    transitions: List[IQLRawTransition] = []
    for i in range(n_patients):
        active = data["active_entries"][i]
        length = int(active.sum())
        if length < 3:
            continue
        last_idx = length - 1

        for t in range(1, length - 1):
            if decision_interval_days > 1:
                if "sim_day" in data:
                    day_value = int(np.asarray(data["sim_day"][i, t]).reshape(-1)[0])
                else:
                    day_value = int(t)
                if day_value % decision_interval_days != 0:
                    continue

            target_indices = _sample_target_indices(
                t=t,
                last_idx=last_idx,
                max_tau=max_tau,
                samples_per_transition=samples_per_transition,
                rng=rng,
                target_sampling=target_sampling,
                target_horizons=target_horizons,
            )
            if not target_indices:
                continue

            H_t = _build_H_slice(data, i, t + 1)
            H_next = _build_H_slice(data, i, t + 2)

            a_prev_raw = data["current_treatments"][i, t - 1, :].astype(np.float32)
            if dataset_actions_unit_interval:
                a_prev_feat = dataset_actions_to_tanh_policy_space(a_prev_raw, max_action)
            else:
                a_prev_feat = a_prev_raw

            a = data["current_treatments"][i, t, :].astype(np.float32)
            if dataset_actions_unit_interval:
                a_policy = dataset_actions_to_tanh_policy_space(a, max_action)
            else:
                a_policy = a

            y_next = data["outputs"][i, t + 1, :].astype(np.float32)
            y_cur = data["outputs"][i, t, :].astype(np.float32)

            for t_target in target_indices:
                y_target_np = data["outputs"][i, t_target, :].astype(np.float32)
                delta_t_norm = max(0.0, float(t_target - t) / max_tau)
                delta_t_next_norm = max(0.0, float(t_target - t - 1) / max_tau)
                done = 1.0 if (t + 1) >= last_idx else 0.0
                if horizon_terminal_done and (t + 1) >= t_target:
                    done = 1.0

                if reward_type == "negative_outcome_mse":
                    r = -float(np.mean((y_next - y_target_np) ** 2))
                elif reward_type in ("negative_outcome_huber", "huber", "smooth_l1"):
                    r = -float(np.mean(huber_loss_np(y_next - y_target_np, reward_huber_delta)))
                elif reward_type == "negative_outcome":
                    r = -float(np.mean(np.abs(y_next - y_target_np)))
                elif reward_type in ("horizon_remaining", "remaining_distance", "horizon_remaining_distance"):
                    r = horizon_remaining_distance_reward_np(
                        y_next,
                        y_target_np,
                        t=t,
                        t_target=t_target,
                    )
                elif reward_type == "progress":
                    d_cur = float(np.mean(np.abs(y_cur - y_target_np)))
                    d_nxt = float(np.mean(np.abs(y_next - y_target_np)))
                    r = d_cur - d_nxt
                else:
                    r = -float(np.mean(np.abs(y_next - y_target_np)))

                if reward_clip is not None and float(reward_clip) > 0.0:
                    c = float(reward_clip)
                    r = max(-c, min(c, r))

                transitions.append(
                    IQLRawTransition(
                        patient_idx=i,
                        t=t,
                        t_target=t_target,
                        H_t=H_t,
                        H_t_next=H_next,
                        action=torch.tensor(a_policy, dtype=torch.float32),
                        reward=float(r),
                        done=float(done),
                        y_target=torch.tensor(y_target_np, dtype=torch.float32),
                        delta_t_norm=delta_t_norm,
                        delta_t_next_norm=delta_t_next_norm,
                        a_prev_tanh=torch.tensor(a_prev_feat, dtype=torch.float32),
                    )
                )

    if str(reward_scale).lower() == "auto" and transitions:
        rewards = np.array([tr.reward for tr in transitions], dtype=np.float32)
        r_std = float(rewards.std()) + 1e-8
        for tr in transitions:
            tr.reward = float(tr.reward / r_std)

    import logging

    logging.getLogger(__name__).info(
        "Built %d raw IQL transitions (no precomputed states, samples_per_transition=%d, "
        "target_sampling=%s, target_horizons=%s, horizon_terminal_done=%s, "
        "decision_interval_days=%d).",
        len(transitions),
        samples_per_transition,
        target_sampling,
        _coerce_target_horizons(target_horizons, max_tau)
        if str(target_sampling).strip().lower() in ("horizon_aligned", "aligned", "fixed_horizon")
        else None,
        horizon_terminal_done,
        decision_interval_days,
    )
    return transitions


def collate_iql_raw_batch(samples: List[IQLRawTransition]) -> IQLRawBatch:
    dtype = samples[0].H_t["prev_treatments"].dtype
    H_t = _collate_pad_H([{"H_t": s.H_t} for s in samples], "H_t", dtype)
    H_t_next = _collate_pad_H([{"H_t_next": s.H_t_next} for s in samples], "H_t_next", dtype)
    return IQLRawBatch(
        H_t=H_t,
        H_t_next=H_t_next,
        action=torch.stack([s.action for s in samples], dim=0),
        reward=torch.tensor([s.reward for s in samples], dtype=dtype),
        done=torch.tensor([s.done for s in samples], dtype=dtype),
        y_target=torch.stack([s.y_target for s in samples], dim=0),
        delta_t_norm=torch.tensor(
            [[s.delta_t_norm] for s in samples], dtype=dtype
        ),
        delta_t_next_norm=torch.tensor(
            [[s.delta_t_next_norm] for s in samples], dtype=dtype
        ),
        a_prev_tanh=torch.stack([s.a_prev_tanh for s in samples], dim=0),
    )


class IQLRawTransitionDataset(Dataset):
    def __init__(self, transitions: List[IQLRawTransition]):
        self.transitions = transitions

    def __len__(self) -> int:
        return len(self.transitions)

    def __getitem__(self, idx: int) -> IQLRawTransition:
        return self.transitions[idx]


class IQLRawReplayBuffer:
    """Sample collated batches for M-step."""

    def __init__(self, transitions: List[IQLRawTransition], device: str = "cpu"):
        self.transitions = transitions
        self.device = device
        self.size = len(transitions)

    def sample(self, batch_size: int) -> IQLRawBatch:
        if self.size <= 0:
            raise ValueError("Cannot sample from an empty IQLRawReplayBuffer.")
        idx = np.random.randint(0, self.size, size=batch_size)
        batch = collate_iql_raw_batch([self.transitions[i] for i in idx])
        return _batch_to_device(batch, self.device)


def _batch_to_device(batch: IQLRawBatch, device: str) -> IQLRawBatch:
    def _mv_h(H: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return {k: v.to(device) for k, v in H.items()}

    return IQLRawBatch(
        H_t=_mv_h(batch.H_t),
        H_t_next=_mv_h(batch.H_t_next),
        action=batch.action.to(device),
        reward=batch.reward.to(device),
        done=batch.done.to(device),
        y_target=batch.y_target.to(device),
        delta_t_norm=batch.delta_t_norm.to(device),
        delta_t_next_norm=batch.delta_t_next_norm.to(device),
        a_prev_tanh=batch.a_prev_tanh.to(device),
    )
