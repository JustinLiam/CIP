"""Local empirical action-support utilities for IQL planner diagnostics."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch


def policy_to_sim_interval_np(action_policy: np.ndarray, max_action: float) -> np.ndarray:
    """Map policy/tanh action space ``[-max_action, max_action]`` to simulator ``[0, 1]``."""
    if max_action <= 0:
        return np.asarray(action_policy, dtype=np.float32)
    raw = np.asarray(action_policy, dtype=np.float32)
    return np.clip((raw + float(max_action)) / (2.0 * float(max_action)), 0.0, 1.0).astype(np.float32)


def sim_to_policy_interval_np(action_sim: np.ndarray, max_action: float) -> np.ndarray:
    """Map simulator action space ``[0, 1]`` to policy/tanh ``[-max_action, max_action]``."""
    raw = np.clip(np.asarray(action_sim, dtype=np.float32), 0.0, 1.0)
    if max_action <= 0:
        return raw
    return ((2.0 * raw - 1.0) * float(max_action)).astype(np.float32)


def sim_to_policy_interval_torch(action_sim: torch.Tensor, max_action: float) -> torch.Tensor:
    """Torch equivalent of :func:`sim_to_policy_interval_np`."""
    action_sim = torch.clamp(action_sim, 0.0, 1.0)
    if max_action <= 0:
        return action_sim
    return (2.0 * action_sim - 1.0) * float(max_action)


def parse_iql_state_array(
    states: np.ndarray,
    *,
    z_dim: int,
    output_dim: int,
    action_dim: int,
) -> Dict[str, np.ndarray]:
    """
    Parse IQL states built as ``concat(Z_t, Y_target, delta_t_norm, previous_action_policy)``.
    """
    states = np.asarray(states, dtype=np.float32)
    if states.ndim != 2:
        raise ValueError(f"states must be 2D, got shape={states.shape}")
    expected = int(z_dim) + int(output_dim) + 1 + int(action_dim)
    if states.shape[1] != expected:
        raise ValueError(
            f"IQL state dimension mismatch: got {states.shape[1]}, expected {expected} "
            f"(z_dim={z_dim}, output_dim={output_dim}, action_dim={action_dim})."
        )
    z_end = int(z_dim)
    y_end = z_end + int(output_dim)
    delta_end = y_end + 1
    return {
        "z": states[:, :z_end],
        "y_target": states[:, z_end:y_end],
        "delta": states[:, y_end:delta_end],
        "previous_action_policy": states[:, delta_end:],
    }


def build_support_context_array(
    z: np.ndarray,
    previous_action_sim: np.ndarray,
    delta: Optional[np.ndarray],
    *,
    include_delta: bool,
) -> np.ndarray:
    """Build local behavior-policy context, deliberately excluding target outcomes."""
    z = np.asarray(z, dtype=np.float32)
    previous_action_sim = np.asarray(previous_action_sim, dtype=np.float32)
    parts = [z, previous_action_sim]
    if include_delta:
        if delta is None:
            raise ValueError("delta must be provided when include_delta=True")
        d = np.asarray(delta, dtype=np.float32)
        if d.ndim == 1:
            d = d[:, None]
        parts.append(d)
    return np.concatenate(parts, axis=1).astype(np.float32)


def make_context_weight_vector(
    *,
    z_dim: int,
    action_dim: int,
    include_delta: bool,
    z_weight: float,
    prev_action_weight: float,
    delta_weight: float,
) -> np.ndarray:
    parts = [
        np.full(int(z_dim), float(z_weight), dtype=np.float32),
        np.full(int(action_dim), float(prev_action_weight), dtype=np.float32),
    ]
    if include_delta:
        parts.append(np.asarray([float(delta_weight)], dtype=np.float32))
    return np.concatenate(parts, axis=0)


def deduplicate_context_actions(
    context_raw: np.ndarray,
    behavior_actions_sim: np.ndarray,
    *,
    decimals: int = 6,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Deduplicate rounded ``concat(context, behavior_action)`` rows."""
    context_raw = np.asarray(context_raw, dtype=np.float32)
    behavior_actions_sim = np.asarray(behavior_actions_sim, dtype=np.float32)
    key = np.round(np.concatenate([context_raw, behavior_actions_sim], axis=1), int(decimals))
    _, first_idx = np.unique(key, axis=0, return_index=True)
    keep = np.sort(first_idx)
    return context_raw[keep], behavior_actions_sim[keep], keep.astype(np.int64)


@dataclass
class SupportIndex:
    """KNN index over local behavior-policy contexts with simulator-space actions."""

    context_raw: np.ndarray
    behavior_actions_sim: np.ndarray
    z_dim: int
    action_dim: int
    include_delta: bool
    context_mean: np.ndarray
    context_std: np.ndarray
    context_weights: np.ndarray
    k: int
    size_before_dedup: int
    size_after_dedup: int
    deduplicate: bool
    used_sklearn: bool = False
    _nn: Any = None
    _context_scaled: Any = None

    @classmethod
    def from_iql_transitions(
        cls,
        transitions: Dict[str, np.ndarray],
        *,
        z_dim: int,
        output_dim: int,
        action_dim: int,
        max_action: float,
        include_delta: bool,
        k: int,
        z_weight: float,
        prev_action_weight: float,
        delta_weight: float,
        deduplicate: bool = True,
        dedup_decimals: int = 6,
    ) -> "SupportIndex":
        parsed = parse_iql_state_array(
            transitions["states"],
            z_dim=z_dim,
            output_dim=output_dim,
            action_dim=action_dim,
        )
        prev_action_sim = policy_to_sim_interval_np(parsed["previous_action_policy"], max_action)
        behavior_actions_sim = policy_to_sim_interval_np(transitions["actions"], max_action)
        context_raw = build_support_context_array(
            parsed["z"],
            prev_action_sim,
            parsed["delta"],
            include_delta=include_delta,
        )
        before = int(context_raw.shape[0])
        if deduplicate:
            context_raw, behavior_actions_sim, _ = deduplicate_context_actions(
                context_raw,
                behavior_actions_sim,
                decimals=dedup_decimals,
            )
        return cls.from_arrays(
            context_raw=context_raw,
            behavior_actions_sim=behavior_actions_sim,
            z_dim=z_dim,
            action_dim=action_dim,
            include_delta=include_delta,
            k=k,
            z_weight=z_weight,
            prev_action_weight=prev_action_weight,
            delta_weight=delta_weight,
            size_before_dedup=before,
            deduplicate=deduplicate,
        )

    @classmethod
    def from_arrays(
        cls,
        *,
        context_raw: np.ndarray,
        behavior_actions_sim: np.ndarray,
        z_dim: int,
        action_dim: int,
        include_delta: bool,
        k: int,
        z_weight: float,
        prev_action_weight: float,
        delta_weight: float,
        size_before_dedup: Optional[int] = None,
        deduplicate: bool = True,
        context_mean: Optional[np.ndarray] = None,
        context_std: Optional[np.ndarray] = None,
    ) -> "SupportIndex":
        context_raw = np.asarray(context_raw, dtype=np.float32)
        behavior_actions_sim = np.asarray(behavior_actions_sim, dtype=np.float32)
        if context_raw.ndim != 2 or behavior_actions_sim.ndim != 2:
            raise ValueError("context_raw and behavior_actions_sim must be 2D arrays.")
        if context_raw.shape[0] != behavior_actions_sim.shape[0]:
            raise ValueError("context_raw and behavior_actions_sim must have the same number of rows.")
        if behavior_actions_sim.shape[1] != int(action_dim):
            raise ValueError(
                f"behavior action dimension mismatch: got {behavior_actions_sim.shape[1]}, expected {action_dim}"
            )
        if context_raw.shape[0] == 0:
            raise ValueError("Cannot build SupportIndex from zero support rows.")
        if context_mean is None:
            context_mean = context_raw.mean(axis=0)
        if context_std is None:
            context_std = context_raw.std(axis=0)
        context_std = np.asarray(context_std, dtype=np.float32)
        context_std = np.where(context_std < 1e-6, 1.0, context_std).astype(np.float32)
        context_weights = make_context_weight_vector(
            z_dim=z_dim,
            action_dim=action_dim,
            include_delta=include_delta,
            z_weight=z_weight,
            prev_action_weight=prev_action_weight,
            delta_weight=delta_weight,
        )
        if context_weights.shape[0] != context_raw.shape[1]:
            raise ValueError(
                f"context weight dimension mismatch: got {context_weights.shape[0]}, "
                f"expected {context_raw.shape[1]}"
            )
        obj = cls(
            context_raw=context_raw.astype(np.float32),
            behavior_actions_sim=behavior_actions_sim.astype(np.float32),
            z_dim=int(z_dim),
            action_dim=int(action_dim),
            include_delta=bool(include_delta),
            context_mean=np.asarray(context_mean, dtype=np.float32),
            context_std=context_std,
            context_weights=context_weights.astype(np.float32),
            k=int(k),
            size_before_dedup=int(size_before_dedup or context_raw.shape[0]),
            size_after_dedup=int(context_raw.shape[0]),
            deduplicate=bool(deduplicate),
        )
        obj.fit()
        return obj

    def scaled_context(self, context_raw: np.ndarray) -> np.ndarray:
        context_raw = np.asarray(context_raw, dtype=np.float32)
        if context_raw.ndim == 1:
            context_raw = context_raw[None, :]
        if context_raw.shape[1] != self.context_raw.shape[1]:
            raise ValueError(f"context dimension mismatch: got {context_raw.shape[1]}, expected {self.context_raw.shape[1]}")
        return ((context_raw - self.context_mean) / self.context_std * self.context_weights).astype(np.float32)

    def fit(self) -> None:
        scaled = self.scaled_context(self.context_raw)
        self._context_scaled = scaled
        try:
            from sklearn.neighbors import NearestNeighbors

            n_neighbors = min(max(1, int(self.k)), int(scaled.shape[0]))
            self._nn = NearestNeighbors(n_neighbors=n_neighbors, algorithm="auto", metric="euclidean")
            self._nn.fit(scaled)
            self.used_sklearn = True
        except Exception:
            self._nn = None
            self.used_sklearn = False

    def build_context(
        self,
        z: np.ndarray,
        previous_action_sim: np.ndarray,
        delta: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        return build_support_context_array(
            z,
            previous_action_sim,
            delta,
            include_delta=self.include_delta,
        )

    def query(self, context_raw: np.ndarray, k: Optional[int] = None) -> Dict[str, np.ndarray]:
        context_raw = np.asarray(context_raw, dtype=np.float32)
        if context_raw.ndim == 1:
            context_raw = context_raw[None, :]
        query_k = min(max(1, int(k or self.k)), int(self.context_raw.shape[0]))
        query_scaled = self.scaled_context(context_raw)
        train_scaled = self._context_scaled
        if train_scaled is None:
            train_scaled = self.scaled_context(self.context_raw)
            self._context_scaled = train_scaled
        if self._nn is not None and query_k <= int(self._nn.n_neighbors):
            distances, indices = self._nn.kneighbors(query_scaled, n_neighbors=query_k)
        else:
            distances = np.empty((query_scaled.shape[0], query_k), dtype=np.float32)
            indices = np.empty((query_scaled.shape[0], query_k), dtype=np.int64)
            for i, row in enumerate(query_scaled):
                d = np.linalg.norm(train_scaled - row[None, :], axis=1)
                idx = np.argpartition(d, query_k - 1)[:query_k]
                idx = idx[np.argsort(d[idx])]
                distances[i] = d[idx]
                indices[i] = idx
        return {
            "indices": indices.astype(np.int64),
            "distances": distances.astype(np.float32),
            "actions": self.behavior_actions_sim[indices].astype(np.float32),
        }


def local_action_quantiles(local_actions: np.ndarray) -> np.ndarray:
    """Return per-dimension quantiles [0.05, 0.25, 0.50, 0.75, 0.95]."""
    return np.quantile(np.asarray(local_actions, dtype=np.float32), [0.05, 0.25, 0.50, 0.75, 0.95], axis=0).astype(np.float32)


def local_support_metrics(
    local_actions: np.ndarray,
    candidate_action: np.ndarray,
    *,
    r: float,
    eta: float,
) -> Dict[str, Any]:
    """Compute empirical small-ball support for a candidate simulator-space action."""
    local_actions = np.asarray(local_actions, dtype=np.float32)
    candidate_action = np.asarray(candidate_action, dtype=np.float32).reshape(1, -1)
    distances = np.linalg.norm(local_actions - candidate_action, axis=1)
    mass = float(np.mean(distances <= float(r)))
    quant = local_action_quantiles(local_actions)
    return {
        "small_ball_mass": mass,
        "is_ood": bool(mass < float(eta)),
        "knn_min_distance": float(np.min(distances)) if distances.size else float("nan"),
        "quantiles": quant,
        "local_median": quant[2],
    }


def batch_local_support_metrics(
    local_actions: np.ndarray,
    candidate_actions: np.ndarray,
    *,
    r: float,
    eta: float,
) -> Dict[str, np.ndarray]:
    """Vectorized local support metrics for ``[B,K,A]`` local actions and ``[B,A]`` candidates."""
    local_actions = np.asarray(local_actions, dtype=np.float32)
    candidate_actions = np.asarray(candidate_actions, dtype=np.float32)
    distances = np.linalg.norm(local_actions - candidate_actions[:, None, :], axis=2)
    masses = np.mean(distances <= float(r), axis=1).astype(np.float32)
    min_dist = np.min(distances, axis=1).astype(np.float32)
    quantiles = np.quantile(local_actions, [0.05, 0.25, 0.50, 0.75, 0.95], axis=1)
    quantiles = np.transpose(quantiles, (1, 0, 2)).astype(np.float32)
    return {
        "small_ball_mass": masses,
        "is_ood": (masses < float(eta)),
        "knn_min_distance": min_dist,
        "quantiles": quantiles,
        "local_median": quantiles[:, 2, :],
    }


def make_action_grid_sim(action_dim: int, grid_points: int, device: str | torch.device, dtype: torch.dtype) -> torch.Tensor:
    points = max(2, int(grid_points))
    vals = torch.linspace(0.0, 1.0, points, device=device, dtype=dtype)
    return torch.cartesian_prod(*([vals] * int(action_dim))).contiguous()


@torch.no_grad()
def q_grid_argmax_action(
    planner,
    obs: torch.Tensor,
    action_grid_sim: torch.Tensor,
    max_action: float,
    *,
    device: str | torch.device,
    q_chunk_size: int = 8192,
) -> torch.Tensor:
    """
    Return critic-greedy action in simulator space by evaluating a full action grid in chunks.
    """
    obs = obs.to(device)
    action_grid_sim = action_grid_sim.to(device=device, dtype=obs.dtype)
    action_grid_policy = sim_to_policy_interval_torch(action_grid_sim, max_action)
    bsz = obs.size(0)
    grid_n = action_grid_sim.size(0)
    chunk = max(1, int(q_chunk_size))
    best_q = torch.full((bsz,), -torch.inf, device=obs.device, dtype=obs.dtype)
    best_idx = torch.zeros((bsz,), device=obs.device, dtype=torch.long)
    for start in range(0, grid_n, chunk):
        end = min(start + chunk, grid_n)
        g = end - start
        obs_rep = obs.unsqueeze(1).expand(bsz, g, obs.size(-1)).reshape(bsz * g, obs.size(-1))
        act_rep = action_grid_policy[start:end].unsqueeze(0).expand(bsz, g, action_grid_policy.size(-1))
        act_rep = act_rep.reshape(bsz * g, action_grid_policy.size(-1))
        q_values = planner.qf(obs_rep, act_rep).view(bsz, g)
        local_q, local_idx = torch.max(q_values, dim=1)
        update = local_q > best_q
        best_q = torch.where(update, local_q, best_q)
        best_idx = torch.where(update, local_idx + start, best_idx)
    return action_grid_sim[best_idx].detach()
