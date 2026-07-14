import ast
from typing import Dict, Iterable, List, Optional

import numpy as np
import torch


def dataset_actions_to_tanh_policy_space(actions: np.ndarray, max_action: float) -> np.ndarray:
    """
    Map simulator/data treatments in [0, 1] to the bounded policy-action range
    used by GaussianPolicy / DeterministicPolicy after tanh scaling:
    [-max_action, max_action].

    Inverse of eval mapping: (a_policy + max_action) / (2 * max_action) -> sim [0, 1].
    """
    if max_action <= 0:
        return actions.astype(np.float32)
    a = np.clip(actions.astype(np.float32), 0.0, 1.0)
    return ((2.0 * a - 1.0) * float(max_action)).astype(np.float32)


def huber_loss_np(error: np.ndarray, delta: float) -> np.ndarray:
    """Elementwise Huber loss in outcome space."""
    d = float(delta)
    if d <= 0:
        raise ValueError("huber delta must be positive.")
    abs_error = np.abs(error)
    return np.where(abs_error <= d, 0.5 * error ** 2, d * (abs_error - 0.5 * d))


def horizon_remaining_distance_reward_np(
    y_next: np.ndarray,
    y_target: np.ndarray,
    *,
    t: int,
    t_target: int,
) -> float:
    """Distance-to-goal reward scaled by remaining horizon after the transition.

    ``t`` is the processed-row decision index. ``t_target`` is the processed-row
    output index for the goal. Thus horizon 1 has ``t_target == t`` and no
    bootstrap horizon remains after applying ``current_treatments[t]``.
    """
    h_next = max(int(t_target) - int(t), 0)
    d_next = float(np.sqrt(np.mean((y_next - y_target) ** 2)))
    return -d_next / np.sqrt(float(h_next + 1))


def align_h_t_static_to_history(H_t: Dict) -> Dict:
    """
    Slice batched static_features [B, T_max, d] to match prev_treatments length T_hist
    so ct_hidden_history attention streams align (same fix as _static_for_prefix).
    """
    if "static_features" not in H_t or "prev_treatments" not in H_t:
        return H_t
    sf = H_t["static_features"]
    if not isinstance(sf, torch.Tensor) or sf.dim() != 3:
        return H_t
    t_hist = H_t["prev_treatments"].size(1)
    if sf.size(1) <= t_hist:
        return H_t
    out = dict(H_t)
    out["static_features"] = sf[:, :t_hist, :].contiguous()
    return out


def _to_torch(arr: np.ndarray, device: str) -> torch.Tensor:
    return torch.tensor(arr, dtype=torch.float32, device=device)


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
    target_sampling: str,
    target_horizons: Optional[Iterable[int]],
) -> List[int]:
    hi = min(t + int(max_tau) - 1, last_idx)
    if hi < t:
        return []

    mode = str(target_sampling).strip().lower()
    if mode in ("random", "random_future", "future"):
        return [int(np.random.randint(low=t, high=hi + 1)) for _ in range(samples_per_transition)]

    if mode in ("horizon_aligned", "aligned", "fixed_horizon"):
        horizons = [
            h
            for h in _coerce_target_horizons(target_horizons, max_tau)
            if h <= int(max_tau) and t + h - 1 <= last_idx
        ]
        if not horizons:
            return []
        replace = len(horizons) < samples_per_transition
        size = samples_per_transition if replace else min(samples_per_transition, len(horizons))
        chosen = np.random.choice(horizons, size=size, replace=replace)
        return [t + int(h) - 1 for h in sorted(np.asarray(chosen, dtype=np.int64).tolist())]

    raise ValueError(
        f"Unknown target_sampling={target_sampling!r}; expected 'random_future' or 'horizon_aligned'."
    )


def _static_for_prefix(data: Dict[str, np.ndarray], patient_idx: int, prefix_len: int) -> np.ndarray:
    """
    Align static_features time dimension with other H_t fields (prefix_len steps).

    After repeat_static, static_features is often [N, max_seq, static_dim] while
    prev_treatments are sliced to [:prefix_len]. If we pass full-length static
    into ct_hidden_history, the covariate stream (x_v) has a different T than
    treatment/outcome streams -> attention mask / scores shape mismatch.
    """
    sf = data["static_features"][patient_idx : patient_idx + 1]
    if sf.ndim == 2:
        return sf
    if sf.ndim == 3:
        return sf[:, :prefix_len, :]
    return sf


def build_iql_transitions_from_ct(
    data: Dict[str, np.ndarray],
    inference_model,
    device: str = "cuda",
    reward_type: str = "progress",
    max_patients: Optional[int] = None,
    max_action: float = 1.0,
    dataset_actions_unit_interval: bool = True,
    max_tau: float = 12.0,
    reward_clip: float = 1.0,
    reward_scale: str = "none",
    reward_huber_delta: float = 1.0,
    samples_per_transition: int = 1,
    target_sampling: str = "horizon_aligned",
    target_horizons: Optional[Iterable[int]] = None,
    horizon_terminal_done: bool = True,
) -> Dict[str, np.ndarray]:
    """
    Build offline RL transitions (s, a, r, s', done) from longitudinal dataset.
    State s is ``concat(Z_t, Y_target, delta_t_norm, a_{t-1})`` (horizon-aware HER + previous action
    in policy space) where Z_t comes from ``inference_model.ct_hidden_history(H_t)``.
    Next state uses ``a_t`` (the same vector as the transition action a) as the previous-action
    channel at time t+1.

    If ``dataset_actions_unit_interval``, actions from data/simulator space [0, 1] are mapped to
    [-max_action, max_action] to match Tanh-bounded IQL policies.

    ``max_tau`` scales time-to-go; must match evaluation (``exp.max_tau``).

    The processed-row contract is shared by Tumor/MIMIC/EpiCF:
    ``prev_outputs[t]`` / ``prev_treatments[t]`` encode the state before the
    logged action, ``current_treatments[t]`` is that action, and ``outputs[t]``
    is the immediate post-action outcome. Horizon ``h`` therefore targets
    ``outputs[t + h - 1]``.

    Reward design (the 2026-04 fix for q_loss spikes + long-τ regressions):

    ``reward_type`` selects the per-step scalar:
      * ``"negative_outcome_mse"``  (legacy)      r = -mean((y_next - y_target)^2)
      * ``"negative_outcome_huber"`` (Huber)      r = -mean(Huber(y_next - y_target))
      * ``"negative_outcome"``      (L1)          r = -mean(|y_next - y_target|)
      * ``"horizon_remaining"``                   r = -RMSE(y_next, y_target) / sqrt(h_next + 1)
      * ``"progress"`` (RECOMMENDED, new default) r = |y_cur - y_target| - |y_next - y_target|
        Progress reward. Σ_t r_t telescopes to ``|y_t - y_target| - |y_T - y_target|``, so
        V*(s_t) is the expected *remaining* distance improvement under the policy —
        the exact quantity a long-horizon planner wants. Action-sensitive, zero-mean
        on noise, naturally bounded by |Δy|.

    ``reward_clip`` (float, >=0; 0 disables): hard-clip r to ``[-c, +c]``. On
    cancer_sim γ=4 the raw MSE distribution has q99=0 but min≈-117 (rare simulator
    overflow patients). Without clipping a single outlier in a batch creates TD
    targets of O(100), driving q_loss spikes. Default ``1.0`` preserves > 99% of
    the ``progress`` / L1 signal while killing the heavy tail completely.

    ``reward_scale`` ("none" / "auto"): when "auto", divide all rewards by their
    empirical std after clipping. Standard IQL practice; keeps ``Q`` in ``O(1)``
    so that default ``iql_discount=0.99`` and ``iql_beta=1.0`` work out-of-the-box
    across datasets with different outcome scales.
    """
    inference_model = inference_model.to(device)
    inference_model.eval()

    n_patients = data["current_treatments"].shape[0]
    if max_patients is not None:
        n_patients = min(n_patients, max_patients)

    if max_tau <= 0:
        raise ValueError("max_tau must be positive for horizon-aware IQL transitions.")
    samples_per_transition = max(1, int(samples_per_transition))

    states, actions, rewards, next_states, dones = [], [], [], [], []

    with torch.no_grad():
        for i in range(n_patients):
            # history valid length from active entries
            active = data["active_entries"][i]  # [T, 1]
            length = int(active.sum())
            # need at least three rows: decision row t, next-state row t+1, and
            # an outcome label outputs[t] aligned with current_treatments[t].
            if length < 3:
                continue

            last_idx = length - 1

            for t in range(1, length - 1):
                # Build H_t with history [0, t]
                H_t = {
                    "prev_treatments": _to_torch(data["prev_treatments"][i : i + 1, : t + 1, :], device),
                    "current_treatments": _to_torch(data["current_treatments"][i : i + 1, : t + 1, :], device),
                    "prev_outputs": _to_torch(data["prev_outputs"][i : i + 1, : t + 1, :], device),
                    "outputs": _to_torch(data["outputs"][i : i + 1, : t + 1, :], device),
                    "active_entries": _to_torch(data["active_entries"][i : i + 1, : t + 1, :], device),
                }

                if "current_covariates" in data:
                    H_t["current_covariates"] = _to_torch(data["current_covariates"][i : i + 1, : t + 1, :], device)
                if "vitals" in data:
                    H_t["vitals"] = _to_torch(data["vitals"][i : i + 1, : t + 1, :], device)
                if "static_features" in data:
                    H_t["static_features"] = _to_torch(_static_for_prefix(data, i, t + 1), device)

                z_t, _, _ = inference_model.ct_hidden_history(H_t)
                z_vec = z_t.squeeze(0).detach().cpu().numpy()

                target_indices = _sample_target_indices(
                    t=t,
                    last_idx=last_idx,
                    max_tau=max_tau,
                    samples_per_transition=samples_per_transition,
                    target_sampling=target_sampling,
                    target_horizons=target_horizons,
                )
                if not target_indices:
                    continue

                a_prev_raw = data["current_treatments"][i, t - 1, :].astype(np.float32)
                if dataset_actions_unit_interval:
                    a_prev_feat = dataset_actions_to_tanh_policy_space(a_prev_raw, max_action)
                else:
                    a_prev_feat = a_prev_raw

                a = data["current_treatments"][i, t, :].astype(np.float32)
                if dataset_actions_unit_interval:
                    a = dataset_actions_to_tanh_policy_space(a, max_action)

                # Build H_{t+1} (loop range guarantees t + 1 < length)
                H_next = {
                    "prev_treatments": _to_torch(data["prev_treatments"][i : i + 1, : t + 2, :], device),
                    "current_treatments": _to_torch(data["current_treatments"][i : i + 1, : t + 2, :], device),
                    "prev_outputs": _to_torch(data["prev_outputs"][i : i + 1, : t + 2, :], device),
                    "outputs": _to_torch(data["outputs"][i : i + 1, : t + 2, :], device),
                    "active_entries": _to_torch(data["active_entries"][i : i + 1, : t + 2, :], device),
                }
                if "current_covariates" in data:
                    H_next["current_covariates"] = _to_torch(
                        data["current_covariates"][i : i + 1, : t + 2, :], device
                    )
                if "vitals" in data:
                    H_next["vitals"] = _to_torch(data["vitals"][i : i + 1, : t + 2, :], device)
                if "static_features" in data:
                    H_next["static_features"] = _to_torch(_static_for_prefix(data, i, t + 2), device)

                z_next, _, _ = inference_model.ct_hidden_history(H_next)
                z_next_vec = z_next.squeeze(0).detach().cpu().numpy()
                y_next = data["outputs"][i, t, :].astype(np.float32)
                y_cur = data["prev_outputs"][i, t, :].astype(np.float32)

                for t_target in target_indices:
                    y_target = data["outputs"][i, t_target, :].astype(np.float32)
                    horizon = float(t_target - t + 1)
                    delta_t_norm = max(0.0, horizon / max_tau)
                    delta_t_next_norm = max(0.0, (horizon - 1.0) / max_tau)

                    # At t+1 the "previous action" feature is a_t (behavior action at t), i.e. this transition's a.
                    s_next = np.concatenate(
                        [z_next_vec, y_target, np.array([delta_t_next_norm], dtype=np.float32), a], axis=0
                    )

                    done = 1.0 if (t + 1) >= last_idx else 0.0
                    if horizon_terminal_done and horizon <= 1.0:
                        done = 1.0
                    # See docstring for the reward design rationale. The default
                    # "progress" signal telescopes so V(s) encodes remaining distance
                    # improvement — directly aligned with long-horizon planning.
                    if reward_type == "negative_outcome_mse":
                        r = -float(np.mean((y_next - y_target) ** 2))
                    elif reward_type in ("negative_outcome_huber", "huber", "smooth_l1"):
                        r = -float(np.mean(huber_loss_np(y_next - y_target, reward_huber_delta)))
                    elif reward_type == "negative_outcome":
                        r = -float(np.mean(np.abs(y_next - y_target)))
                    elif reward_type in ("horizon_remaining", "remaining_distance", "horizon_remaining_distance"):
                        r = horizon_remaining_distance_reward_np(y_next, y_target, t=t, t_target=t_target)
                    elif reward_type == "progress":
                        d_cur = float(np.mean(np.abs(y_cur - y_target)))
                        d_nxt = float(np.mean(np.abs(y_next - y_target)))
                        r = d_cur - d_nxt
                    else:
                        r = -float(np.mean(np.abs(y_next - y_target)))

                    if reward_clip is not None and float(reward_clip) > 0.0:
                        c = float(reward_clip)
                        if r > c:
                            r = c
                        elif r < -c:
                            r = -c

                    s = np.concatenate(
                        [z_vec, y_target, np.array([delta_t_norm], dtype=np.float32), a_prev_feat], axis=0
                    )
                    states.append(s)
                    actions.append(a)
                    rewards.append([r])
                    next_states.append(s_next)
                    dones.append([done])
                
    rewards_arr = np.asarray(rewards, dtype=np.float32)
    if str(reward_scale).lower() == "auto":
        r_std = float(rewards_arr.std()) + 1e-8
        rewards_arr = rewards_arr / r_std

    import logging as _logging
    _logger = _logging.getLogger(__name__)
    r_flat = rewards_arr.reshape(-1)
    _logger.info(
        "IQL reward stats | type=%s clip=%s scale=%s | "
        "mean=%.6f std=%.6f min=%.4f max=%.4f "
        "q01=%.4f q50=%.4f q99=%.4f",
        reward_type, reward_clip, reward_scale,
        float(r_flat.mean()), float(r_flat.std()),
        float(r_flat.min()), float(r_flat.max()),
        float(np.quantile(r_flat, 0.01)),
        float(np.quantile(r_flat, 0.50)),
        float(np.quantile(r_flat, 0.99)),
    )

    return {
        "states": np.asarray(states, dtype=np.float32),
        "actions": np.asarray(actions, dtype=np.float32),
        "rewards": rewards_arr,
        "next_states": np.asarray(next_states, dtype=np.float32),
        "dones": np.asarray(dones, dtype=np.float32),
    }
