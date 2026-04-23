from typing import Dict, Optional

import numpy as np
import torch


def dataset_actions_to_tanh_policy_space(actions: np.ndarray, max_action: float) -> np.ndarray:
    """
    Map factual treatments in [0, max_action] (e.g. cancer chemo/radio in [0, 1]) to the same
    range as GaussianPolicy / DeterministicPolicy: tanh then * max_action -> [-max_action, max_action].

    Inverse of eval mapping: (a_policy + max_action) / (2 * max_action) -> sim [0, 1].
    """
    if max_action <= 0:
        return actions.astype(np.float32)
    a = np.clip(actions.astype(np.float32), 0.0, max_action)
    return (2.0 * a - max_action).astype(np.float32)


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
) -> Dict[str, np.ndarray]:
    """
    Build offline RL transitions (s, a, r, s', done) from longitudinal dataset.
    State s is ``concat(Z_t, Y_target, delta_t_norm, a_{t-1})`` (horizon-aware HER + previous action
    in policy space) where Z_t comes from ``inference_model.ct_hidden_history(H_t)``.
    Next state uses ``a_t`` (the same vector as the transition action a) as the previous-action
    channel at time t+1.

    If ``dataset_actions_unit_interval``, actions from data in [0, max_action] are mapped to
    [-max_action, max_action] to match Tanh-bounded IQL policies.

    ``max_tau`` scales time-to-go; must match evaluation (``exp.max_tau``).

    Reward design (the 2026-04 fix for q_loss spikes + long-τ regressions):

    ``reward_type`` selects the per-step scalar:
      * ``"negative_outcome_mse"``  (legacy)      r = -mean((y_{t+1} - y_target)^2)
      * ``"negative_outcome"``      (L1)          r = -mean(|y_{t+1} - y_target|)
      * ``"progress"`` (RECOMMENDED, new default) r = |y_t - y_target| - |y_{t+1} - y_target|
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

    states, actions, rewards, next_states, dones = [], [], [], [], []

    with torch.no_grad():
        for i in range(n_patients):
            # history valid length from active entries
            active = data["active_entries"][i]  # [T, 1]
            length = int(active.sum())
            # need at least three steps: t, t+1 for Y_{t+1}, and a future t_target > t
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

                # HER: sample future target index t_target with t < t_target <= min(t + max_tau, last_idx)
                hi = min(t + int(max_tau), last_idx)
                if hi <= t:
                    continue
                t_target = int(np.random.randint(low=t + 1, high=hi + 1))

                y_target = data["outputs"][i, t_target, :].astype(np.float32)
                delta_t = float(t_target - t)
                delta_t_norm = max(0.0, delta_t / max_tau)
                delta_t_next_norm = max(0.0, (delta_t - 1.0) / max_tau)

                a_prev_raw = data["current_treatments"][i, t - 1, :].astype(np.float32)
                if dataset_actions_unit_interval:
                    a_prev_feat = dataset_actions_to_tanh_policy_space(a_prev_raw, max_action)
                else:
                    a_prev_feat = a_prev_raw

                s = np.concatenate(
                    [z_vec, y_target, np.array([delta_t_norm], dtype=np.float32), a_prev_feat], axis=0
                )

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
                # At t+1 the "previous action" feature is a_t (behavior action at t), i.e. this transition's a.
                s_next = np.concatenate(
                    [z_next_vec, y_target, np.array([delta_t_next_norm], dtype=np.float32), a], axis=0
                )

                done = 1.0 if (t + 1) >= last_idx else 0.0
                y_next = data["outputs"][i, t + 1, :].astype(np.float32)
                y_cur = data["outputs"][i, t, :].astype(np.float32)
                # See docstring for the reward design rationale. The default
                # "progress" signal telescopes so V(s) encodes remaining distance
                # improvement — directly aligned with long-horizon planning.
                if reward_type == "negative_outcome_mse":
                    r = -float(np.mean((y_next - y_target) ** 2))
                elif reward_type == "negative_outcome":
                    r = -float(np.mean(np.abs(y_next - y_target)))
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
                

                # # 4. 恢复最朴素的密集误差奖励 (逼迫模型给药，消除 33.6 飙升)
                # y_next = data["outputs"][i, t + 1, :].astype(np.float32)
                # if reward_type == "negative_outcome_mse":
                #     r_target = -float(np.mean((y_next - y_target) ** 2))
                # else:
                #     r_target = -float(np.mean(np.abs(y_next - y_target)))
                
                # # B. 动作平滑度惩罚 (衡量用药的震荡程度) TODO 加入了平滑度
                # # 提取当前动作和上一步动作 (因为 t>=1, 所以 t-1 安全)
                # a_curr = data["current_treatments"][i, t, :].astype(np.float32)
                # a_prev = data["current_treatments"][i, t - 1, :].astype(np.float32)
                
                # # 如果做过归一化映射，为了算真实平滑度误差，建议用原值算。
                # # alpha_smoothness 是惩罚权重。如果你觉得平滑度不够，可以适当调大 (比如 0.5, 1.0)
                # alpha_smoothness = 0.2 
                # r_smoothness = -alpha_smoothness * float(np.mean((a_curr - a_prev) ** 2))

                # # 综合奖励：既要努力靠近目标，又不能剧烈改变药量
                # r = r_target + r_smoothness

                # # 5. 取消任何提前终止，严格走完患者的历史
                # done = 1.0 if (t + 1) >= last_idx else 0.0

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
