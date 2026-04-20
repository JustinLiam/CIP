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
    reward_type: str = "negative_outcome_mse",
    max_patients: Optional[int] = None,
    max_action: float = 1.0,
    dataset_actions_unit_interval: bool = True,
    max_tau: float = 12.0,
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
                # Dense shaping: every step penalizes deviation from the sampled HER target.
                # Sparse r=0 on most steps makes r + γV(s') ≈ 0 and collapses Q/V (q_loss/value_loss → 0).
                if reward_type == "negative_outcome_mse":
                    r = -float(np.mean((y_next - y_target) ** 2))
                elif reward_type == "negative_outcome":
                    r = -float(np.mean(np.abs(y_next - y_target)))
                else:
                    r = -float(np.mean(np.abs(y_next - y_target)))
                

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

    return {
        "states": np.asarray(states, dtype=np.float32),
        "actions": np.asarray(actions, dtype=np.float32),
        "rewards": np.asarray(rewards, dtype=np.float32),
        "next_states": np.asarray(next_states, dtype=np.float32),
        "dones": np.asarray(dones, dtype=np.float32),
    }
