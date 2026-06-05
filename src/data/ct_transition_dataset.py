"""
Transitions (patient, t) for standalone Causal Transformer training (ctd.md).
"""
import random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from src.data.iql_dataset_builder import _static_for_prefix


def rollout_horizon_distribution(
    k_max: int,
    k_dist: str,
    eta: float,
) -> Tuple[float, List[float]]:
    """
    Return (E[k], P(k=1..k_max)) for logging.

    geometric: P(k=j) ∝ eta^(j-1), j=1..k_max (favors short horizons when eta<1).
    uniform: P(k=j) = 1/k_max.
    """
    k_max = max(1, int(k_max))
    if str(k_dist).strip().lower() == "uniform":
        probs = [1.0 / k_max] * k_max
    else:
        weights = [float(eta) ** (j - 1) for j in range(1, k_max + 1)]
        s = sum(weights)
        probs = [w / s for w in weights]
    mean_k = sum((j + 1) * p for j, p in enumerate(probs))
    return float(mean_k), probs


def _sample_rollout_horizon_k(
    k_max: int,
    k_dist: str,
    eta: float,
    *,
    train: bool,
    val_k: int,
) -> int:
    """Sample rollout horizon k in {1, ..., k_max} (1-based)."""
    if not train:
        return max(1, min(int(val_k), int(k_max)))
    if k_dist == "uniform":
        return random.randint(1, int(k_max))
    # geometric: P(k=j) ∝ eta^(j-1)
    _, probs = rollout_horizon_distribution(k_max, k_dist, eta)
    return random.choices(range(1, int(k_max) + 1), weights=probs, k=1)[0]


def _covariate_stream_dim(dataset_cfg: Dict[str, Any]) -> int:
    """Last-dim size of x after InferenceModel.build_H_t-style concat (covariate branch)."""
    static_size = int(dataset_cfg["static_size"])
    input_size = int(dataset_cfg["input_size"])
    predict_x = bool(dataset_cfg.get("predict_X", False))
    autoregressive = bool(dataset_cfg.get("autoregressive", False))
    output_size = int(dataset_cfg["output_size"])
    treatment_size = int(dataset_cfg["treatment_size"])

    if static_size > 0:
        if predict_x:
            d = input_size + static_size
        else:
            d = static_size
    else:
        d = input_size
    if autoregressive:
        d += output_size
    d += treatment_size
    return d


def _build_H_slice(data: Dict[str, np.ndarray], i: int, tp1: int) -> Dict[str, torch.Tensor]:
    """Prefix time slice [0:tp1] for patient row i (batch dim 1)."""
    sl = slice(i, i + 1)
    H: Dict[str, torch.Tensor] = {
        "prev_treatments": torch.tensor(data["prev_treatments"][sl, :tp1, :], dtype=torch.float32),
        "current_treatments": torch.tensor(data["current_treatments"][sl, :tp1, :], dtype=torch.float32),
        "prev_outputs": torch.tensor(data["prev_outputs"][sl, :tp1, :], dtype=torch.float32),
        "outputs": torch.tensor(data["outputs"][sl, :tp1, :], dtype=torch.float32),
        "active_entries": torch.tensor(data["active_entries"][sl, :tp1, :], dtype=torch.float32),
    }
    if "current_covariates" in data:
        H["current_covariates"] = torch.tensor(data["current_covariates"][sl, :tp1, :], dtype=torch.float32)
    if "vitals" in data:
        H["vitals"] = torch.tensor(data["vitals"][sl, :tp1, :], dtype=torch.float32)
    if "static_features" in data:
        H["static_features"] = torch.tensor(_static_for_prefix(data, i, tp1), dtype=torch.float32)
    return H


class CTTransitionDataset(Dataset):
    """
    One sample = one time index t with history [0..t], target Y_{t+1}.
    t runs from 1 .. length-2 so that outputs[t+1] exists.

    If ``multi_k_max`` > 1, also returns longer teacher prefixes for k=2..K targets Y_{t+2}..Y_{t+K}
    (requires length >= t + K + 1).

    If ``rollout_mode`` == ``latent_dynamics``, returns action sequence A_{t:t+k-1}, horizon k,
    Y_{t+k}, and optional future history prefixes for latent consistency targets.
    """

    def __init__(
        self,
        data: Dict[str, np.ndarray],
        multi_k_max: int = 1,
        include_next_prefix: bool = False,
        *,
        rollout_mode: str = "none",
        rollout_k_max: int = 1,
        rollout_k_dist: str = "geometric",
        rollout_eta: float = 0.7,
        rollout_return_future_prefixes: bool = True,
        train: bool = True,
        rollout_val_k: int = 1,
    ):
        self.data = data
        self.multi_k_max = int(multi_k_max)
        self.include_next_prefix = bool(include_next_prefix)
        self.rollout_mode = str(rollout_mode).strip().lower()
        self.rollout_k_max = max(1, int(rollout_k_max))
        self.rollout_k_dist = str(rollout_k_dist).strip().lower()
        self.rollout_eta = float(rollout_eta)
        self.rollout_return_future_prefixes = bool(rollout_return_future_prefixes)
        self.train = bool(train)
        self.rollout_val_k = max(1, int(rollout_val_k))
        # Optional [num_patients, max_seq_len, 1] dataset/time-level weights (offline refresh).
        self.weight_table: Optional[torch.Tensor] = None
        self.index: List[Tuple[int, int]] = []
        n = data["current_treatments"].shape[0]
        if self.rollout_mode == "latent_dynamics":
            k_bound = self.rollout_k_max
        else:
            k_bound = self.multi_k_max
        min_len = k_bound + 2
        for i in range(n):
            active = data["active_entries"][i]
            length = int(active.sum())
            if length < min_len:
                continue
            for t in range(1, length - k_bound):
                self.index.append((i, t))

    def __len__(self) -> int:
        return len(self.index)

    def set_weight_table(self, weight_table: Optional[torch.Tensor]) -> None:
        """Attach dataset/time-level weights; None resets to uniform w=1."""
        self.weight_table = weight_table

    def get_weight(self, patient_id: int, time_id: int) -> float:
        if self.weight_table is None:
            return 1.0
        return float(self.weight_table[int(patient_id), int(time_id), 0].item())

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        i, t = self.index[idx]
        tp1 = t + 1
        H = _build_H_slice(self.data, i, tp1)

        y_next = torch.tensor(self.data["outputs"][i, t + 1, :], dtype=torch.float32)
        w_val = self.get_weight(i, t)
        out: Dict = {
            "H_t": H,
            "y_next": y_next,
            "patient_id": torch.tensor(i, dtype=torch.long),
            "time_id": torch.tensor(t, dtype=torch.long),
            "w": torch.tensor(w_val, dtype=torch.float32),
        }

        if self.rollout_mode == "latent_dynamics":
            k = _sample_rollout_horizon_k(
                self.rollout_k_max,
                self.rollout_k_dist,
                self.rollout_eta,
                train=self.train,
                val_k=self.rollout_val_k,
            )
            action_dim = self.data["current_treatments"].shape[-1]
            a_seq = torch.zeros(self.rollout_k_max, action_dim, dtype=torch.float32)
            a_raw = self.data["current_treatments"][i, t : t + k, :]
            a_seq[:k, :] = torch.tensor(a_raw, dtype=torch.float32)
            a_seq_mask = torch.zeros(self.rollout_k_max, dtype=torch.float32)
            a_seq_mask[:k] = 1.0
            out["a_seq"] = a_seq
            out["a_seq_mask"] = a_seq_mask
            out["horizon_k"] = torch.tensor(k, dtype=torch.long)
            out["y_future"] = torch.tensor(self.data["outputs"][i, t + k, :], dtype=torch.float32)
            if self.rollout_return_future_prefixes:
                for j in range(1, self.rollout_k_max + 1):
                    out[f"H_t_future_{j}"] = _build_H_slice(self.data, i, t + j + 1)
            return out

        if self.include_next_prefix:
            out["H_t_next"] = _build_H_slice(self.data, i, t + 2)
        if self.multi_k_max >= 2:
            out["H_t_k2"] = _build_H_slice(self.data, i, t + 2)
            out["y_next2"] = torch.tensor(self.data["outputs"][i, t + 2, :], dtype=torch.float32)
        if self.multi_k_max >= 3:
            out["H_t_k3"] = _build_H_slice(self.data, i, t + 3)
            out["y_next3"] = torch.tensor(self.data["outputs"][i, t + 3, :], dtype=torch.float32)
        return out


def _collate_pad_H(samples: List[Dict], h_key: str, device_dtype) -> Dict[str, torch.Tensor]:
    """Pad one H dict (nested under h_key in each sample) to max T in batch."""
    B = len(samples)
    T_max = max(s[h_key]["prev_treatments"].shape[1] for s in samples)

    def pad2(name: str, last_dim: int) -> torch.Tensor:
        out = torch.zeros(B, T_max, last_dim, dtype=device_dtype)
        for b, s in enumerate(samples):
            x = s[h_key][name]
            L = x.shape[1]
            out[b, :L, :] = x.squeeze(0)
        return out

    H_batch: Dict[str, torch.Tensor] = {}
    keys = list(samples[0][h_key].keys())
    for key in keys:
        if key == "static_features":
            sf0 = samples[0][h_key]["static_features"]
            ld = sf0.shape[-1]
            if sf0.dim() == 3:
                out = torch.zeros(B, T_max, ld, dtype=device_dtype)
                for b, s in enumerate(samples):
                    x = s[h_key]["static_features"]
                    L = min(x.shape[1], T_max)
                    out[b, :L, :] = x.squeeze(0)[:L, :]
                H_batch[key] = out
            else:
                H_batch[key] = torch.stack([s[h_key]["static_features"].squeeze(0) for s in samples], dim=0)
            continue
        ld = samples[0][h_key][key].shape[-1]
        H_batch[key] = pad2(key, ld)

    act = H_batch["active_entries"]
    for b, s in enumerate(samples):
        L = s[h_key]["prev_treatments"].shape[1]
        if L < T_max:
            act[b, L:, :] = 0.0
    return H_batch


def collate_ct_batch(samples: List[Dict]) -> Dict[str, Any]:
    """Pad variable-length history to max length in batch."""
    device_dtype = samples[0]["H_t"]["prev_treatments"].dtype
    H_batch = _collate_pad_H(samples, "H_t", device_dtype)
    y_next = torch.stack([s["y_next"] for s in samples], dim=0)
    patient_id = torch.stack([s["patient_id"] for s in samples], dim=0)
    time_id = torch.stack([s["time_id"] for s in samples], dim=0)
    w = torch.stack([s["w"] for s in samples], dim=0)
    out: Dict[str, Any] = {
        "H_t": H_batch,
        "y_next": y_next,
        "patient_id": patient_id,
        "time_id": time_id,
        "w": w,
    }
    if "H_t_next" in samples[0]:
        out["H_t_next"] = _collate_pad_H(samples, "H_t_next", device_dtype)
    if "H_t_k2" in samples[0]:
        out["H_t_k2"] = _collate_pad_H(samples, "H_t_k2", device_dtype)
        out["y_next2"] = torch.stack([s["y_next2"] for s in samples], dim=0)
    if "H_t_k3" in samples[0]:
        out["H_t_k3"] = _collate_pad_H(samples, "H_t_k3", device_dtype)
        out["y_next3"] = torch.stack([s["y_next3"] for s in samples], dim=0)
    if "a_seq" in samples[0]:
        out["a_seq"] = torch.stack([s["a_seq"] for s in samples], dim=0)
        out["a_seq_mask"] = torch.stack([s["a_seq_mask"] for s in samples], dim=0)
        out["horizon_k"] = torch.stack([s["horizon_k"] for s in samples], dim=0)
        out["y_future"] = torch.stack([s["y_future"] for s in samples], dim=0)
        for j in range(1, 64):
            key = f"H_t_future_{j}"
            if key not in samples[0]:
                break
            out[key] = _collate_pad_H(samples, key, device_dtype)
    return out
