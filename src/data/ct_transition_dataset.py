"""
Transitions (patient, t) for standalone Causal Transformer training (ctd.md).
"""
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from src.data.iql_dataset_builder import _static_for_prefix


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
    One sample = one processed-row decision index t with history [0..t].

    Across Tumor/MIMIC/EpiCF, the processed-row contract is:
    ``prev_outputs[t]`` and ``prev_treatments[t]`` encode the pre-action state,
    ``current_treatments[t]`` is the action, and ``outputs[t]`` is the outcome
    immediately after that action.
    """

    def __init__(
        self,
        data: Dict[str, np.ndarray],
        include_next_prefix: bool = False,
    ):
        self.data = data
        self.include_next_prefix = bool(include_next_prefix)
        self.index: List[Tuple[int, int]] = []
        n = data["current_treatments"].shape[0]
        min_len = 3
        for i in range(n):
            active = data["active_entries"][i]
            length = int(active.sum())
            if length < min_len:
                continue
            for t in range(1, length - 1):
                self.index.append((i, t))

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        i, t = self.index[idx]
        tp1 = t + 1
        H = _build_H_slice(self.data, i, tp1)

        y_next = torch.tensor(self.data["outputs"][i, t, :], dtype=torch.float32)
        out: Dict = {"H_t": H, "y_next": y_next}
        if self.include_next_prefix:
            out["H_t_next"] = _build_H_slice(self.data, i, t + 2)
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
    out: Dict[str, Any] = {"H_t": H_batch, "y_next": y_next}
    if "H_t_next" in samples[0]:
        out["H_t_next"] = _collate_pad_H(samples, "H_t_next", device_dtype)
    return out


class CTEstepDataset(Dataset):
    """
    E-step only: (patient, t) history ``H_t`` without outcome labels.
    Shares the same index as :class:`CTTransitionDataset` but omits ``y_next``.
    """

    def __init__(self, data: Dict[str, np.ndarray]):
        self.data = data
        self.index: List[Tuple[int, int]] = []
        n = data["current_treatments"].shape[0]
        min_len = 3
        for i in range(n):
            active = data["active_entries"][i]
            length = int(active.sum())
            if length < min_len:
                continue
            for t in range(1, length - 1):
                self.index.append((i, t))

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> Dict[str, Dict[str, torch.Tensor]]:
        i, t = self.index[idx]
        H = _build_H_slice(self.data, i, t + 1)
        return {"H_t": H}


def collate_ct_estep_batch(samples: List[Dict]) -> Dict[str, Any]:
    """Collate E-step batches (``H_t`` only)."""
    device_dtype = samples[0]["H_t"]["prev_treatments"].dtype
    return {"H_t": _collate_pad_H(samples, "H_t", device_dtype)}
