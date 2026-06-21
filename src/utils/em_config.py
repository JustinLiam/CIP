"""Lightweight EM configuration parsing and validation helpers."""
from __future__ import annotations

try:
    from omegaconf import OmegaConf
except ImportError:  # pragma: no cover - only for minimal utility imports.
    OmegaConf = None


def worlds_from_config(value, default=("sim",)):
    if value is None:
        return tuple(default)
    if OmegaConf is not None and OmegaConf.is_config(value):
        value = OmegaConf.to_container(value, resolve=True)
    if isinstance(value, str):
        raw = value.strip()
        if raw.startswith("[") and raw.endswith("]"):
            raw = raw[1:-1]
        value = [x.strip().strip("'\"") for x in raw.split(",") if x.strip()]
    worlds = tuple(str(w).strip() for w in value if str(w).strip())
    if not worlds:
        raise ValueError("exp.em_val_worlds must contain at least one world.")
    valid = {"sim", "predictor"}
    bad = [w for w in worlds if w not in valid]
    if bad:
        raise ValueError(f"Unknown exp.em_val_worlds entries {bad}; valid worlds are {sorted(valid)}.")
    return worlds


def selection_world_from_config(value, worlds):
    sel_world = str(value if value is not None else worlds[0]).strip()
    if sel_world not in worlds:
        raise ValueError(
            f"exp.em_val_selection_world={sel_world!r} must be one of exp.em_val_worlds={worlds}."
        )
    return sel_world


def empty_replay_error(data, *, max_patients, target_sampling, target_horizons, max_tau, samples_per_transition):
    active = data["active_entries"]
    n_total = int(active.shape[0])
    n_used = min(n_total, int(max_patients)) if max_patients is not None else n_total
    lengths = active[:n_used].reshape(n_used, -1).sum(axis=1) if n_used > 0 else []
    if len(lengths) > 0:
        length_msg = (
            f"min={float(lengths.min()):.0f} max={float(lengths.max()):.0f} "
            f"mean={float(lengths.mean()):.2f} ge3={int((lengths >= 3).sum())}/{n_used}"
        )
    else:
        length_msg = "no selected patients"
    return (
        "Empty raw IQL replay: no transitions were generated. "
        f"patients_total={n_total}, patients_used={n_used}, active_lengths=({length_msg}), "
        f"target_sampling={target_sampling!r}, target_horizons={target_horizons}, "
        f"max_tau={max_tau}, samples_per_transition={samples_per_transition}. "
        "Check max_patients, active sequence lengths, target horizons, and max_tau."
    )
