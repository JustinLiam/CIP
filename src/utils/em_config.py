"""Lightweight EM configuration parsing and validation helpers."""
from __future__ import annotations

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
