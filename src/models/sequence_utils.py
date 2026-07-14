"""Utilities for active-entry aware sequence indexing."""
from __future__ import annotations

import torch


def _active_2d(active_entries: torch.Tensor) -> torch.Tensor:
    if active_entries.dim() == 3 and active_entries.size(-1) == 1:
        return active_entries.squeeze(-1)
    if active_entries.dim() == 2:
        return active_entries
    raise ValueError(
        "active_entries must have shape [B, T] or [B, T, 1], "
        f"got {tuple(active_entries.shape)}"
    )


def last_valid_indices(active_entries: torch.Tensor) -> torch.Tensor:
    """Return [B] indices of the rightmost active timestep, clamped for empty rows.

    Uses the rightmost ``active > 0.5`` position rather than ``sum(active) - 1``,
    so contiguous prefix padding (``[1,1,0,0]``) and non-contiguous masks
    (``[1,1,0,0,1]``) are both handled correctly.
    """
    active = _active_2d(active_entries)
    is_active = active > 0.5
    positions = torch.arange(active.size(1), device=active.device, dtype=torch.long)
    positions = positions.unsqueeze(0).expand_as(is_active)
    # Inactive -> -1 so max yields the rightmost active index; empty rows -> -1 -> 0.
    return torch.where(is_active, positions, positions.new_full(positions.shape, -1)).max(dim=1).values.clamp(min=0)


def last_valid_mask(active_entries: torch.Tensor) -> torch.Tensor:
    """Return [B] float mask indicating whether each row has any active timestep."""
    active = _active_2d(active_entries)
    lengths = (active > 0.5).long().sum(dim=1)
    return (lengths > 0).to(dtype=active_entries.dtype, device=active_entries.device)


def gather_last_valid(sequence: torch.Tensor, active_entries: torch.Tensor | None) -> torch.Tensor:
    """Gather [B, D] from sequence [B, T, D] at each row's last active timestep."""
    if sequence.dim() != 3:
        raise ValueError(f"sequence must have shape [B, T, D], got {tuple(sequence.shape)}")
    if active_entries is None:
        return sequence[:, -1, :]
    idx = last_valid_indices(active_entries).to(device=sequence.device)
    gather_idx = idx.view(-1, 1, 1).expand(-1, 1, sequence.size(-1))
    return sequence.gather(dim=1, index=gather_idx).squeeze(1)


def active_time_mask(active_entries: torch.Tensor | None, reference: torch.Tensor) -> torch.Tensor:
    """Return active_entries as [B, T, 1] on reference's device/dtype."""
    if active_entries is None:
        return torch.ones(
            reference.size(0),
            reference.size(1),
            1,
            device=reference.device,
            dtype=reference.dtype,
        )
    active = active_entries.to(device=reference.device, dtype=reference.dtype)
    if active.dim() == 2:
        active = active.unsqueeze(-1)
    return active
