"""Action selection helpers for IQL policy evaluation."""

from __future__ import annotations

from typing import Literal

import torch
from torch.distributions import Distribution

from src.planners.iql_planner import IQLPlanner


ActionSelector = Literal["mean", "q_sample"]


def _policy_output_tanh(policy_out) -> torch.Tensor:
    if isinstance(policy_out, Distribution):
        return policy_out.mean
    return policy_out


@torch.no_grad()
def select_iql_policy_action(
    planner: IQLPlanner,
    obs: torch.Tensor,
    *,
    selector: str = "mean",
    candidate_actions: int = 16,
    q_bc_penalty: float = 0.0,
    candidate_noise_std: float = 0.25,
    include_mean: bool = True,
) -> torch.Tensor:
    """Return policy-space actions in ``[-max_action, max_action]``.

    ``mean`` preserves the historical deterministic actor-mean evaluation.
    ``q_sample`` samples candidate tanh-space actions from the learned behavior
    actor and picks the highest critic value, optionally penalizing distance
    from the actor mean to keep selection inside behavior support.
    """
    selector = str(selector).strip().lower()
    policy_out = planner.actor(obs)
    max_action = float(planner.actor.max_action)
    mean_tanh = _policy_output_tanh(policy_out)
    if selector in ("", "mean", "actor_mean"):
        return torch.clamp(mean_tanh * max_action, -max_action, max_action)
    if selector not in ("q_sample", "sample_q", "support_q"):
        raise ValueError(
            f"Unknown IQL action selector {selector!r}; expected 'mean' or 'q_sample'."
        )

    bsz, action_dim = mean_tanh.shape
    n_total = max(1, int(candidate_actions))
    candidates = []
    if include_mean:
        candidates.append(mean_tanh.unsqueeze(1))
    n_sample = max(0, n_total - len(candidates))
    if n_sample > 0:
        if isinstance(policy_out, Distribution):
            sampled = policy_out.rsample((n_sample,)).permute(1, 0, 2)
        else:
            noise = torch.randn(
                bsz,
                n_sample,
                action_dim,
                device=obs.device,
                dtype=obs.dtype,
            ) * float(candidate_noise_std)
            sampled = mean_tanh.unsqueeze(1) + noise
        candidates.append(sampled)
    cand_tanh = torch.cat(candidates, dim=1).clamp(-1.0, 1.0)
    cand_action = torch.clamp(cand_tanh * max_action, -max_action, max_action)

    n_cand = cand_action.size(1)
    obs_rep = obs.unsqueeze(1).expand(bsz, n_cand, obs.size(-1)).reshape(bsz * n_cand, obs.size(-1))
    act_rep = cand_action.reshape(bsz * n_cand, action_dim)
    q_values = planner.qf(obs_rep, act_rep).view(bsz, n_cand)
    penalty = float(q_bc_penalty)
    if penalty > 0.0:
        dist_sq = (cand_tanh - mean_tanh.unsqueeze(1)).pow(2).sum(dim=-1)
        q_values = q_values - q_values.new_tensor(penalty) * dist_sq
    best = torch.argmax(q_values, dim=1)
    return cand_action[torch.arange(bsz, device=obs.device), best]
