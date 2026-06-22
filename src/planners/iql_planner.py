import copy
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import CosineAnnealingLR


TensorBatch = List[torch.Tensor]
EXP_ADV_MAX = 100.0
LOG_STD_MIN = -20.0
LOG_STD_MAX = 2.0

if TYPE_CHECKING:
    from src.data.iql_raw_transition_dataset import IQLRawBatch
    from src.models.ct_encoder_weight import CTEncoderWeightModel


@dataclass
class IQLPlannerConfig:
    # Horizon-aware + prev action: state_dim = z_dim + output_dim + 1 + action_dim (a_{t-1} in policy space)
    state_dim: int
    action_dim: int
    max_action: float = 1.0
    hidden_dim: int = 256
    n_hidden: int = 2
    iql_tau: float = 0.7
    beta: float = 3.0
    adv_max: float = EXP_ADV_MAX
    weight_max: Optional[float] = 10.0
    actor_update: str = "awr"
    actor_bc_loss: str = "nll"
    actor_bc_expectile: float = 0.7
    td3bc_q_alpha: float = 2.5
    td3bc_bc_alpha: float = 1.0
    cql_alpha: float = 0.0
    cql_n_actions: int = 10
    discount: float = 0.99
    tau: float = 0.005
    actor_lr: float = 3e-4
    qf_lr: float = 3e-4
    vf_lr: float = 3e-4
    max_steps: int = 200000
    deterministic_actor: bool = False
    actor_dropout: Optional[float] = None
    max_grad_norm: Optional[float] = None
    encoder_max_grad_norm: Optional[float] = 1.0
    device: str = "cuda"
    goal_adapter_enabled: bool = False
    z_dim: Optional[int] = None
    output_dim: Optional[int] = None
    goal_adapter_hidden_dim: int = 64
    goal_adapter_init_scale: float = 1e-3


def soft_update(target: nn.Module, source: nn.Module, tau: float):
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_((1 - tau) * target_param.data + tau * source_param.data)


def asymmetric_l2_loss(u: torch.Tensor, tau: float) -> torch.Tensor:
    return torch.mean(torch.abs(tau - (u < 0).float()) * u**2)


def weighted_asymmetric_l2_loss(u: torch.Tensor, tau: float, w: torch.Tensor) -> torch.Tensor:
    """Weighted expectile loss; w should be non-negative, mean ~ 1 per batch."""
    pinball = torch.abs(tau - (u < 0).float()) * u**2
    denom = w.sum().clamp(min=1e-8)
    return (w * pinball).sum() / denom


def _weighted_mean_sq(err: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    denom = w.sum().clamp(min=1e-8)
    return (w * err).sum() / denom


def _weighted_mean(values: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    denom = w.sum().clamp(min=1e-8)
    return (w * values).sum() / denom


def _cap_renormalize_weights(
    w: torch.Tensor, weight_max: Optional[float]
) -> torch.Tensor:
    """Cap M-step sample weights while keeping the batch mean close to 1."""
    if weight_max is None:
        return w
    cap = float(weight_max)
    if cap <= 0:
        return w
    if cap < 1.0:
        raise ValueError("weight_max must be >= 1.0 when enabled.")

    w_nonneg = w.clamp_min(0.0)
    if w_nonneg.numel() == 0:
        return w_nonneg
    if cap == 1.0:
        return torch.ones_like(w_nonneg)

    target_sum_value = float(w_nonneg.numel())
    target_sum = w_nonneg.new_tensor(target_sum_value)
    current_sum = w_nonneg.sum()
    if float(current_sum.detach().item()) <= 1e-8:
        return torch.ones_like(w_nonneg)

    positive_count = int((w_nonneg > 0).sum().detach().item())
    if positive_count == 0 or positive_count * cap < w_nonneg.numel():
        return torch.ones_like(w_nonneg)

    lo = w_nonneg.new_zeros(())
    hi = torch.clamp(target_sum / current_sum.clamp(min=1e-8), min=1.0)
    for _ in range(32):
        if float(torch.clamp(w_nonneg * hi, max=cap).sum().detach().item()) >= target_sum_value:
            break
        hi = hi * 2.0

    for _ in range(48):
        mid = (lo + hi) * 0.5
        total = torch.clamp(w_nonneg * mid, max=cap).sum()
        if float(total.detach().item()) < target_sum_value:
            lo = mid
        else:
            hi = mid
    return torch.clamp(w_nonneg * hi, max=cap)


def _log_weight_stats(log_dict: Dict[str, float], prefix: str, w: torch.Tensor) -> None:
    w_det = w.detach()
    if w_det.numel() == 0:
        return
    w_float = w_det.float()
    log_dict[f"{prefix}_mean"] = float(w_float.mean().item())
    log_dict[f"{prefix}_std"] = float(w_float.std(unbiased=False).item())
    log_dict[f"{prefix}_max"] = float(w_float.max().item())
    sorted_w = torch.sort(w_float.reshape(-1))[0]
    p95_idx = min(int(0.95 * (sorted_w.numel() - 1)), sorted_w.numel() - 1)
    log_dict[f"{prefix}_p95"] = float(sorted_w[p95_idx].item())
    ess = w_float.sum().pow(2) / w_float.pow(2).sum().clamp(min=1e-8)
    log_dict[f"{prefix}_ess_frac"] = float((ess / w_float.numel()).item())


def _tensor_l2_norm(values: List[torch.Tensor]) -> float:
    if not values:
        return 0.0
    total = torch.zeros((), device=values[0].device)
    for value in values:
        total = total + value.detach().pow(2).sum()
    return float(torch.sqrt(total).item())


def _encoder_diagnostic_groups(
    encoder_model: "CTEncoderWeightModel",
    planner: Optional["IQLPlanner"] = None,
) -> List[Tuple[str, List[Tuple[str, torch.nn.Parameter]]]]:
    groups: List[Tuple[str, List[Tuple[str, torch.nn.Parameter]]]] = []
    for child_name, module in encoder_model.ct_encoder.named_children():
        params = [
            (f"ct_encoder.{child_name}.{name}", param)
            for name, param in module.named_parameters(recurse=True)
            if param.requires_grad
        ]
        if params:
            groups.append((f"ct_encoder.{child_name}", params))
    projection_params = [
        (f"projection.{name}", param)
        for name, param in encoder_model.projection.named_parameters(recurse=True)
        if param.requires_grad
    ]
    if projection_params:
        groups.append(("projection", projection_params))
    if planner is not None and planner.goal_adapter is not None:
        adapter_params = [
            (f"goal_adapter.{name}", param)
            for name, param in planner.goal_adapter.named_parameters(recurse=True)
            if param.requires_grad
        ]
        if adapter_params:
            groups.append(("goal_adapter", adapter_params))
    return groups


class Squeeze(nn.Module):
    def __init__(self, dim: int = -1):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.squeeze(dim=self.dim)


class MLP(nn.Module):
    def __init__(
        self,
        dims: List[int],
        activation_fn=nn.ReLU,
        output_activation_fn=None,
        squeeze_output: bool = False,
        dropout: Optional[float] = None,
    ):
        super().__init__()
        layers = []
        for i in range(len(dims) - 2):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(activation_fn())
            if dropout is not None:
                layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(dims[-2], dims[-1]))
        if output_activation_fn is not None:
            layers.append(output_activation_fn())
        if squeeze_output:
            layers.append(Squeeze(-1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class GoalAdapter(nn.Module):
    """Residual goal-conditioned adapter: [z_t, y_target, delta] -> z_t^g."""

    def __init__(self, z_dim: int, output_dim: int, hidden_dim: int = 64, init_scale: float = 1e-3):
        super().__init__()
        self.z_dim = int(z_dim)
        self.output_dim = int(output_dim)
        self.net = nn.Sequential(
            nn.Linear(self.z_dim + self.output_dim + 1, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, self.z_dim),
        )
        final = self.net[-1]
        nn.init.normal_(final.weight, mean=0.0, std=float(init_scale))
        nn.init.zeros_(final.bias)

    def forward(self, z_t: torch.Tensor, y_target: torch.Tensor, delta_t_norm: torch.Tensor) -> torch.Tensor:
        if delta_t_norm.dim() == 1:
            delta_t_norm = delta_t_norm.unsqueeze(-1)
        adapter_in = torch.cat([z_t, y_target, delta_t_norm], dim=-1)
        return z_t + self.net(adapter_in)


class GaussianPolicy(nn.Module):
    """
    Tanh output in (-1, 1), then scaled to [-max_action, max_action].
    For dataset/simulator actions in [0, 1], use ``dataset_actions_to_tanh_policy_space`` in the transition builder.
    """

    def __init__(
        self,
        state_dim: int,
        act_dim: int,
        max_action: float,
        hidden_dim: int = 256,
        n_hidden: int = 2,
        dropout: Optional[float] = None,
    ):
        super().__init__()
        self.net = MLP(
            [state_dim, *([hidden_dim] * n_hidden), act_dim],
            output_activation_fn=nn.Tanh,
            dropout=dropout,
        )
        self.log_std = nn.Parameter(torch.zeros(act_dim, dtype=torch.float32))
        self.max_action = max_action

    def forward(self, obs: torch.Tensor) -> Normal:
        mean = self.net(obs)
        std = torch.exp(self.log_std.clamp(LOG_STD_MIN, LOG_STD_MAX))
        return Normal(mean, std)

    @torch.no_grad()
    def act(self, state: np.ndarray, device: str = "cpu") -> np.ndarray:
        state = torch.tensor(state.reshape(1, -1), device=device, dtype=torch.float32)
        dist = self(state)
        action = dist.mean if not self.training else dist.sample()
        action = torch.clamp(self.max_action * action, -self.max_action, self.max_action)
        return action.cpu().numpy().flatten()


class DeterministicPolicy(nn.Module):
    """Same action range convention as :class:`GaussianPolicy`."""

    def __init__(
        self,
        state_dim: int,
        act_dim: int,
        max_action: float,
        hidden_dim: int = 256,
        n_hidden: int = 2,
        dropout: Optional[float] = None,
    ):
        super().__init__()
        self.net = MLP(
            [state_dim, *([hidden_dim] * n_hidden), act_dim],
            output_activation_fn=nn.Tanh,
            dropout=dropout,
        )
        self.max_action = max_action

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)

    @torch.no_grad()
    def act(self, state: np.ndarray, device: str = "cpu") -> np.ndarray:
        state = torch.tensor(state.reshape(1, -1), device=device, dtype=torch.float32)
        action = torch.clamp(self(state) * self.max_action, -self.max_action, self.max_action)
        return action.cpu().numpy().flatten()


class TwinQ(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256, n_hidden: int = 2):
        super().__init__()
        dims = [state_dim + action_dim, *([hidden_dim] * n_hidden), 1]
        self.q1 = MLP(dims, squeeze_output=True)
        self.q2 = MLP(dims, squeeze_output=True)

    def both(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        sa = torch.cat([state, action], dim=1)
        return self.q1(sa), self.q2(sa)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return torch.min(*self.both(state, action))


class ValueFunction(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int = 256, n_hidden: int = 2):
        super().__init__()
        dims = [state_dim, *([hidden_dim] * n_hidden), 1]
        self.v = MLP(dims, squeeze_output=True)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.v(state)


class TransitionReplayBuffer:
    def __init__(self, transitions: Dict[str, np.ndarray], device: str = "cpu"):
        self.device = device
        self.states = torch.tensor(transitions["states"], dtype=torch.float32, device=device)
        self.actions = torch.tensor(transitions["actions"], dtype=torch.float32, device=device)
        self.rewards = torch.tensor(transitions["rewards"], dtype=torch.float32, device=device)
        self.next_states = torch.tensor(transitions["next_states"], dtype=torch.float32, device=device)
        self.dones = torch.tensor(transitions["dones"], dtype=torch.float32, device=device)
        self.size = self.states.shape[0]
        if self.rewards.ndim == 1:
            self.rewards = self.rewards.unsqueeze(-1)
        if self.dones.ndim == 1:
            self.dones = self.dones.unsqueeze(-1)

    def sample(self, batch_size: int) -> TensorBatch:
        idx = np.random.randint(0, self.size, size=batch_size)
        return [
            self.states[idx],
            self.actions[idx],
            self.rewards[idx],
            self.next_states[idx],
            self.dones[idx],
        ]


class IQLPlanner:
    def __init__(self, cfg: IQLPlannerConfig):
        self.cfg = cfg
        self.goal_adapter: Optional[GoalAdapter] = None
        if bool(cfg.goal_adapter_enabled):
            if cfg.z_dim is None or cfg.output_dim is None:
                raise ValueError("goal_adapter_enabled=True requires cfg.z_dim and cfg.output_dim.")
            self.goal_adapter = GoalAdapter(
                z_dim=int(cfg.z_dim),
                output_dim=int(cfg.output_dim),
                hidden_dim=int(cfg.goal_adapter_hidden_dim),
                init_scale=float(cfg.goal_adapter_init_scale),
            ).to(cfg.device)
        actor_cls = DeterministicPolicy if cfg.deterministic_actor else GaussianPolicy
        self.actor = actor_cls(
            state_dim=cfg.state_dim,
            act_dim=cfg.action_dim,
            max_action=cfg.max_action,
            hidden_dim=cfg.hidden_dim,
            n_hidden=cfg.n_hidden,
            dropout=cfg.actor_dropout,
        ).to(cfg.device)
        self.qf = TwinQ(cfg.state_dim, cfg.action_dim, cfg.hidden_dim, cfg.n_hidden).to(cfg.device)
        self.q_target = copy.deepcopy(self.qf).requires_grad_(False).to(cfg.device)
        self.vf = ValueFunction(cfg.state_dim, cfg.hidden_dim, cfg.n_hidden).to(cfg.device)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=cfg.actor_lr)
        self.q_optimizer = torch.optim.Adam(self.qf.parameters(), lr=cfg.qf_lr)
        actor_update = str(cfg.actor_update).strip().lower()
        if actor_update not in ("awr", "bc", "td3bc", "awr_td3bc"):
            raise ValueError(
                f"Unknown actor_update={cfg.actor_update!r}; expected 'awr', 'bc', 'td3bc', or 'awr_td3bc'."
            )
        self.cfg.actor_update = actor_update
        actor_bc_loss = str(cfg.actor_bc_loss).strip().lower()
        if actor_bc_loss not in ("nll", "mse", "expectile"):
            raise ValueError(
                f"Unknown actor_bc_loss={cfg.actor_bc_loss!r}; expected 'nll', 'mse', or 'expectile'."
            )
        self.cfg.actor_bc_loss = actor_bc_loss
        if not 0.0 < float(cfg.actor_bc_expectile) < 1.0:
            raise ValueError("actor_bc_expectile must be in (0, 1).")

        self.v_optimizer = torch.optim.Adam(self.vf.parameters(), lr=cfg.vf_lr)
        self.actor_lr_schedule = CosineAnnealingLR(self.actor_optimizer, cfg.max_steps)
        self.total_it = 0

    def goal_adapter_parameters(self) -> List[nn.Parameter]:
        if self.goal_adapter is None:
            return []
        return list(self.goal_adapter.parameters())

    def representation_parameters(self, encoder_model: "CTEncoderWeightModel") -> List[nn.Parameter]:
        return list(encoder_model.encoder_parameters()) + self.goal_adapter_parameters()

    def build_state(
        self,
        z_t: torch.Tensor,
        y_target: torch.Tensor,
        delta_t_norm: torch.Tensor,
        a_prev_tanh: torch.Tensor,
    ) -> torch.Tensor:
        """Build IQL state, replacing only the z component with goal-conditioned z when enabled."""
        if delta_t_norm.dim() == 1:
            delta_t_norm = delta_t_norm.unsqueeze(-1)
        if self.goal_adapter is not None:
            z_t = self.goal_adapter(z_t, y_target, delta_t_norm)
        return torch.cat([z_t, y_target, delta_t_norm, a_prev_tanh], dim=-1)

    def _actor_bc_targets(self, actions: torch.Tensor) -> torch.Tensor:
        """Normalize policy-space actions to the actor network's tanh output space."""
        max_action = float(self.cfg.max_action)
        if max_action > 0.0:
            return torch.clamp(actions / max_action, -1.0, 1.0)
        return actions

    def _update_v(self, observations, actions, log_dict) -> torch.Tensor:
        """
        value loss（v_loss）的含义和计算方式如下：

        含义：
        -------------------------
        value loss 表示 value 网络（V-function）的损失函数。其目标是让 value 网络 v(s) 拟合 Q 网络（target Q）下的
        “数据采样动作下的 Q 值的 τ 分位”。直观来说，value 网络要学习：在每个状态 s 上，对过去历史中采取的动作 a，
        学会逼近“这些动作对应 Q(s, a) 的 τ 分位数”（例如 τ=0.7 表示高于 70% 历史表现的价值）。
        这样 value network 就能避免过分高估低概率、巨额回报的 outlier。

        计算方式：
        -------------------------
        1. 先禁止梯度地用 target Q 网络计算 Q(s, a)：target_q = q_target(s, a)
        2. 用 value 网络计算 v(s)：v = vf(s)
        3. 求 Q(s, a) - v(s)，记为 advantage（优势 adv）
        4. 用 asymmetric_l2_loss 计算分位 L2 损失——相当于 quantile regression 的 pinball loss 的 L2 变种。此损失对 u > 0（高于 v(s) 的 Q(s, a)）与 u < 0（低于 v(s) 的 Q(s, a)）
           采用不同权重（tau 与 1-tau），从而引导 v(s) 向 Q(s, a) 分布的 τ 分位数收敛。
        5. 用 v_optimizer 对 value 网络参数做梯度下降
        """
        with torch.no_grad():
            q1, q2 = self.qf.both(observations, actions)
            target_q = torch.min(q1, q2)
        v = self.vf(observations)
        adv = target_q - v
        v_loss = asymmetric_l2_loss(adv, self.cfg.iql_tau)
        self.v_optimizer.zero_grad()
        v_loss.backward()
        if self.cfg.max_grad_norm is not None:
            clip_grad_norm_(self.vf.parameters(), self.cfg.max_grad_norm)
        self.v_optimizer.step()
        log_dict["value_loss"] = float(v_loss.item())
        return adv

    def _update_q(self, next_v, observations, actions, rewards, dones, log_dict):
        """
        q_loss 表示 Q 网络（Critic）的损失函数，衡量 Q 网络当前对 (s, a) 的预测值与目标 Q-value 之间的均方误差（MSE）。
        作用是让训练出来的 Q 网络能更准确地拟合环境的价值函数。

        目标 Q-value 的含义：
        ------------------------------------
        目标 Q-value 是指“当前获得的即时奖励 reward”与“未终止时未来所有奖励的期望折扣和（通过 value 网络估计）”之和：
            target = reward + gamma * next_v    （仅在 done=0 时才加未来价值 next_v）
        其中 gamma 是折扣因子（取值 0~1）；next_v 是下一个状态 s' 的 value 网络输出。

        为什么使用“双 Q 网络”？
        ------------------------------------
        在这里，qf.both(observations, actions) 分别输出 q1 与 q2 两个 Q 分支（即“双 Q 网络”）。
        其主要作用如下：
          - 抑制或减弱 Q 函数高估偏差（overestimation bias）：因为 Q 学习中最大化 Q 值时容易高估未来价值，引入两个独立 Q 网络，再随机采样/取最小值/平均等方式聚合，可有效缓解这种高估问题。
          - 训练中对两个 Q 分支分别计算损失、共同优化（如上按均值聚合 loss），使 Q 估计更稳定、更鲁棒。

        计算流程如下：
            1. 计算目标 Q-value: 若 done=1，则只包含 reward；否则 reward + 折扣 * next_v
            2. Qf 的两条分支分别计算当前 Q 值
            3. 以两个分支各自的输出与目标 Q-value 做均方误差，结果加权平均，作为最终损失
            4. 梯度下降优化 Q 网络，target Q 网络用软更新跟随最新 Q 网络

        """
        targets = rewards + (1.0 - dones.float()) * self.cfg.discount * next_v.detach()
        q1, q2 = self.qf.both(observations, actions)
        q_loss = 0.5 * (F.mse_loss(q1, targets) + F.mse_loss(q2, targets))
        cql_loss = self._cql_regularizer(observations, q1, q2)
        if cql_loss is not None:
            q_loss = q_loss + cql_loss
            log_dict["cql_loss"] = float(cql_loss.detach().item())
            log_dict["cql_alpha"] = float(self.cfg.cql_alpha)
        self.q_optimizer.zero_grad()
        q_loss.backward()
        if self.cfg.max_grad_norm is not None:
            clip_grad_norm_(self.qf.parameters(), self.cfg.max_grad_norm)
        self.q_optimizer.step()
        soft_update(self.q_target, self.qf, self.cfg.tau)
        log_dict["q_loss"] = float(q_loss.item())

    def _actor_output_tanh(self, policy_out) -> torch.Tensor:
        if isinstance(policy_out, torch.distributions.Distribution):
            return policy_out.mean
        return policy_out

    def _policy_bc_losses(self, policy_out, target_actions: torch.Tensor) -> torch.Tensor:
        if self.cfg.actor_bc_loss == "expectile":
            pred = self._actor_output_tanh(policy_out)
            diff = target_actions - pred
            tau = float(self.cfg.actor_bc_expectile)
            weights = torch.where(diff > 0.0, diff.new_tensor(tau), diff.new_tensor(1.0 - tau))
            return (weights * diff.pow(2)).sum(-1)
        if self.cfg.actor_bc_loss == "mse":
            return ((self._actor_output_tanh(policy_out) - target_actions) ** 2).sum(-1)
        if isinstance(policy_out, torch.distributions.Distribution):
            return -policy_out.log_prob(target_actions).sum(-1)
        return ((policy_out - target_actions) ** 2).sum(-1)

    def _td3bc_q_weight(self, q_values: torch.Tensor) -> torch.Tensor:
        alpha = float(self.cfg.td3bc_q_alpha)
        denom = q_values.detach().abs().mean().clamp(min=1e-6)
        return q_values.new_tensor(alpha) / denom

    def _cql_regularizer(
        self,
        observations: torch.Tensor,
        q1_data: torch.Tensor,
        q2_data: torch.Tensor,
        w: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        alpha = float(self.cfg.cql_alpha)
        n_actions = int(self.cfg.cql_n_actions)
        if alpha <= 0.0 or n_actions <= 0:
            return None
        batch_size = int(observations.shape[0])
        if batch_size <= 0:
            return None
        random_actions = torch.empty(
            batch_size * n_actions,
            int(self.cfg.action_dim),
            device=observations.device,
            dtype=observations.dtype,
        ).uniform_(-float(self.cfg.max_action), float(self.cfg.max_action))
        obs_rep = observations.repeat_interleave(n_actions, dim=0)
        q1_rand, q2_rand = self.qf.both(obs_rep, random_actions)
        log_n = np.log(float(n_actions))
        cql1 = torch.logsumexp(q1_rand.view(batch_size, n_actions), dim=1) - log_n - q1_data
        cql2 = torch.logsumexp(q2_rand.view(batch_size, n_actions), dim=1) - log_n - q2_data
        if w is None:
            cql = 0.5 * (cql1.mean() + cql2.mean())
        else:
            cql = 0.5 * (_weighted_mean(cql1, w) + _weighted_mean(cql2, w))
        return cql * q1_data.new_tensor(alpha)

    def _update_policy_td3bc(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        log_dict: Dict[str, float],
        w: Optional[torch.Tensor] = None,
    ) -> None:
        was_requires_grad = [p.requires_grad for p in self.qf.parameters()]
        for p in self.qf.parameters():
            p.requires_grad_(False)
        try:
            policy_out = self.actor(observations)
            pi_tanh = self._actor_output_tanh(policy_out)
            max_action = float(self.cfg.max_action)
            pi_action = torch.clamp(pi_tanh * max_action, -max_action, max_action)
            q_pi = self.qf(observations, pi_action)
            q_coef = self._td3bc_q_weight(q_pi)
            target_actions = self._actor_bc_targets(actions)
            bc_losses = ((pi_tanh - target_actions) ** 2).sum(-1)
            if w is None:
                q_term = q_pi.mean()
                bc_term = bc_losses.mean()
            else:
                denom = w.sum().clamp(min=1e-8)
                q_term = (w * q_pi).sum() / denom
                bc_term = (w * bc_losses).sum() / denom
            policy_loss = -q_coef * q_term + float(self.cfg.td3bc_bc_alpha) * bc_term
            self.actor_optimizer.zero_grad(set_to_none=True)
            policy_loss.backward()
            if self.cfg.max_grad_norm is not None:
                clip_grad_norm_(self.actor.parameters(), self.cfg.max_grad_norm)
            self.actor_optimizer.step()
            self.actor_lr_schedule.step()
        finally:
            for param, req in zip(self.qf.parameters(), was_requires_grad):
                param.requires_grad_(req)

        log_dict["actor_loss"] = float(policy_loss.item())
        log_dict["actor_update_td3bc"] = 1.0
        log_dict["actor_td3bc_q_term"] = float(q_term.detach().item())
        log_dict["actor_td3bc_bc_loss"] = float(bc_term.detach().item())
        log_dict["actor_td3bc_q_coef"] = float(q_coef.detach().item())

    def _update_policy_bc(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        log_dict: Dict[str, float],
        w: Optional[torch.Tensor] = None,
    ) -> None:
        policy_out = self.actor(observations)
        target_actions = self._actor_bc_targets(actions)
        bc_losses = self._policy_bc_losses(policy_out, target_actions)
        if w is None:
            policy_loss = bc_losses.mean()
        else:
            policy_loss = _weighted_mean(bc_losses, w)
        self.actor_optimizer.zero_grad(set_to_none=True)
        policy_loss.backward()
        if self.cfg.max_grad_norm is not None:
            clip_grad_norm_(self.actor.parameters(), self.cfg.max_grad_norm)
        self.actor_optimizer.step()
        self.actor_lr_schedule.step()
        log_dict["actor_loss"] = float(policy_loss.item())
        log_dict["actor_update_bc"] = 1.0
        log_dict["actor_bc_loss"] = float(policy_loss.detach().item())

    def _update_policy_awr_td3bc(
        self,
        adv: torch.Tensor,
        observations: torch.Tensor,
        actions: torch.Tensor,
        log_dict: Dict[str, float],
        w: Optional[torch.Tensor] = None,
    ) -> None:
        exp_adv = torch.exp(self.cfg.beta * adv.detach()).clamp(max=float(self.cfg.adv_max))
        was_requires_grad = [p.requires_grad for p in self.qf.parameters()]
        for p in self.qf.parameters():
            p.requires_grad_(False)
        try:
            policy_out = self.actor(observations)
            target_actions = self._actor_bc_targets(actions)
            bc_losses = self._policy_bc_losses(policy_out, target_actions)
            awr_losses = exp_adv * bc_losses

            pi_tanh = self._actor_output_tanh(policy_out)
            max_action = float(self.cfg.max_action)
            pi_action = torch.clamp(pi_tanh * max_action, -max_action, max_action)
            q_pi = self.qf(observations, pi_action)
            q_coef = self._td3bc_q_weight(q_pi)
            if w is None:
                awr_term = awr_losses.mean()
                q_term = q_pi.mean()
            else:
                awr_term = _weighted_mean(awr_losses, w)
                q_term = _weighted_mean(q_pi, w)
            policy_loss = float(self.cfg.td3bc_bc_alpha) * awr_term - q_coef * q_term
            self.actor_optimizer.zero_grad(set_to_none=True)
            policy_loss.backward()
            if self.cfg.max_grad_norm is not None:
                clip_grad_norm_(self.actor.parameters(), self.cfg.max_grad_norm)
            self.actor_optimizer.step()
            self.actor_lr_schedule.step()
        finally:
            for param, req in zip(self.qf.parameters(), was_requires_grad):
                param.requires_grad_(req)

        log_dict["actor_loss"] = float(policy_loss.item())
        log_dict["actor_update_awr_td3bc"] = 1.0
        log_dict["actor_exp_adv_mean"] = float(exp_adv.mean().item())
        log_dict["actor_exp_adv_max"] = float(exp_adv.max().item())
        log_dict["actor_awr_td3bc_awr_loss"] = float(awr_term.detach().item())
        log_dict["actor_awr_td3bc_q_term"] = float(q_term.detach().item())
        log_dict["actor_awr_td3bc_q_coef"] = float(q_coef.detach().item())

    def _update_policy(self, adv, observations, actions, log_dict):
        """
        该函数用于更新 actor policy（策略网络）的参数。policy loss 的计算方式如下：

        - Q(s, a): 评估在状态 s 下采取动作 a 的期望回报（即 critic 网络输出，通常为 Q-function）。
        - V(s): 评估在状态 s 下最佳动作的平均回报（即 value 网络输出，通常为 V-function）。
        - advantage（优势）：adv = Q(s, a) - V(s)，衡量了实际动作 a 相比当前策略最优动作的优势。
        - exp_adv: 对 advantage 进行 exp(beta * adv) 放缩，并做裁剪（最大为 cfg.adv_max）。
        - actor 的输出如果是概率分布，则行为克隆损失为 -log_prob(actions) 求和；如果为确定性输出，则为均方差损失。
        - 最终 policy loss = mean(exp_adv * bc_loss)
        - 使用 actor_optimizer 对 policy loss 优化并更新 actor。

        换句话说，policy loss 是将“advantage”作为权重，对行为克隆损失做加权平均，从而鼓励策略在优势大（Q(s, a) > V(s)）的状态-动作对上尽量去模仿数据分布。
        """
        if self.cfg.actor_update == "td3bc":
            self._update_policy_td3bc(observations, actions, log_dict)
            return
        if self.cfg.actor_update == "bc":
            self._update_policy_bc(observations, actions, log_dict)
            return
        if self.cfg.actor_update == "awr_td3bc":
            self._update_policy_awr_td3bc(adv, observations, actions, log_dict)
            return
        exp_adv = torch.exp(self.cfg.beta * adv.detach()).clamp(max=float(self.cfg.adv_max))
        policy_out = self.actor(observations)
        target_actions = self._actor_bc_targets(actions)
        bc_losses = self._policy_bc_losses(policy_out, target_actions)
        policy_loss = torch.mean(exp_adv * bc_losses)
        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        if self.cfg.max_grad_norm is not None:
            clip_grad_norm_(self.actor.parameters(), self.cfg.max_grad_norm)
        self.actor_optimizer.step()
        self.actor_lr_schedule.step()
        log_dict["actor_loss"] = float(policy_loss.item())
        log_dict["actor_exp_adv_mean"] = float(exp_adv.mean().item())
        log_dict["actor_exp_adv_max"] = float(exp_adv.max().item())

    def train_step(self, batch: TensorBatch) -> Dict[str, float]:
        self.total_it += 1
        observations, actions, rewards, next_observations, dones = batch
        rewards = rewards.squeeze(-1)
        dones = dones.squeeze(-1)
        log_dict: Dict[str, float] = {}
        adv = self._update_v(observations, actions, log_dict)
        with torch.no_grad():
            next_v = self.vf(next_observations)
        self._update_q(next_v, observations, actions, rewards, dones, log_dict)
        self._update_policy(adv, observations, actions, log_dict)
        return log_dict

    def _update_v_weighted(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        w: torch.Tensor,
        log_dict: Dict[str, float],
    ) -> torch.Tensor:
        """V-step with detached states; weighted expectile loss."""
        with torch.no_grad():
            q1, q2 = self.qf.both(observations, actions)
            target_q = torch.min(q1, q2)
        v = self.vf(observations)
        adv = target_q - v
        v_loss = weighted_asymmetric_l2_loss(adv, self.cfg.iql_tau, w)
        self.v_optimizer.zero_grad(set_to_none=True)
        v_loss.backward()
        if self.cfg.max_grad_norm is not None:
            clip_grad_norm_(self.vf.parameters(), self.cfg.max_grad_norm)
        self.v_optimizer.step()
        log_dict["value_loss"] = float(v_loss.item())
        return adv

    def _update_q_weighted_encoder(
        self,
        observations: torch.Tensor,
        next_observations: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        w: torch.Tensor,
        encoder_model: "CTEncoderWeightModel",
        encoder_optimizer: torch.optim.Optimizer,
        log_dict: Dict[str, float],
        collect_encoder_diagnostics: bool = False,
    ) -> None:
        """Q-step: weighted TD loss; gradients flow to Q and encoder (observations has grad)."""
        with torch.no_grad():
            next_v = self.vf(next_observations)
        targets = rewards + (1.0 - dones.float()) * self.cfg.discount * next_v.detach()
        q1, q2 = self.qf.both(observations, actions)
        err1 = (q1 - targets) ** 2
        err2 = (q2 - targets) ** 2
        q_loss = 0.5 * (_weighted_mean_sq(err1, w) + _weighted_mean_sq(err2, w))
        cql_loss = self._cql_regularizer(observations, q1, q2, w=w)
        if cql_loss is not None:
            q_loss = q_loss + cql_loss
            log_dict["cql_loss"] = float(cql_loss.detach().item())
            log_dict["cql_alpha"] = float(self.cfg.cql_alpha)

        self.q_optimizer.zero_grad(set_to_none=True)
        encoder_optimizer.zero_grad(set_to_none=True)
        q_loss.backward()
        diag_groups = (
            _encoder_diagnostic_groups(encoder_model, self)
            if collect_encoder_diagnostics
            else []
        )
        diag_before = {}
        if diag_groups:
            for group_name, params in diag_groups:
                param_tensors = [param for _, param in params]
                grad_tensors = [
                    param.grad
                    for _, param in params
                    if param.grad is not None
                ]
                log_dict[f"enc_grad_norm/{group_name}"] = _tensor_l2_norm(grad_tensors)
                log_dict[f"enc_param_norm/{group_name}"] = _tensor_l2_norm(param_tensors)
                for param_name, param in params:
                    diag_before[param_name] = param.detach().clone()
        if self.cfg.max_grad_norm is not None:
            clip_grad_norm_(self.qf.parameters(), self.cfg.max_grad_norm)
        enc_clip = getattr(self.cfg, "encoder_max_grad_norm", None)
        if enc_clip is not None and enc_clip > 0:
            clip_grad_norm_(self.representation_parameters(encoder_model), max_norm=float(enc_clip))
        if diag_groups:
            for group_name, params in diag_groups:
                grad_tensors = [
                    param.grad
                    for _, param in params
                    if param.grad is not None
                ]
                log_dict[f"enc_grad_norm_postclip/{group_name}"] = _tensor_l2_norm(grad_tensors)
        self.q_optimizer.step()
        encoder_optimizer.step()
        if diag_groups:
            for group_name, params in diag_groups:
                updates = [
                    param.detach() - diag_before[param_name]
                    for param_name, param in params
                    if param_name in diag_before
                ]
                update_norm = _tensor_l2_norm(updates)
                param_norm = max(log_dict[f"enc_param_norm/{group_name}"], 1e-12)
                log_dict[f"enc_update_norm/{group_name}"] = update_norm
                log_dict[f"enc_update_ratio/{group_name}"] = update_norm / param_norm
        soft_update(self.q_target, self.qf, self.cfg.tau)
        log_dict["q_loss"] = float(q_loss.item())

    def _update_policy_weighted(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        adv: torch.Tensor,
        w: torch.Tensor,
        log_dict: Dict[str, float],
    ) -> None:
        """π-step with detached states; weighted actor loss."""
        if self.cfg.actor_update == "td3bc":
            self._update_policy_td3bc(observations, actions, log_dict, w=w)
            return
        if self.cfg.actor_update == "bc":
            self._update_policy_bc(observations, actions, log_dict, w=w)
            return
        if self.cfg.actor_update == "awr_td3bc":
            self._update_policy_awr_td3bc(adv, observations, actions, log_dict, w=w)
            return
        exp_adv = torch.exp(self.cfg.beta * adv.detach()).clamp(max=float(self.cfg.adv_max))
        policy_out = self.actor(observations)
        target_actions = self._actor_bc_targets(actions)
        bc_losses = self._policy_bc_losses(policy_out, target_actions)
        policy_loss = _weighted_mean(exp_adv * bc_losses, w)
        self.actor_optimizer.zero_grad(set_to_none=True)
        policy_loss.backward()
        if self.cfg.max_grad_norm is not None:
            clip_grad_norm_(self.actor.parameters(), self.cfg.max_grad_norm)
        self.actor_optimizer.step()
        self.actor_lr_schedule.step()
        log_dict["actor_loss"] = float(policy_loss.item())
        log_dict["actor_exp_adv_mean"] = float(exp_adv.mean().item())
        log_dict["actor_exp_adv_max"] = float(exp_adv.max().item())

    def m_step_weighted(
        self,
        batch: "IQLRawBatch",
        *,
        encoder_model: "CTEncoderWeightModel",
        encoder_optimizer: torch.optim.Optimizer,
        uniform_weights: bool = False,
        collect_encoder_diagnostics: bool = False,
    ) -> Dict[str, float]:
        """
        M-step on one batch: V → Q → π with WeightNet weights.
        Only Q-step updates encoder; V/π use detached states.
        """
        self.total_it += 1
        log_dict: Dict[str, float] = {}

        Z_t, A_t = encoder_model.encode(batch.H_t)
        Z_next, _ = encoder_model.encode(batch.H_t_next)

        _, w = encoder_model.compute_weights(
            Z_t, A_t, detach_z=True, uniform=uniform_weights
        )
        w_raw = w.detach()
        w = _cap_renormalize_weights(w_raw, self.cfg.weight_max)

        s_grad = self.build_state(
            Z_t, batch.y_target, batch.delta_t_norm, batch.a_prev_tanh
        )
        s_det = s_grad.detach()
        with torch.no_grad():
            s_next_det = self.build_state(
                Z_next.detach(),
                batch.y_target,
                batch.delta_t_next_norm,
                batch.action,
            ).detach()

        adv = self._update_v_weighted(s_det, batch.action, w, log_dict)
        self._update_q_weighted_encoder(
            s_grad,
            s_next_det,
            batch.action,
            batch.reward,
            batch.done,
            w,
            encoder_model,
            encoder_optimizer,
            log_dict,
            collect_encoder_diagnostics=collect_encoder_diagnostics,
        )
        self._update_policy_weighted(s_det, batch.action, adv, w, log_dict)
        _log_weight_stats(log_dict, "w_raw", w_raw)
        _log_weight_stats(log_dict, "w", w)
        if self.cfg.weight_max is not None and self.cfg.weight_max > 0:
            log_dict["w_max_config"] = float(self.cfg.weight_max)
            log_dict["w_clip_frac"] = float(
                (w_raw > float(self.cfg.weight_max)).float().mean().item()
            )
            log_dict["w_at_cap_frac"] = float(
                (w >= float(self.cfg.weight_max) - 1e-6).float().mean().item()
            )
        return log_dict

    @torch.no_grad()
    def act(self, state: np.ndarray) -> np.ndarray:
        return self.actor.act(state, device=self.cfg.device)

    def state_dict(self) -> Dict:
        out = {
            "actor": self.actor.state_dict(),
            "qf": self.qf.state_dict(),
            "q_target": self.q_target.state_dict(),
            "vf": self.vf.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "q_optimizer": self.q_optimizer.state_dict(),
            "v_optimizer": self.v_optimizer.state_dict(),
            "actor_lr_schedule": self.actor_lr_schedule.state_dict(),
            "total_it": self.total_it,
            "cfg": self.cfg.__dict__,
        }
        if self.goal_adapter is not None:
            out["goal_adapter"] = self.goal_adapter.state_dict()
        return out

    def load_eval_weights(self, state: Dict) -> None:
        """Load networks for evaluation (ignores optimizers / schedulers)."""
        self.actor.load_state_dict(state["actor"])
        self.qf.load_state_dict(state["qf"])
        self.q_target.load_state_dict(state["q_target"])
        self.vf.load_state_dict(state["vf"])
        if self.goal_adapter is not None:
            if "goal_adapter" not in state:
                raise ValueError("Checkpoint is missing goal_adapter weights.")
            self.goal_adapter.load_state_dict(state["goal_adapter"])
        self.total_it = int(state.get("total_it", 0))

    @classmethod
    def from_checkpoint(cls, path: str, device: str = "cpu") -> "IQLPlanner":
        state = torch.load(path, map_location=device)
        cfg_dict = dict(state["cfg"])
        cfg_dict["device"] = device
        cfg_dict.setdefault("max_grad_norm", None)
        cfg_dict.setdefault("encoder_max_grad_norm", 1.0)
        cfg_dict.setdefault("adv_max", EXP_ADV_MAX)
        cfg_dict.setdefault("weight_max", 10.0)
        cfg_dict.setdefault("actor_update", "awr")
        cfg_dict.setdefault("actor_bc_loss", "nll")
        cfg_dict.setdefault("actor_bc_expectile", 0.7)
        cfg_dict.setdefault("td3bc_q_alpha", 2.5)
        cfg_dict.setdefault("td3bc_bc_alpha", 1.0)
        cfg_dict.setdefault("cql_alpha", 0.0)
        cfg_dict.setdefault("cql_n_actions", 10)
        cfg_dict.setdefault("goal_adapter_enabled", False)
        cfg_dict.setdefault("z_dim", None)
        cfg_dict.setdefault("output_dim", None)
        cfg_dict.setdefault("goal_adapter_hidden_dim", 64)
        cfg_dict.setdefault("goal_adapter_init_scale", 1e-3)
        cfg = IQLPlannerConfig(**cfg_dict)
        planner = cls(cfg)
        planner.load_eval_weights(state)
        planner.actor.eval()
        return planner
