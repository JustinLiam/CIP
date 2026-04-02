import copy
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.optim.lr_scheduler import CosineAnnealingLR


TensorBatch = List[torch.Tensor]
EXP_ADV_MAX = 100.0
LOG_STD_MIN = -20.0
LOG_STD_MAX = 2.0


@dataclass
class IQLPlannerConfig:
    state_dim: int
    action_dim: int
    max_action: float = 1.0
    hidden_dim: int = 256
    n_hidden: int = 2
    iql_tau: float = 0.7
    beta: float = 3.0
    discount: float = 0.99
    tau: float = 0.005
    actor_lr: float = 3e-4
    qf_lr: float = 3e-4
    vf_lr: float = 3e-4
    max_steps: int = 200000
    deterministic_actor: bool = False
    actor_dropout: Optional[float] = None
    device: str = "cuda"


def soft_update(target: nn.Module, source: nn.Module, tau: float):
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_((1 - tau) * target_param.data + tau * source_param.data)


def asymmetric_l2_loss(u: torch.Tensor, tau: float) -> torch.Tensor:
    return torch.mean(torch.abs(tau - (u < 0).float()) * u**2)


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


class GaussianPolicy(nn.Module):
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
        self.v_optimizer = torch.optim.Adam(self.vf.parameters(), lr=cfg.vf_lr)
        self.actor_lr_schedule = CosineAnnealingLR(self.actor_optimizer, cfg.max_steps)
        self.total_it = 0

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
            target_q = self.q_target(observations, actions)
        v = self.vf(observations)
        adv = target_q - v
        v_loss = asymmetric_l2_loss(adv, self.cfg.iql_tau)
        self.v_optimizer.zero_grad()
        v_loss.backward()
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
        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()
        soft_update(self.q_target, self.qf, self.cfg.tau)
        log_dict["q_loss"] = float(q_loss.item())

    def _update_policy(self, adv, observations, actions, log_dict):
        """
        该函数用于更新 actor policy（策略网络）的参数。policy loss 的计算方式如下：

        - Q(s, a): 评估在状态 s 下采取动作 a 的期望回报（即 critic 网络输出，通常为 Q-function）。
        - V(s): 评估在状态 s 下最佳动作的平均回报（即 value 网络输出，通常为 V-function）。
        - advantage（优势）：adv = Q(s, a) - V(s)，衡量了实际动作 a 相比当前策略最优动作的优势。
        - exp_adv: 对 advantage 进行 exp(beta * adv) 放缩，并做裁剪（最大为 EXP_ADV_MAX）。
        - actor 的输出如果是概率分布，则行为克隆损失为 -log_prob(actions) 求和；如果为确定性输出，则为均方差损失。
        - 最终 policy loss = mean(exp_adv * bc_loss)
        - 使用 actor_optimizer 对 policy loss 优化并更新 actor。

        换句话说，policy loss 是将“advantage”作为权重，对行为克隆损失做加权平均，从而鼓励策略在优势大（Q(s, a) > V(s)）的状态-动作对上尽量去模仿数据分布。
        """
        exp_adv = torch.exp(self.cfg.beta * adv.detach()).clamp(max=EXP_ADV_MAX)
        policy_out = self.actor(observations)
        if isinstance(policy_out, torch.distributions.Distribution):
            bc_losses = -policy_out.log_prob(actions).sum(-1)
        else:
            bc_losses = ((policy_out - actions) ** 2).sum(-1)
        policy_loss = torch.mean(exp_adv * bc_losses)
        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()
        self.actor_lr_schedule.step()
        log_dict["actor_loss"] = float(policy_loss.item())

    def train_step(self, batch: TensorBatch) -> Dict[str, float]:
        self.total_it += 1
        observations, actions, rewards, next_observations, dones = batch
        rewards = rewards.squeeze(-1)
        dones = dones.squeeze(-1)
        with torch.no_grad():
            next_v = self.vf(next_observations)
        log_dict: Dict[str, float] = {}
        adv = self._update_v(observations, actions, log_dict)
        self._update_q(next_v, observations, actions, rewards, dones, log_dict)
        self._update_policy(adv, observations, actions, log_dict)
        return log_dict

    @torch.no_grad()
    def act(self, state: np.ndarray) -> np.ndarray:
        return self.actor.act(state, device=self.cfg.device)

    def state_dict(self) -> Dict:
        return {
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

    def load_eval_weights(self, state: Dict) -> None:
        """Load networks for evaluation (ignores optimizers / schedulers)."""
        self.actor.load_state_dict(state["actor"])
        self.qf.load_state_dict(state["qf"])
        self.q_target.load_state_dict(state["q_target"])
        self.vf.load_state_dict(state["vf"])
        self.total_it = int(state.get("total_it", 0))

    @classmethod
    def from_checkpoint(cls, path: str, device: str = "cpu") -> "IQLPlanner":
        state = torch.load(path, map_location=device)
        cfg_dict = dict(state["cfg"])
        cfg_dict["device"] = device
        cfg = IQLPlannerConfig(**cfg_dict)
        planner = cls(cfg)
        planner.load_eval_weights(state)
        planner.actor.eval()
        return planner
