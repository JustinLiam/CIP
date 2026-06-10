"""
Unit tests for CT+IQL EM: E-step freezes encoder; M-step encoder grad only on Q-step.
"""
import torch
from omegaconf import OmegaConf

from src.data.iql_raw_transition_dataset import IQLRawBatch
from src.models.ct_encoder_weight import CTEncoderWeightModel
from src.planners.iql_planner import IQLPlanner, IQLPlannerConfig


def _tiny_cfg():
    cfg = OmegaConf.create({
        "dataset": {
            "static_size": 1,
            "predict_X": False,
            "autoregressive": True,
            "input_size": 1,
            "output_size": 1,
            "treatment_size": 2,
        },
        "model": {
            "z_dim": 4,
            "inference": {"num_layers": 1},
        },
        "exp": {"dropout": 0.0, "ct_weight_hidden": 8},
    })
    return cfg


def _fake_H(B: int, T: int, device: str):
    return {
        "prev_treatments": torch.randn(B, T, 2, device=device),
        "current_treatments": torch.randn(B, T, 2, device=device),
        "prev_outputs": torch.randn(B, T, 1, device=device),
        "outputs": torch.randn(B, T, 1, device=device),
        "active_entries": torch.ones(B, T, 1, device=device),
        "static_features": torch.randn(B, T, 1, device=device),
    }


def test_e_step_does_not_update_encoder():
    device = "cpu"
    cfg = _tiny_cfg()
    x_dim = 4
    model = CTEncoderWeightModel(cfg, x_dim).to(device)
    opt_w = torch.optim.Adam(model.weight_net.parameters(), lr=1e-2)
    enc_before = [p.clone() for p in model.encoder_parameters()]
    H = _fake_H(8, 3, device)
    model.e_step_batch(
        H, align_mode="sinkhorn", sinkhorn_blur=0.01, k_inner=2, optimizer_w=opt_w
    )
    for p0, p1 in zip(enc_before, model.encoder_parameters()):
        assert torch.allclose(p0, p1), "encoder should be frozen in E-step"


def test_m_step_encoder_grad_only_on_q_step():
    device = "cpu"
    cfg = _tiny_cfg()
    z_dim = 4
    out_dim = 1
    act_dim = 2
    state_dim = z_dim + out_dim + 1 + act_dim
    ct_model = CTEncoderWeightModel(cfg, x_dim=4).to(device)
    enc_opt = torch.optim.Adam(ct_model.encoder_parameters(), lr=1e-3)
    planner = IQLPlanner(
        IQLPlannerConfig(
            state_dim=state_dim,
            action_dim=act_dim,
            max_action=1.0,
            hidden_dim=32,
            n_hidden=1,
            max_steps=100,
            device=device,
            encoder_max_grad_norm=1.0,
        )
    )
    B, T = 4, 3
    batch = IQLRawBatch(
        H_t=_fake_H(B, T, device),
        H_t_next=_fake_H(B, T + 1, device),
        action=torch.randn(B, act_dim, device=device),
        reward=torch.randn(B, device=device),
        done=torch.zeros(B, device=device),
        y_target=torch.randn(B, out_dim, device=device),
        delta_t_norm=torch.ones(B, 1, device=device) * 0.5,
        delta_t_next_norm=torch.ones(B, 1, device=device) * 0.4,
        a_prev_tanh=torch.randn(B, act_dim, device=device),
    )

    for p in ct_model.encoder_parameters():
        if p.grad is not None:
            p.grad = None

    Z_t, A_t = ct_model.encode(batch.H_t)
    s_grad = torch.cat(
        [Z_t, batch.y_target, batch.delta_t_norm, batch.a_prev_tanh], dim=-1
    )
    s_det = s_grad.detach()
    _, w = ct_model.compute_weights(Z_t, A_t, detach_z=True)
    w = w.detach()

    planner._update_v_weighted(s_det, batch.action, w, {})
    for p in ct_model.encoder_parameters():
        assert p.grad is None, "encoder should not get grad from V-step"

    planner._update_q_weighted_encoder(
        s_grad,
        s_det,
        batch.action,
        batch.reward,
        batch.done,
        w,
        ct_model,
        enc_opt,
        {},
    )
    has_grad = any(
        p.grad is not None and p.grad.abs().sum() > 0 for p in ct_model.encoder_parameters()
    )
    assert has_grad, "encoder should receive grad from Q-step"

    for p in ct_model.encoder_parameters():
        if p.grad is not None:
            p.grad = None
    adv = torch.zeros(B, device=device)
    planner._update_policy_weighted(s_det, batch.action, adv, w, {})
    for p in ct_model.encoder_parameters():
        assert p.grad is None, "encoder should not get grad from pi-step"


if __name__ == "__main__":
    test_e_step_does_not_update_encoder()
    test_m_step_encoder_grad_only_on_q_step()
    print("All tests passed.")
