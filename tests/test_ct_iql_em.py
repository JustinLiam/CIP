"""
Unit tests for CT+IQL EM: E-step freezes encoder; M-step encoder grad only on Q-step.
"""
import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from src.data.ct_transition_dataset import _collate_pad_H
from src.data.iql_dataset_builder import dataset_actions_to_tanh_policy_space
from src.data.iql_raw_transition_dataset import (
    IQLRawBatch,
    IQLRawReplayBuffer,
    build_iql_raw_transitions,
)
from src.evaluation.iql_planner_eval import _make_action_grid, _q_grid_action_diagnostics
from src.models.ct_deconfound import CTDeconfoundModel
from src.models.ct_encoder_weight import CTEncoderWeightModel
from src.models.ct_history_encoder import CTHistoryEncoder
from src.models.sequence_utils import gather_last_valid, last_valid_indices, last_valid_mask
from src.planners.iql_planner import IQLPlanner, IQLPlannerConfig, _weighted_mean
from src.utils.em_config import empty_replay_error, selection_world_from_config, worlds_from_config


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


def _make_variable_history(current_values):
    T = len(current_values)
    current = torch.tensor(current_values, dtype=torch.float32).view(1, T, 2)
    prev = current - 1.0
    outputs = torch.arange(T, dtype=torch.float32).view(1, T, 1)
    return {
        "prev_treatments": prev,
        "current_treatments": current,
        "prev_outputs": outputs - 0.5,
        "outputs": outputs,
        "active_entries": torch.ones(1, T, 1),
        "static_features": torch.ones(1, T, 1),
    }


def _variable_padded_batch():
    samples = [
        {"H_t": _make_variable_history([[10.0, 11.0], [12.0, 13.0]])},
        {"H_t": _make_variable_history([[20.0, 21.0], [22.0, 23.0], [24.0, 25.0], [26.0, 27.0]])},
    ]
    return _collate_pad_H(samples, "H_t", torch.float32)


def _raw_iql_data(length: int = 4):
    return {
        "prev_treatments": np.zeros((1, length, 2), dtype=np.float32),
        "current_treatments": np.linspace(0.0, 1.0, num=length * 2, dtype=np.float32).reshape(1, length, 2),
        "prev_outputs": np.zeros((1, length, 1), dtype=np.float32),
        "outputs": np.arange(length, dtype=np.float32).reshape(1, length, 1),
        "active_entries": np.ones((1, length, 1), dtype=np.float32),
    }


def test_last_valid_gather_after_collate_padding():
    H = _variable_padded_batch()

    assert torch.allclose(H["current_treatments"][0, -1, :], torch.zeros(2))
    assert torch.equal(last_valid_indices(H["active_entries"]), torch.tensor([1, 3]))
    assert torch.allclose(last_valid_mask(H["active_entries"]), torch.ones(2))

    A_t = gather_last_valid(H["current_treatments"], H["active_entries"])
    assert torch.allclose(A_t[0], torch.tensor([12.0, 13.0]))
    assert torch.allclose(A_t[1], torch.tensor([26.0, 27.0]))


def test_ct_models_encode_use_last_valid_action_after_padding():
    cfg = _tiny_cfg()
    H = _variable_padded_batch()

    weight_model = CTEncoderWeightModel(cfg, x_dim=4)
    _, A_weight = weight_model.encode(H)
    assert torch.allclose(A_weight[0], torch.tensor([12.0, 13.0]))
    assert torch.allclose(A_weight[1], torch.tensor([26.0, 27.0]))

    deconfound_model = CTDeconfoundModel(cfg, x_dim=4)
    _, A_deconf = deconfound_model.encode(H)
    assert torch.allclose(A_deconf[0], torch.tensor([12.0, 13.0]))
    assert torch.allclose(A_deconf[1], torch.tensor([26.0, 27.0]))


def test_ct_history_encoder_zeroes_padded_positions():
    encoder = CTHistoryEncoder(
        x_dim=4,
        a_dim=2,
        y_dim=1,
        static_dim=1,
        d_model=8,
        num_heads=2,
        num_layers=1,
        dropout=0.0,
        local_conv_layers=0,
    )
    encoder.eval()
    active = torch.tensor([[[1.0], [1.0], [0.0], [0.0]]])
    rep = encoder(
        x=torch.randn(1, 4, 4),
        a=torch.randn(1, 4, 2),
        y=torch.randn(1, 4, 1),
        active_entries=active,
        static_features=torch.ones(1, 4, 1),
    )
    assert torch.allclose(rep[:, 2:, :], torch.zeros_like(rep[:, 2:, :]), atol=1e-6)


def test_raw_her_target_reached_sets_done():
    transitions = build_iql_raw_transitions(
        _raw_iql_data(length=4),
        max_tau=2,
        reward_clip=0.0,
        reward_scale="none",
        samples_per_transition=1,
        target_sampling="horizon_aligned",
        target_horizons=[1],
        horizon_terminal_done=True,
        seed=7,
    )
    assert transitions
    for transition in transitions:
        assert transition.t_target == transition.t + 1
        assert transition.delta_t_next_norm == 0.0
        assert transition.done == 1.0


def test_empty_replay_sampling_fails_readably():
    replay = IQLRawReplayBuffer([], device="cpu")
    with pytest.raises(ValueError, match="empty IQLRawReplayBuffer"):
        replay.sample(1)


def test_em_worlds_config_parses_and_validates():
    assert worlds_from_config(["sim"]) == ("sim",)
    assert worlds_from_config("sim,predictor") == ("sim", "predictor")
    assert worlds_from_config("[sim,predictor]") == ("sim", "predictor")
    assert selection_world_from_config(None, ("sim", "predictor")) == "sim"
    assert selection_world_from_config("predictor", ("sim", "predictor")) == "predictor"
    with pytest.raises(ValueError, match="Unknown exp.em_val_worlds"):
        worlds_from_config("invalid")
    with pytest.raises(ValueError, match="em_val_selection_world"):
        selection_world_from_config("predictor", ("sim",))

    msg = empty_replay_error(
        {"active_entries": np.zeros((2, 4, 1), dtype=np.float32)},
        max_patients=None,
        target_sampling="horizon_aligned",
        target_horizons=[1, 2],
        max_tau=2,
        samples_per_transition=1,
    )
    assert "patients_total=2" in msg
    assert "active_lengths" in msg
    assert "target_horizons=[1, 2]" in msg


def test_max_action_mapping_and_actor_bc_targets_are_consistent():
    actions = np.array([[0.0, 0.5, 1.0]], dtype=np.float32)
    mapped = dataset_actions_to_tanh_policy_space(actions, max_action=2.0)
    np.testing.assert_allclose(mapped, np.array([[-2.0, 0.0, 2.0]], dtype=np.float32))
    np.testing.assert_allclose((mapped + 2.0) / 4.0, actions)

    planner = IQLPlanner(
        IQLPlannerConfig(
            state_dim=3,
            action_dim=3,
            max_action=2.0,
            hidden_dim=8,
            n_hidden=1,
            max_steps=10,
            device="cpu",
        )
    )
    target = planner._actor_bc_targets(torch.tensor(mapped))
    assert torch.allclose(target, torch.tensor([[-1.0, 0.0, 1.0]]))


def test_predictor_world_without_loaded_predictor_hard_fails():
    from src.evaluation.iql_planner_eval import _rollout_one_step_y

    class DummyInference:
        _outcome_predictor_loaded = False

    with pytest.raises(RuntimeError, match="outcome_predictor weights"):
        _rollout_one_step_y(
            "predictor",
            {},
            torch.zeros(1, 2),
            fold=None,
            scaling_params=None,
            inference_model=DummyInference(),
            device="cpu",
        )


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



def test_actor_update_default_awr_path_logs_adv_weights():
    planner = IQLPlanner(
        IQLPlannerConfig(
            state_dim=5,
            action_dim=2,
            max_action=1.0,
            hidden_dim=16,
            n_hidden=1,
            max_steps=10,
            device="cpu",
        )
    )
    obs = torch.randn(6, 5)
    actions = torch.clamp(torch.randn(6, 2), -1.0, 1.0)
    adv = torch.randn(6)
    logs = {}
    planner._update_policy(adv, obs, actions, logs)
    assert logs["actor_loss"] >= 0.0
    assert "actor_exp_adv_mean" in logs
    assert "actor_td3bc_q_term" not in logs


def test_td3bc_actor_update_updates_actor_without_q_grad_accumulation():
    planner = IQLPlanner(
        IQLPlannerConfig(
            state_dim=5,
            action_dim=2,
            max_action=1.0,
            hidden_dim=16,
            n_hidden=1,
            max_steps=10,
            device="cpu",
            actor_update="td3bc",
        )
    )
    obs = torch.randn(8, 5)
    actions = torch.clamp(torch.randn(8, 2), -1.0, 1.0)
    adv = torch.randn(8)
    for p in planner.qf.parameters():
        p.grad = None
    before = [p.detach().clone() for p in planner.actor.parameters()]
    logs = {}
    planner._update_policy(adv, obs, actions, logs)
    assert logs["actor_update_td3bc"] == 1.0
    assert logs["actor_td3bc_q_coef"] > 0.0
    assert all(p.grad is None for p in planner.qf.parameters())
    assert all(p.requires_grad for p in planner.qf.parameters())
    changed = any((a.detach() - b).abs().sum() > 0 for a, b in zip(planner.actor.parameters(), before))
    assert changed


def test_bc_actor_update_uses_behavior_cloning_without_advantage_weights():
    planner = IQLPlanner(
        IQLPlannerConfig(
            state_dim=5,
            action_dim=2,
            max_action=1.0,
            hidden_dim=16,
            n_hidden=1,
            max_steps=10,
            device="cpu",
            actor_update="bc",
        )
    )
    obs = torch.randn(8, 5)
    actions = torch.clamp(torch.randn(8, 2), -1.0, 1.0)
    adv = torch.randn(8)
    before = [p.detach().clone() for p in planner.actor.parameters()]
    logs = {}
    planner._update_policy(adv, obs, actions, logs)
    assert logs["actor_update_bc"] == 1.0
    assert logs["actor_bc_loss"] >= 0.0
    assert "actor_exp_adv_mean" not in logs
    assert "actor_td3bc_q_term" not in logs
    changed = any((a.detach() - b).abs().sum() > 0 for a, b in zip(planner.actor.parameters(), before))
    assert changed


def test_awr_td3bc_actor_update_keeps_adv_weights_and_q_gradient():
    planner = IQLPlanner(
        IQLPlannerConfig(
            state_dim=5,
            action_dim=2,
            max_action=1.0,
            hidden_dim=16,
            n_hidden=1,
            max_steps=10,
            device="cpu",
            actor_update="awr_td3bc",
            td3bc_q_alpha=0.1,
        )
    )
    obs = torch.randn(8, 5)
    actions = torch.clamp(torch.randn(8, 2), -1.0, 1.0)
    adv = torch.randn(8)
    for p in planner.qf.parameters():
        p.grad = None
    before = [p.detach().clone() for p in planner.actor.parameters()]
    logs = {}
    planner._update_policy(adv, obs, actions, logs)
    assert logs["actor_update_awr_td3bc"] == 1.0
    assert "actor_exp_adv_mean" in logs
    assert logs["actor_awr_td3bc_q_coef"] > 0.0
    assert all(p.grad is None for p in planner.qf.parameters())
    changed = any((a.detach() - b).abs().sum() > 0 for a, b in zip(planner.actor.parameters(), before))
    assert changed


def test_weighted_mean_does_not_square_policy_losses():
    values = torch.tensor([2.0, 4.0])
    weights = torch.tensor([1.0, 3.0])
    assert torch.isclose(_weighted_mean(values, weights), torch.tensor(3.5))


def test_cql_regularizer_logs_when_enabled():
    planner = IQLPlanner(
        IQLPlannerConfig(
            state_dim=5,
            action_dim=2,
            max_action=1.0,
            hidden_dim=16,
            n_hidden=1,
            max_steps=10,
            device="cpu",
            cql_alpha=0.01,
            cql_n_actions=3,
        )
    )
    batch = [
        torch.randn(8, 5),
        torch.clamp(torch.randn(8, 2), -1.0, 1.0),
        torch.randn(8, 1),
        torch.randn(8, 5),
        torch.zeros(8, 1),
    ]
    logs = planner.train_step(batch)
    assert "cql_loss" in logs
    assert logs["cql_alpha"] == 0.01


def test_q_grid_action_diagnostics_returns_argmax_and_slope():
    planner = IQLPlanner(
        IQLPlannerConfig(
            state_dim=5,
            action_dim=2,
            max_action=1.0,
            hidden_dim=16,
            n_hidden=1,
            max_steps=10,
            device="cpu",
        )
    )
    obs = torch.randn(4, 5)
    grid = _make_action_grid(action_dim=2, grid_points=3, device="cpu", dtype=torch.float32)
    diag = _q_grid_action_diagnostics(planner, obs, grid, max_action=1.0)
    assert diag["q_argmax"].shape == (4, 2)
    assert diag["q_slope"].shape == (4,)
    assert np.isfinite(diag["q_argmax"]).all()

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
