"""
Unit tests for CT+IQL EM: E-step freezes encoder; M-step encoder grad only on Q-step.
"""
import numpy as np
import pytest
import torch
from torch.distributions import Normal
from torch.utils.data import DataLoader
from omegaconf import OmegaConf

from src.data.cip_dataset import CIPDataset
from src.data.cancer_sim_cont.cancer_simulation import simulate_factual
from src.data.ct_transition_dataset import CTTransitionDataset, _collate_pad_H
from src.data.iql_dataset_builder import dataset_actions_to_tanh_policy_space
from src.data.iql_raw_transition_dataset import (
    IQLRawBatch,
    IQLRawReplayBuffer,
    build_iql_raw_transitions,
)
from src.evaluation.iql_action_selection import select_iql_policy_action
from src.evaluation.iql_planner_eval import (
    _build_decision_history_view,
    _extend_h_work_after_one_step,
    _make_action_grid,
    _q_grid_action_diagnostics,
)
from src.models.ct_deconfound import CTDeconfoundModel
from src.models.ct_encoder_weight import CTEncoderWeightModel, _stratified_permutation
from src.models.ct_history_encoder import CTHistoryEncoder
from src.models.sequence_utils import gather_last_valid, last_valid_indices, last_valid_mask
from src.planners.iql_planner import IQLPlanner, IQLPlannerConfig, _weighted_mean
from src.utils.em_config import empty_replay_error


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


def test_tumor_factual_last_active_outcome_is_simulated():
    """The final active processed row must not point past the simulated trajectory."""
    n_patients = 2
    seq_length = 6
    params = {
        "initial_stages": np.zeros(n_patients, dtype=np.float64),
        "initial_volumes": np.ones(n_patients, dtype=np.float64),
        "alpha": np.full(n_patients, 0.01, dtype=np.float64),
        "rho": np.full(n_patients, 0.001, dtype=np.float64),
        "beta": np.full(n_patients, 0.001, dtype=np.float64),
        "beta_c": np.full(n_patients, 0.001, dtype=np.float64),
        "K": np.full(n_patients, 1000.0, dtype=np.float64),
        "patient_types": np.zeros(n_patients, dtype=np.float64),
        "window_size": 2,
        "lag": 0,
        "chemo_sigmoid_intercepts": np.ones(n_patients, dtype=np.float64),
        "radio_sigmoid_intercepts": np.ones(n_patients, dtype=np.float64),
        "chemo_sigmoid_betas": np.ones(n_patients, dtype=np.float64),
        "radio_sigmoid_betas": np.ones(n_patients, dtype=np.float64),
    }
    assigned_actions = np.full((n_patients, seq_length, 2), 0.5, dtype=np.float64)

    np.random.seed(7)
    data = simulate_factual(params, seq_length, assigned_actions=assigned_actions)

    assert np.all(data["sequence_lengths"] == seq_length - 1)
    for patient_idx, length in enumerate(data["sequence_lengths"].astype(int)):
        # Processed outputs are cancer_volume[:, 1:], so processed row length - 1
        # corresponds to raw cancer_volume index length.
        assert data["cancer_volume"][patient_idx, length] > 0.0


def test_cip_dataset_uses_exp_seed_for_history_lengths():
    cfg = OmegaConf.create({
        "dataset": {
            "name": "cancer_sim_cont",
            "max_seq_length": 12,
            "min_history_length": 2,
        },
        "model": {"name": "vcip"},
        "exp": {
            "tau": 2,
            "repeats": 6,
            "seed": 10,
        },
    })
    data = {"outputs": np.zeros((3, 20, 1), dtype=np.float32)}

    np.random.seed(10)
    expected = np.unique(np.random.randint(2, 9, 5))
    dataset = CIPDataset(data, cfg, train=False)

    np.testing.assert_array_equal(dataset.history_lengths, expected)


def test_cip_dataset_accepts_internal_sample_seed_for_validation_repeats():
    cfg = OmegaConf.create({
        "dataset": {
            "name": "cancer_sim_cont",
            "max_seq_length": 12,
            "min_history_length": 2,
        },
        "model": {"name": "vcip"},
        "exp": {
            "tau": 2,
            "repeats": 6,
            "seed": 10,
        },
    })
    data = {"outputs": np.zeros((3, 20, 1), dtype=np.float32)}

    np.random.seed(1019)
    expected = np.unique(np.random.randint(2, 9, 5))
    dataset = CIPDataset(data, cfg, train=False, sample_seed=1019)

    np.testing.assert_array_equal(dataset.history_lengths, expected)


def test_cip_dataset_uses_fixed_repeats_for_mimic_and_tumor():
    base = {
        "model": {"name": "vcip"},
        "exp": {
            "tau": 2,
            "repeats": 99,
            "seed": 10,
        },
    }
    data = {"outputs": np.zeros((3, 80, 1), dtype=np.float32)}

    mimic_cfg = OmegaConf.create({
        **base,
        "dataset": {
            "name": "mimic3_synthetic_gift",
            "max_seq_length": 60,
            "min_seq_length": 60,
            "min_history_length": 20,
        },
    })
    tumor_cfg = OmegaConf.create({
        **base,
        "dataset": {
            "name": "tumor_generator",
            "max_seq_length": 60,
            "min_history_length": 20,
        },
    })

    assert CIPDataset(data, mimic_cfg, train=False).repeats == 3
    assert CIPDataset(data, tumor_cfg, train=False).repeats == 5


def test_last_valid_gather_after_collate_padding():
    H = _variable_padded_batch()

    assert torch.allclose(H["current_treatments"][0, -1, :], torch.zeros(2))
    assert torch.equal(last_valid_indices(H["active_entries"]), torch.tensor([1, 3]))
    assert torch.allclose(last_valid_mask(H["active_entries"]), torch.ones(2))

    A_t = gather_last_valid(H["current_treatments"], H["active_entries"])
    assert torch.allclose(A_t[0], torch.tensor([12.0, 13.0]))
    assert torch.allclose(A_t[1], torch.tensor([26.0, 27.0]))


def test_last_valid_indices_rightmost_active_with_noncontiguous_mask():
    """Open decision rows appended after padding yield [1,1,0,0,1]; use rightmost 1."""
    active = torch.tensor([[[1.0], [1.0], [0.0], [0.0], [1.0]]])
    outputs = torch.tensor([[[1.0], [2.0], [7.0], [6.0], [2.0]]])

    assert torch.equal(last_valid_indices(active), torch.tensor([4]))
    assert torch.allclose(gather_last_valid(outputs, active), torch.tensor([[2.0]]))

    # Contiguous prefix padding still matches the old sum-based answer.
    contiguous = torch.tensor([[[1.0], [1.0], [0.0], [0.0]]])
    assert torch.equal(last_valid_indices(contiguous), torch.tensor([1]))

    H = {
        "prev_treatments": torch.tensor([[[0.1, 0.2], [0.3, 0.4], [0.0, 0.0], [0.0, 0.0]]]),
        "current_treatments": torch.tensor([[[1.0, 0.0], [0.5, 0.5], [9.9, 9.9], [8.8, 8.8]]]),
        "prev_outputs": torch.tensor([[[0.0], [1.0], [0.0], [0.0]]]),
        "outputs": torch.tensor([[[1.0], [2.0], [7.0], [6.0]]]),
        "active_entries": torch.tensor([[[1.0], [1.0], [0.0], [0.0]]]),
    }
    H_dec = _build_decision_history_view(H)
    assert torch.equal(
        H_dec["active_entries"].squeeze(-1),
        torch.tensor([[1.0, 1.0, 0.0, 0.0, 1.0]]),
    )
    assert torch.equal(last_valid_indices(H_dec["active_entries"]), torch.tensor([4]))
    assert torch.allclose(
        gather_last_valid(H_dec["outputs"], H_dec["active_entries"]),
        torch.tensor([[2.0]]),
    )
    assert torch.allclose(H_dec["outputs"][:, -1, :], torch.tensor([[2.0]]))


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
        assert transition.t_target == transition.t
        assert transition.delta_t_next_norm == 0.0
        assert transition.done == 1.0


def test_raw_transition_indices_follow_processed_row_contract():
    data = _raw_iql_data(length=5)
    data["prev_outputs"][0, :, 0] = np.array([0.0, 10.0, 20.0, 30.0, 40.0], dtype=np.float32)
    data["outputs"][0, :, 0] = np.array([5.0, 25.0, 80.0, 160.0, 320.0], dtype=np.float32)

    transitions = build_iql_raw_transitions(
        data,
        max_tau=3,
        reward_clip=0.0,
        reward_scale="none",
        samples_per_transition=1,
        target_sampling="horizon_aligned",
        target_horizons=[2],
        horizon_terminal_done=True,
        seed=0,
    )
    tr = next(t for t in transitions if t.t == 1)

    assert tr.t_target == 2
    assert tr.delta_t_norm == pytest.approx(2.0 / 3.0)
    assert tr.delta_t_next_norm == pytest.approx(1.0 / 3.0)
    assert tr.done == 0.0
    assert float(tr.y_target.item()) == pytest.approx(80.0)
    assert tr.reward == pytest.approx(abs(10.0 - 80.0) - abs(25.0 - 80.0))


def test_ct_transition_dataset_uses_same_row_output_after_action():
    data = _raw_iql_data(length=4)
    data["outputs"][0, :, 0] = np.array([3.0, 7.0, 11.0, 19.0], dtype=np.float32)
    ds = CTTransitionDataset(data)

    sample = ds[0]
    assert ds.index[0] == (0, 1)
    assert torch.allclose(sample["y_next"], torch.tensor([7.0]))


def test_decision_history_view_exposes_latest_outcome_after_closed_rollout():
    H = {
        "prev_treatments": torch.tensor([[[0.0, 0.0], [0.1, 0.2]]], dtype=torch.float32),
        "current_treatments": torch.tensor([[[0.1, 0.2], [0.3, 0.4]]], dtype=torch.float32),
        "prev_outputs": torch.tensor([[[0.0], [1.0]]], dtype=torch.float32),
        "outputs": torch.tensor([[[1.0], [2.0]]], dtype=torch.float32),
        "active_entries": torch.ones(1, 2, 1),
        "static_features": torch.ones(1, 2, 1),
        "current_covariates": torch.tensor([[[10.0, 11.0], [12.0, 13.0]]]),
    }

    H_dec = _build_decision_history_view(H)
    assert H["outputs"].shape[1] == 2
    assert H_dec["outputs"].shape[1] == 3
    assert torch.allclose(H_dec["prev_outputs"][0, -1, :], torch.tensor([2.0]))
    assert torch.allclose(H_dec["prev_treatments"][0, -1, :], torch.tensor([0.3, 0.4]))
    assert torch.allclose(H_dec["current_covariates"][0, -1, :], torch.tensor([12.0, 13.0]))

    _extend_h_work_after_one_step(
        H,
        torch.tensor([[0.7, 0.8]], dtype=torch.float32),
        torch.tensor([[3.0]], dtype=torch.float32),
        {"output_means": [0.0], "output_stds": [1.0]},
        torch.device("cpu"),
    )
    assert torch.allclose(H["outputs"][0, -1, :], torch.tensor([3.0]))
    assert torch.allclose(H["prev_outputs"][0, -1, :], torch.tensor([2.0]))
    assert torch.allclose(H["current_covariates"][0, -1, :], torch.tensor([12.0, 13.0]))

    H_dec2 = _build_decision_history_view(H)
    assert torch.allclose(H_dec2["prev_outputs"][0, -1, :], torch.tensor([3.0]))
    assert torch.allclose(H_dec2["prev_treatments"][0, -1, :], torch.tensor([0.7, 0.8]))


def test_empty_replay_sampling_fails_readably():
    replay = IQLRawReplayBuffer([], device="cpu")
    with pytest.raises(ValueError, match="empty IQLRawReplayBuffer"):
        replay.sample(1)


def test_empty_replay_error_includes_context():
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


def test_without_weightnet_uses_exact_unit_weights():
    model = CTEncoderWeightModel(_tiny_cfg(), x_dim=4).to("cpu")
    Z_t = torch.randn(8, model.z_dim)
    A_t = torch.randn(8, model.treatment_dim)

    for parameter in model.weight_net.parameters():
        parameter.grad = None
    logits, weights = model.compute_weights(Z_t, A_t, uniform=True)

    assert torch.equal(logits, torch.zeros_like(logits))
    assert torch.equal(weights, torch.ones_like(weights))
    assert not logits.requires_grad
    assert not weights.requires_grad
    assert all(parameter.grad is None for parameter in model.weight_net.parameters())


def test_estep_marginal_permutation_never_crosses_time_strata():
    strata = torch.tensor([1, 1, 1, 2, 2, 3, 3, 3, 3])
    generator = torch.Generator().manual_seed(23)
    permutation = _stratified_permutation(strata, generator=generator)

    assert torch.equal(strata[permutation], strata)
    assert sorted(permutation.tolist()) == list(range(strata.numel()))
    assert any(int(src) != dst for dst, src in enumerate(permutation))


def test_weightnet_scores_are_bounded_like_reference_implementation():
    model = CTEncoderWeightModel(_tiny_cfg(), x_dim=4)
    scores = model.weight_net(torch.randn(32, model.z_dim + model.treatment_dim))

    assert torch.all(scores > 0.0)
    assert torch.all(scores < 1.0)


def test_sinkhorn_estep_updates_once_per_epoch_after_averaging_time_losses():
    class CountingAdam(torch.optim.Adam):
        def __init__(self, params, **kwargs):
            super().__init__(params, **kwargs)
            self.step_calls = 0

        def step(self, closure=None):
            self.step_calls += 1
            return super().step(closure)

    torch.manual_seed(29)
    model = CTEncoderWeightModel(_tiny_cfg(), x_dim=4).to("cpu")
    optimizer_w = CountingAdam(model.weight_net.parameters(), lr=1e-2)
    H = _fake_H(8, 3, "cpu")
    samples = [
        {
            "H_t": {key: value[idx] for key, value in H.items()},
            "time_index": torch.tensor(1 if idx < 4 else 2),
        }
        for idx in range(H["outputs"].size(0))
    ]

    metrics = model.e_step_full_dataset(
        DataLoader(samples, batch_size=4, shuffle=False),
        optimizer_w,
        align_mode="sinkhorn",
        sinkhorn_blur=0.01,
        e_epochs=3,
        train_batch_size=2,
        w_clip=5.0,
        device="cpu",
        outer_seed=31,
    )

    assert optimizer_w.step_calls == 3
    assert metrics["e_optimizer_steps"] == 3.0
    assert metrics["n_time_strata"] == 2.0


@pytest.mark.parametrize("align_mode", ["sinkhorn", "mmd"])
def test_full_dataset_e_step_supports_alignment_modes(align_mode):
    torch.manual_seed(17)
    cfg = _tiny_cfg()
    model = CTEncoderWeightModel(cfg, x_dim=4).to("cpu")
    optimizer_w = torch.optim.Adam(model.weight_net.parameters(), lr=1e-2)
    H = _fake_H(8, 3, "cpu")
    samples = [
        {"H_t": {key: value[idx] for key, value in H.items()}}
        for idx in range(H["outputs"].size(0))
    ]

    metrics = model.e_step_full_dataset(
        DataLoader(samples, batch_size=4, shuffle=False),
        optimizer_w,
        align_mode=align_mode,
        sinkhorn_blur=0.01,
        e_epochs=1,
        train_batch_size=4,
        w_clip=5.0,
        device="cpu",
        outer_seed=23,
    )

    assert metrics["n_samples"] == 8.0
    assert np.isfinite(metrics["align_pre"])
    assert np.isfinite(metrics["align_post"])
    assert np.isfinite(metrics["w_ess_frac"])
    assert np.isfinite(metrics["w_std"])


def test_full_dataset_e_step_rejects_unknown_alignment_mode():
    cfg = _tiny_cfg()
    model = CTEncoderWeightModel(cfg, x_dim=4).to("cpu")
    optimizer_w = torch.optim.Adam(model.weight_net.parameters(), lr=1e-2)
    H = _fake_H(2, 3, "cpu")
    samples = [
        {"H_t": {key: value[idx] for key, value in H.items()}}
        for idx in range(H["outputs"].size(0))
    ]

    with pytest.raises(ValueError, match="sinkhorn.*mmd"):
        model.e_step_full_dataset(
            DataLoader(samples, batch_size=2),
            optimizer_w,
            align_mode="invalid",
            sinkhorn_blur=0.01,
            e_epochs=1,
            train_batch_size=2,
            device="cpu",
        )



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


def test_mse_actor_bc_loss_optimizes_actor_mean_not_log_std():
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
            actor_bc_loss="mse",
        )
    )
    obs = torch.randn(8, 5)
    actions = torch.clamp(torch.randn(8, 2), -1.0, 1.0)
    adv = torch.randn(8)
    before_log_std = planner.actor.log_std.detach().clone()
    logs = {}
    planner._update_policy(adv, obs, actions, logs)
    assert logs["actor_update_bc"] == 1.0
    assert logs["actor_bc_loss"] >= 0.0
    assert planner.actor.log_std.grad is None
    assert torch.allclose(planner.actor.log_std.detach(), before_log_std)


def test_expectile_actor_bc_loss_weights_under_prediction_more():
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
            actor_bc_loss="expectile",
            actor_bc_expectile=0.8,
        )
    )
    pred = torch.zeros(2, 2)
    target = torch.tensor([[0.5, 0.5], [-0.5, -0.5]])
    losses = planner._policy_bc_losses(pred, target)
    assert torch.allclose(losses, torch.tensor([0.4, 0.1]), atol=1e-6)


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


def test_awr_td3bc_action_penalty_logs_support_constraint():
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
            td3bc_action_penalty_alpha=3.0,
        )
    )
    obs = torch.randn(8, 5)
    actions = torch.clamp(torch.randn(8, 2), -1.0, 1.0)
    adv = torch.randn(8)
    for p in planner.qf.parameters():
        p.grad = None
    logs = {}
    planner._update_policy(adv, obs, actions, logs)
    assert logs["actor_update_awr_td3bc"] == 1.0
    assert logs["actor_awr_td3bc_action_penalty"] >= 0.0
    assert logs["actor_awr_td3bc_action_penalty_alpha"] == 3.0
    assert all(p.grad is None for p in planner.qf.parameters())


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


def test_q_high_action_penalty_logs_when_enabled():
    planner = IQLPlanner(
        IQLPlannerConfig(
            state_dim=5,
            action_dim=2,
            max_action=1.0,
            hidden_dim=16,
            n_hidden=1,
            max_steps=10,
            device="cpu",
            q_high_action_penalty_alpha=0.2,
            q_high_action_penalty_n_actions=2,
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
    assert "q_high_action_penalty" in logs
    assert logs["q_high_action_penalty_alpha"] == 0.2
    assert 0.0 <= logs["q_high_action_penalty_positive_frac"] <= 1.0


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


def test_q_sample_action_selector_keeps_mean_default_and_filters_by_q():
    class DummyActor(torch.nn.Module):
        max_action = 1.0

        def forward(self, obs):
            mean = torch.zeros(obs.size(0), 2, device=obs.device)
            std = torch.ones_like(mean)
            return Normal(mean, std)

    class SumQ(torch.nn.Module):
        def forward(self, obs, action):
            return action.sum(dim=-1)

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
    planner.actor = DummyActor()
    planner.qf = SumQ()
    obs = torch.zeros(6, 5)

    mean_action = select_iql_policy_action(planner, obs, selector="mean")
    assert torch.allclose(mean_action, torch.zeros_like(mean_action))

    torch.manual_seed(0)
    q_action = select_iql_policy_action(
        planner,
        obs,
        selector="q_sample",
        candidate_actions=128,
        q_bc_penalty=0.0,
    )
    assert q_action.shape == (6, 2)
    assert torch.all(q_action <= 1.0)
    assert torch.all(q_action >= -1.0)
    assert torch.all(q_action.sum(dim=-1) >= mean_action.sum(dim=-1))


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
