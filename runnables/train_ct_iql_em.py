"""
CT+IQL unified EM training (planning-oriented representation learning).

E-step: fix encoder, Q, V, π; update WeightNet only.
M-step: fix WeightNet; weighted V→Q→π per batch; only Q-step updates encoder.
"""
import logging
import os
import sys
import ast
import random
from contextlib import contextmanager
from pathlib import Path
from typing import Dict

import hydra
import numpy as np
import torch
from hydra.utils import get_original_cwd, instantiate
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.ct_transition_dataset import CTEstepDataset, collate_ct_estep_batch, _covariate_stream_dim
from src.data.iql_raw_transition_dataset import IQLRawReplayBuffer, build_iql_raw_transitions
from src.evaluation.iql_planner_eval import aggregate_iql_planner_metrics
from src.models.ct_encoder_weight import CTEncoderWeightModel
from src.models.inference_model import InferenceModel
from src.planners.iql_planner import IQLPlanner, IQLPlannerConfig
from src.training.ct_iql_em_loop import EMTrainConfig, run_e_step_full, run_m_step_steps
from src.utils.em_ckpt import load_encoder_into_inference, save_em_checkpoint
from src.utils.em_config import empty_replay_error as _empty_replay_error
from src.utils.stable_iql_em_defaults import stable_select
from src.utils.mlflow_vcip import VCIPMlflowTracker
from src.utils.utils import repeat_static, set_seed, to_float

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

OmegaConf.register_new_resolver("toint", lambda x: int(x), replace=True)

VAL_METRIC_KEYS = (
    "mae_uns", "mae_norm", "rmse_uns", "rmse_norm", "rmse_norm_x_std",
)


def _state_dict_to_cpu(obj):
    if torch.is_tensor(obj):
        return obj.detach().cpu().clone()
    if isinstance(obj, dict):
        return {k: _state_dict_to_cpu(v) for k, v in obj.items()}
    return obj


def _list_from_config(value):
    if value is None:
        return None
    if OmegaConf.is_config(value):
        value = OmegaConf.to_container(value, resolve=True)
    if isinstance(value, str):
        raw = value.strip()
        if raw.startswith("["):
            value = ast.literal_eval(raw)
        else:
            value = [x.strip() for x in raw.split(",") if x.strip()]
    return [int(v) for v in value]


@contextmanager
def _isolated_rng(seed: int):
    py_state = random.getstate()
    np_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        seed = int(seed)
        random.seed(seed)
        np.random.seed(seed % (2**32 - 1))
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        yield
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)
        torch.set_rng_state(torch_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)


def _mean_metric(metric_rows, key: str):
    vals = [float(row[key]) for row in metric_rows if row.get(key) is not None]
    if not vals:
        return None
    return float(sum(vals) / len(vals))


def _dataset_run_slug(args: DictConfig) -> str:
    name = str(OmegaConf.select(args, "dataset.name", default="dataset")).replace("/", "_")
    coeff = OmegaConf.select(args, "dataset.coeff", default=None)
    if coeff is not None:
        return f"seed_{int(args.exp.seed)}_gamma_{int(coeff)}"
    return f"seed_{int(args.exp.seed)}_{name}"


@hydra.main(version_base=None, config_name="config.yaml", config_path="../configs/")
def main(args: DictConfig):
    OmegaConf.set_struct(args, False)
    logger.info("\n" + OmegaConf.to_yaml(args, resolve=True))

    set_seed(int(args.exp.seed))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    original_cwd = Path(get_original_cwd())
    args["exp"]["processed_data_dir"] = os.path.join(str(original_cwd), args["exp"]["processed_data_dir"])

    em_outer_iters = int(stable_select(args, "exp.em_outer_iters"))
    em_m_steps = int(stable_select(args, "exp.em_m_steps_per_outer"))
    em_encoder_lr = float(stable_select(args, "exp.em_encoder_lr"))
    em_val_every = int(stable_select(args, "exp.em_val_every"))
    em_val_repeats = max(1, int(stable_select(args, "exp.em_val_repeats")))
    em_val_seed_offset = int(stable_select(args, "exp.em_val_seed_offset"))
    em_val_seed_base = int(args.exp.seed) + em_val_seed_offset
    em_save_every_eval_checkpoint = bool(stable_select(args, "exp.em_save_every_eval_checkpoint"))
    em_save_every_outer_checkpoint = bool(stable_select(args, "exp.em_save_every_outer_checkpoint"))
    em_warmup = int(stable_select(args, "exp.em_warmup_outer_iters"))
    em_log_m_every = int(stable_select(args, "exp.em_log_m_every"))
    em_e_epochs = max(1, int(stable_select(args, "exp.em_e_epochs")))
    em_encoder_diagnostics = bool(stable_select(args, "exp.em_encoder_diagnostics"))
    em_encoder_diagnostics_every = max(
        1,
        int(
            stable_select(args, "exp.em_encoder_diagnostics_every", em_log_m_every)
        ),
    )
    em_e_refresh_every = int(stable_select(args, "exp.em_e_refresh_every"))
    em_her_refresh_every = int(stable_select(args, "exp.em_her_refresh_every"))
    em_her_samples_per_transition = max(
        1,
        int(stable_select(args, "exp.em_her_samples_per_transition")),
    )
    iql_target_sampling = str(stable_select(args, "exp.iql_target_sampling"))
    iql_target_horizons = _list_from_config(
        stable_select(args, "exp.iql_target_horizons")
    )
    iql_horizon_terminal_done = bool(stable_select(args, "exp.iql_horizon_terminal_done"))

    ct_use_weight_net = bool(stable_select(args, "exp.ct_use_weight_net"))
    ct_align = str(stable_select(args, "exp.ct_align_loss"))
    if ct_align not in {"sinkhorn", "mmd"}:
        raise ValueError(
            f"EM training requires ct_align_loss='sinkhorn' or 'mmd', got {ct_align!r}"
        )
    ct_blur = float(stable_select(args, "exp.ct_sinkhorn_blur"))
    _em_e_w_lr = stable_select(args, "exp.em_e_w_lr")
    w_lr = (
        float(_em_e_w_lr)
        if _em_e_w_lr is not None
        else float(stable_select(args, "exp.ct_w_lr"))
    )
    _ct_w_clip = stable_select(args, "exp.ct_w_clip")
    w_clip = float(_ct_w_clip) if _ct_w_clip is not None else None
    ct_wd = float(stable_select(args, "exp.ct_weight_decay"))
    ct_batch_size = int(stable_select(args, "exp.ct_batch_size"))
    m_batch_size = int(stable_select(args, "exp.iql_batch_size"))

    dataset_collection = instantiate(args.dataset, _recursive_=True)
    dataset_collection.process_data_multi()
    dataset_collection = to_float(dataset_collection)
    if int(args.dataset.static_size) > 0:
        dims = len(dataset_collection.train_f.data["static_features"].shape)
        if dims == 2:
            dataset_collection = repeat_static(dataset_collection)

    ds_dict = OmegaConf.to_container(args["dataset"], resolve=True)
    x_dim = _covariate_stream_dim(ds_dict)
    ct_model = CTEncoderWeightModel(args, x_dim).to(device)
    logger.info(
        "CTHistoryEncoder layers | local=%d global=%d total=%d",
        int(getattr(ct_model.ct_encoder, "local_conv_layers", 0)),
        int(getattr(ct_model.ct_encoder, "global_attention_layers", 0)),
        int(getattr(ct_model.ct_encoder, "num_layers", 0)),
    )
    if int(getattr(ct_model.ct_encoder, "local_conv_layers", 0)) <= 0:
        logger.warning("local_conv_layers <= 0: local-global encoder is effectively all global attention.")
    if int(getattr(ct_model.ct_encoder, "global_attention_layers", 0)) <= 0:
        logger.warning("global_attention_layers <= 0: local-global encoder has no global attention layer.")

    z_dim = int(stable_select(args, "model.z_dim"))
    out_dim = int(args.dataset.output_size)
    act_dim = int(args.dataset.treatment_size)
    state_dim = z_dim + out_dim + 1 + act_dim
    goal_adapter_enabled = bool(stable_select(args, "exp.iql_goal_adapter_enabled"))
    goal_adapter_hidden_dim = int(stable_select(args, "exp.iql_goal_adapter_hidden_dim"))
    goal_adapter_init_scale = float(stable_select(args, "exp.iql_goal_adapter_init_scale"))

    max_action = float(stable_select(args, "exp.iql_max_action"))
    max_tau = float(stable_select(args, "exp.max_tau"))
    iql_max_grad = stable_select(args, "exp.iql_max_grad_norm")
    iql_max_grad = None if iql_max_grad is None else float(iql_max_grad)
    enc_max_grad = stable_select(args, "exp.em_encoder_max_grad_norm")
    enc_max_grad = None if enc_max_grad is None else float(enc_max_grad)
    iql_weight_max = stable_select(args, "exp.iql_weight_max")
    iql_weight_max = None if iql_weight_max is None else float(iql_weight_max)

    planner_cfg = IQLPlannerConfig(
        state_dim=state_dim,
        action_dim=act_dim,
        max_action=max_action,
        hidden_dim=int(stable_select(args, "exp.iql_hidden_dim")),
        n_hidden=int(stable_select(args, "exp.iql_n_hidden")),
        iql_tau=float(stable_select(args, "exp.iql_tau")),
        beta=float(stable_select(args, "exp.iql_beta")),
        adv_max=float(stable_select(args, "exp.iql_adv_max")),
        weight_max=iql_weight_max,
        actor_update=str(stable_select(args, "exp.iql_actor_update")),
        actor_bc_loss=str(stable_select(args, "exp.iql_actor_bc_loss")),
        actor_bc_expectile=float(stable_select(args, "exp.iql_actor_bc_expectile")),
        td3bc_q_alpha=float(stable_select(args, "exp.iql_td3bc_q_alpha")),
        td3bc_bc_alpha=float(stable_select(args, "exp.iql_td3bc_bc_alpha")),
        td3bc_action_penalty_alpha=float(stable_select(args, "exp.iql_td3bc_action_penalty_alpha")),
        cql_alpha=float(stable_select(args, "exp.iql_cql_alpha")),
        cql_n_actions=int(stable_select(args, "exp.iql_cql_n_actions")),
        q_high_action_penalty_alpha=float(stable_select(args, "exp.iql_q_high_action_penalty_alpha")),
        q_high_action_penalty_margin=float(stable_select(args, "exp.iql_q_high_action_penalty_margin")),
        q_high_action_penalty_n_actions=int(stable_select(args, "exp.iql_q_high_action_penalty_n_actions")),
        discount=float(stable_select(args, "exp.iql_discount")),
        tau=float(stable_select(args, "exp.iql_target_tau")),
        actor_lr=float(stable_select(args, "exp.iql_actor_lr")),
        qf_lr=float(stable_select(args, "exp.iql_qf_lr")),
        vf_lr=float(stable_select(args, "exp.iql_vf_lr")),
        max_steps=em_outer_iters * em_m_steps,
        deterministic_actor=bool(stable_select(args, "exp.iql_deterministic")),
        actor_dropout=stable_select(args, "exp.iql_actor_dropout"),
        max_grad_norm=iql_max_grad,
        encoder_max_grad_norm=enc_max_grad,
        device=device,
        goal_adapter_enabled=goal_adapter_enabled,
        z_dim=z_dim,
        output_dim=out_dim,
        goal_adapter_hidden_dim=goal_adapter_hidden_dim,
        goal_adapter_init_scale=goal_adapter_init_scale,
    )
    planner = IQLPlanner(planner_cfg)
    if goal_adapter_enabled:
        logger.info(
            "IQL GoalAdapter enabled | input=[Z_t,y_target,delta] z_dim=%d output_dim=%d hidden=%d init_scale=%.1e",
            z_dim,
            out_dim,
            goal_adapter_hidden_dim,
            goal_adapter_init_scale,
        )

    optimizer_w = torch.optim.Adam(ct_model.weight_net.parameters(), lr=w_lr, weight_decay=ct_wd)
    optimizer_enc = torch.optim.Adam(
        list(ct_model.encoder_parameters()) + planner.goal_adapter_parameters(),
        lr=em_encoder_lr,
        weight_decay=ct_wd,
    )

    def _build_replay(her_seed: int) -> IQLRawReplayBuffer:
        max_patients = stable_select(args, "exp.iql_max_patients")
        raw = build_iql_raw_transitions(
            data=dataset_collection.train_f.data,
            reward_type=str(stable_select(args, "exp.iql_reward_type")),
            max_patients=max_patients,
            max_action=max_action,
            dataset_actions_unit_interval=bool(stable_select(args, "exp.iql_dataset_actions_unit_interval")),
            max_tau=max_tau,
            reward_clip=float(stable_select(args, "exp.iql_reward_clip")),
            reward_scale=str(stable_select(args, "exp.iql_reward_scale")),
            reward_huber_delta=float(stable_select(args, "exp.iql_reward_huber_delta")),
            samples_per_transition=em_her_samples_per_transition,
            target_sampling=iql_target_sampling,
            target_horizons=iql_target_horizons,
            horizon_terminal_done=iql_horizon_terminal_done,
            decision_interval_days=int(stable_select(args, "exp.iql_decision_interval_days")),
            seed=her_seed,
        )
        if not raw:
            raise ValueError(
                _empty_replay_error(
                    dataset_collection.train_f.data,
                    max_patients=max_patients,
                    target_sampling=iql_target_sampling,
                    target_horizons=iql_target_horizons,
                    max_tau=max_tau,
                    samples_per_transition=em_her_samples_per_transition,
                )
            )
        return IQLRawReplayBuffer(raw, device=device)

    base_seed = int(args.exp.seed)
    replay = None
    if em_her_refresh_every <= 0:
        replay = _build_replay(base_seed)
        logger.info(
            "HER buffer built once (em_her_refresh_every=0, samples_per_transition=%d) | transitions=%d",
            em_her_samples_per_transition,
            replay.size,
        )
        logger.info(
            "HER target sampling | mode=%s horizons=%s horizon_terminal_done=%s",
            iql_target_sampling,
            iql_target_horizons,
            iql_horizon_terminal_done,
        )

    ct_loader = DataLoader(
        CTEstepDataset(dataset_collection.train_f.data),
        batch_size=ct_batch_size,
        shuffle=False,
        num_workers=int(stable_select(args, "exp.ct_num_workers")),
        collate_fn=collate_ct_estep_batch,
        drop_last=False,
    )

    em_cfg = EMTrainConfig(
        use_weight_net=ct_use_weight_net,
        align_mode=ct_align,
        sinkhorn_blur=ct_blur,
        w_clip=w_clip,
        weight_max=iql_weight_max,
        m_batch_size=m_batch_size,
        warmup_outer_iters=em_warmup,
        log_m_every=em_log_m_every,
        e_epochs=em_e_epochs,
        e_batch_size=ct_batch_size,
        encoder_diagnostics=em_encoder_diagnostics,
        encoder_diagnostics_every=em_encoder_diagnostics_every,
    )
    if em_encoder_diagnostics:
        logger.info(
            "Encoder diagnostics enabled | every=%d M-steps | metrics=grad/preclip, grad/postclip, update, update_ratio",
            em_encoder_diagnostics_every,
        )
    logger.info(
        "WeightNet mode | enabled=%s align_loss=%s",
        ct_use_weight_net,
        ct_align if ct_use_weight_net else "none (uniform weights)",
    )

    _ckpt_override = stable_select(args, "exp.em_ckpt_dir")
    if _ckpt_override:
        out_dir = Path(str(_ckpt_override))
        if not out_dir.is_absolute():
            out_dir = original_cwd / out_dir
    else:
        out_dir = (
            original_cwd
            / "em_checkpoints"
            / _dataset_run_slug(args)
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "ct_iql_em_best.pt"

    val_metric_key = str(stable_select(args, "exp.em_val_metric")).strip().lower()
    if val_metric_key not in VAL_METRIC_KEYS:
        raise ValueError(
            f"exp.em_val_metric must be one of {VAL_METRIC_KEYS}, got {val_metric_key!r}"
        )
    val_prefix = "closed_loop"
    eval_tau = int(args.exp.tau)
    em_val_tau_list = _list_from_config(
        stable_select(args, "exp.em_val_tau_list")
    )
    val_action_diag = bool(stable_select(args, "exp.iql_val_action_diagnostics"))
    val_action_grid_points = int(stable_select(args, "exp.iql_val_action_grid_points"))
    val_action_diag_max_batches = stable_select(args, "exp.iql_val_action_diag_max_batches")
    val_action_diag_max_batches = None if val_action_diag_max_batches is None else int(val_action_diag_max_batches)
    if em_val_tau_list is None:
        em_val_tau_list = [eval_tau]
    val_tau_agg = str(stable_select(args, "exp.em_val_tau_agg")).strip().lower()
    if val_tau_agg not in {"mean", "max"}:
        raise ValueError("exp.em_val_tau_agg must be one of {'mean', 'max'}.")
    if len(em_val_tau_list) == 1:
        selection_metric_key = val_metric_key
    else:
        tau_prefix = "mean" if val_tau_agg == "mean" else "max"
        selection_metric_key = (
            f"{tau_prefix}_{val_metric_key}_tau"
            + "_".join(str(int(t)) for t in em_val_tau_list)
        )
    val_bs = int(stable_select(args, "exp.iql_val_batch_size") or args.exp.batch_size_val)
    autoreg = bool(stable_select(args, "exp.iql_eval_autoregressive"))

    inference_model = InferenceModel(args).to(device)
    best_val = float("inf")
    best_outer = 0
    logger.info(
        "EM validation RNG isolation: seed_base=%d repeats=%d",
        em_val_seed_base,
        em_val_repeats,
    )
    last_e_metrics: Dict[str, float] = {
        "align_pre": 0.0,
        "align_post": 0.0,
        "w_ess_frac": 1.0,
        "w_std": 0.0,
        "w_var": 0.0,
        "w_max": 1.0,
        "w_p50": 1.0,
        "w_p90": 1.0,
        "w_p95": 1.0,
        "w_p99": 1.0,
        "n_samples": 0.0,
        "n_samples_total": 0.0,
    }

    mlf = VCIPMlflowTracker.from_hydra(args, stage="ct_iql_em")
    mlf.start(args)

    try:
        for outer in range(1, em_outer_iters + 1):
            if em_her_refresh_every > 0 and (outer - 1) % em_her_refresh_every == 0:
                her_seed = base_seed + outer * 10007
                replay = _build_replay(her_seed)
                logger.info(
                    "HER buffer refresh outer=%d/%d | samples_per_transition=%d | transitions=%d | seed=%d",
                    outer,
                    em_outer_iters,
                    em_her_samples_per_transition,
                    replay.size,
                    her_seed,
                )
            elif replay is None:
                raise RuntimeError(
                    "Replay buffer not initialized; set em_her_refresh_every>0 or refresh at outer=1."
                )
            if ct_use_weight_net and ct_model.fixed_transition_weights:
                replay.assign_fixed_weights(ct_model.fixed_transition_weights)

            if not ct_use_weight_net:
                e_metrics = last_e_metrics
            elif em_e_refresh_every <= 0 or (outer - 1) % em_e_refresh_every == 0:
                e_seed = base_seed + outer * 10007 + 17
                e_metrics = run_e_step_full(
                    ct_model,
                    ct_loader,
                    optimizer_w,
                    em_cfg,
                    device,
                    outer_iter=outer,
                    outer_seed=e_seed,
                )
                last_e_metrics = e_metrics
                assignment = replay.assign_fixed_weights(
                    ct_model.fixed_transition_weights
                )
                logger.info(
                    "E-step full fit outer=%d | n_active=%.0f n_total=%.0f "
                    "time_strata=%.0f epochs=%d optimizer_steps=%.0f w_lr=%.1e "
                    "fixed_weights=%.0f unique_keys=%.0f",
                    outer,
                    e_metrics.get("n_samples", 0),
                    e_metrics.get("n_samples_total", e_metrics.get("n_samples", 0)),
                    e_metrics.get("n_time_strata", 0),
                    em_e_epochs,
                    e_metrics.get("e_optimizer_steps", 0),
                    w_lr,
                    assignment["assigned_transitions"],
                    assignment["unique_replay_keys"],
                )
            else:
                e_metrics = last_e_metrics
            m_metrics = run_m_step_steps(
                ct_model, planner, optimizer_enc, replay, em_m_steps, em_cfg, outer_iter=outer
            )
            if not ct_use_weight_net:
                m_warmup = " (uniform)"
            else:
                m_warmup = " (M-warmup)" if outer <= em_warmup else ""
            logger.info(
                "EM outer %d/%d | E(x%d): align_pre=%.4f align_post=%.4f reduction=%.4f "
                "time_ess=%.3f time_w_std=%.4f za_dep=%.4f->%.4f | "
                "M%s: q=%.4f v=%.4f pi=%.4f w_mean=%.3f w_std=%.4f w_ess=%.3f | replay=%d",
                outer,
                em_outer_iters,
                em_e_epochs,
                e_metrics["align_pre"],
                e_metrics["align_post"],
                e_metrics.get("align_reduction", 0.0),
                e_metrics.get("w_time_ess_mean", e_metrics["w_ess_frac"]),
                e_metrics.get("w_time_std_mean", e_metrics.get("w_std", 0.0)),
                e_metrics.get("za_dependence_pre", 0.0),
                e_metrics.get("za_dependence_post", 0.0),
                m_warmup,
                m_metrics["q_loss"],
                m_metrics["value_loss"],
                m_metrics["actor_loss"],
                m_metrics.get("w_mean", 1.0),
                m_metrics.get("w_std", 0.0),
                m_metrics.get("w_ess_frac", 1.0),
                replay.size,
            )
            logger.info(
                "Weight diagnostics outer=%d | mode=%s ess=%.6f max=%.6f var=%.6f "
                "p50=%.6f p90=%.6f p95=%.6f p99=%.6f",
                outer,
                ct_align if ct_use_weight_net else "uniform",
                e_metrics["w_ess_frac"],
                e_metrics.get("w_max", 1.0),
                e_metrics.get("w_var", 0.0),
                e_metrics.get("w_p50", 1.0),
                e_metrics.get("w_p90", 1.0),
                e_metrics.get("w_p95", 1.0),
                e_metrics.get("w_p99", 1.0),
            )
            if em_encoder_diagnostics:
                diag_groups = sorted(
                    k.split("/", 1)[1]
                    for k in m_metrics
                    if k.startswith("enc_update_norm/")
                )
                for group in diag_groups:
                    logger.info(
                        "Encoder diag outer=%d | %s grad=%.3e postclip=%.3e update=%.3e ratio=%.3e",
                        outer,
                        group,
                        m_metrics.get(f"enc_grad_norm/{group}", 0.0),
                        m_metrics.get(f"enc_grad_norm_postclip/{group}", 0.0),
                        m_metrics.get(f"enc_update_norm/{group}", 0.0),
                        m_metrics.get(f"enc_update_ratio/{group}", 0.0),
                    )
                logger.info(
                    "Encoder diag outer=%d | collected_steps=%.0f",
                    outer,
                    m_metrics.get("enc_diag_collected_steps", 0.0),
                )
            m_log_metrics = {
                "e/align_pre": e_metrics["align_pre"],
                "e/align_post": e_metrics["align_post"],
                "e/align_reduction": e_metrics.get("align_reduction", 0.0),
                "e/align_relative_reduction": e_metrics.get("align_relative_reduction", 0.0),
                "e/w_ess_frac": e_metrics["w_ess_frac"],
                "e/w_std": e_metrics.get("w_std", 0.0),
                "e/w_var": e_metrics.get("w_var", 0.0),
                "e/w_max": e_metrics.get("w_max", 1.0),
                "e/w_p50": e_metrics.get("w_p50", 1.0),
                "e/w_p90": e_metrics.get("w_p90", 1.0),
                "e/w_p95": e_metrics.get("w_p95", 1.0),
                "e/w_p99": e_metrics.get("w_p99", 1.0),
                "e/w_time_ess_mean": e_metrics.get("w_time_ess_mean", 1.0),
                "e/w_time_ess_min": e_metrics.get("w_time_ess_min", 1.0),
                "e/w_time_std_mean": e_metrics.get("w_time_std_mean", 0.0),
                "e/w_time_std_max": e_metrics.get("w_time_std_max", 0.0),
                "e/za_dependence_pre": e_metrics.get("za_dependence_pre", 0.0),
                "e/za_dependence_post": e_metrics.get("za_dependence_post", 0.0),
                "e/e_epochs": float(em_e_epochs),
                "e/n_samples": e_metrics.get("n_samples", 0.0),
                "e/n_samples_total": e_metrics.get("n_samples_total", e_metrics.get("n_samples", 0.0)),
                "m/q_loss": m_metrics["q_loss"],
                "m/value_loss": m_metrics["value_loss"],
                "m/actor_loss": m_metrics["actor_loss"],
                "replay/size": float(replay.size),
            }
            for key, value in m_metrics.items():
                if key.startswith(("enc_", "w_")):
                    m_log_metrics[f"m/{key}"] = float(value)
            mlf.log_metrics(m_log_metrics, step=outer)
            if em_save_every_outer_checkpoint:
                save_em_checkpoint(
                    out_dir / f"ct_iql_em_outer{outer:04d}.pt",
                    ct_model=ct_model,
                    planner=planner,
                    outer_iter=outer,
                    config_yaml=OmegaConf.to_yaml(args, resolve=True),
                    extra={
                        "checkpoint_type": "outer",
                        "m_q_loss": m_metrics["q_loss"],
                        "m_value_loss": m_metrics["value_loss"],
                        "m_actor_loss": m_metrics["actor_loss"],
                        "e_align_pre": e_metrics["align_pre"],
                        "e_align_post": e_metrics["align_post"],
                        "e_align_reduction": e_metrics.get("align_reduction", 0.0),
                        "e_w_ess_frac": e_metrics["w_ess_frac"],
                        "e_w_std": e_metrics.get("w_std", 0.0),
                        "e_w_var": e_metrics.get("w_var", 0.0),
                        "e_w_max": e_metrics.get("w_max", 1.0),
                        "e_w_p50": e_metrics.get("w_p50", 1.0),
                        "e_w_p90": e_metrics.get("w_p90", 1.0),
                        "e_w_p95": e_metrics.get("w_p95", 1.0),
                        "e_w_p99": e_metrics.get("w_p99", 1.0),
                        "e_w_time_ess_mean": e_metrics.get("w_time_ess_mean", 1.0),
                        "e_w_time_ess_min": e_metrics.get("w_time_ess_min", 1.0),
                        "e_w_time_std_mean": e_metrics.get("w_time_std_mean", 0.0),
                        "e_za_dependence_pre": e_metrics.get("za_dependence_pre", 0.0),
                        "e_za_dependence_post": e_metrics.get("za_dependence_post", 0.0),
                    },
                )

            if em_val_every > 0 and (outer % em_val_every == 0 or outer == em_outer_iters):
                em_state = {
                    "ct_history_encoder": _state_dict_to_cpu(ct_model.ct_encoder.state_dict()),
                    "projection_head": _state_dict_to_cpu(ct_model.projection.state_dict()),
                    "weight_net": _state_dict_to_cpu(ct_model.weight_net.state_dict()),
                }
                load_encoder_into_inference(inference_model, em_state)
                inference_model.eval()
                planner.actor.eval()
                val_scores = []
                val_log_metrics = {}
                for tau_i in em_val_tau_list:
                    tau_repeat_metrics = []
                    for rep_i in range(em_val_repeats):
                        val_seed = em_val_seed_base + int(tau_i) * 1000 + rep_i
                        with _isolated_rng(val_seed):
                            metrics = aggregate_iql_planner_metrics(
                                planner,
                                inference_model,
                                dataset_collection,
                                dataset_collection.val_f,
                                args,
                                device=device,
                                tau=int(tau_i),
                                max_tau=max_tau,
                                autoregressive_eval=autoreg,
                                val_batch_size=val_bs,
                                log_batches=False,
                                action_diagnostics=val_action_diag and rep_i == 0,
                                action_grid_points=val_action_grid_points,
                                action_diag_max_batches=val_action_diag_max_batches,
                                sample_seed=val_seed,
                            )
                        tau_repeat_metrics.append(metrics)
                    tau_repeat_scores = [float(m[val_metric_key]) for m in tau_repeat_metrics]
                    tau_score = float(sum(tau_repeat_scores) / len(tau_repeat_scores))
                    val_scores.append(tau_score)
                    tau_prefix = f"val/{val_prefix}/tau{int(tau_i)}"
                    val_log_metrics[f"{tau_prefix}/{val_metric_key}"] = tau_score
                    if len(tau_repeat_scores) > 1:
                        repeat_var = sum((x - tau_score) ** 2 for x in tau_repeat_scores) / len(tau_repeat_scores)
                        val_log_metrics[f"{tau_prefix}/{val_metric_key}_repeat_std"] = float(repeat_var ** 0.5)
                    for key in (
                        "mae_uns",
                        "mae_norm",
                        "rmse_uns",
                        "rmse_norm",
                        "rmse_norm_x_std",
                    ):
                        metric_mean = _mean_metric(tau_repeat_metrics, key)
                        if metric_mean is not None:
                            val_log_metrics[f"{tau_prefix}/{key}"] = metric_mean
                    action_diag = tau_repeat_metrics[0].get("action_diagnostics", {})
                    for key in (
                        "planned_mean",
                        "factual_mean",
                        "q_argmax_mean",
                        "sim_best_proxy_mean",
                        "planned_minus_factual_mean",
                        "planned_minus_q_argmax_mean",
                        "planned_minus_sim_best_proxy_mean",
                        "q_slope_mean",
                    ):
                        if action_diag.get(key) is not None:
                            val_log_metrics[f"{tau_prefix}/action/{key}"] = float(action_diag[key])
                val_score_mean = float(sum(val_scores) / len(val_scores))
                val_score_max = float(max(val_scores))
                val_score = val_score_mean if val_tau_agg == "mean" else val_score_max
                if len(em_val_tau_list) == 1:
                    logger.info(
                        "EM val outer=%d %s=%.6f (%s, tau=%d, repeats=%d)",
                        outer,
                        selection_metric_key,
                        val_score,
                        val_prefix,
                        int(em_val_tau_list[0]),
                        em_val_repeats,
                    )
                    val_log_metrics[f"val/{val_prefix}/{val_metric_key}"] = val_score
                else:
                    logger.info(
                        "EM val outer=%d %s=%.6f mean=%.6f max=%.6f (%s, taus=%s, repeats=%d)",
                        outer,
                        selection_metric_key,
                        val_score,
                        val_score_mean,
                        val_score_max,
                        val_prefix,
                        em_val_tau_list,
                        em_val_repeats,
                    )
                    val_log_metrics[f"val/{val_prefix}/{selection_metric_key}"] = val_score
                    val_log_metrics[f"val/{val_prefix}/mean_{val_metric_key}_tau_list"] = val_score_mean
                    val_log_metrics[f"val/{val_prefix}/max_{val_metric_key}_tau_list"] = val_score_max
                mlf.log_metrics(val_log_metrics, step=outer)
                if em_save_every_eval_checkpoint:
                    save_em_checkpoint(
                        out_dir / f"ct_iql_em_outer{outer:04d}.pt",
                        ct_model=ct_model,
                        planner=planner,
                        outer_iter=outer,
                        config_yaml=OmegaConf.to_yaml(args, resolve=True),
                        extra={
                            "val_score": val_score,
                            "val_metric": selection_metric_key,
                            "val_metric_base": val_metric_key,
                            "val_tau_list": em_val_tau_list,
                            "val_tau_agg": val_tau_agg,
                            "val_repeats": em_val_repeats,
                            "val_seed_base": em_val_seed_base,
                            "val_scores": val_scores,
                        },
                    )
                if val_score < best_val:
                    best_val = val_score
                    best_outer = outer
                    save_em_checkpoint(
                        ckpt_path,
                        ct_model=ct_model,
                        planner=planner,
                        outer_iter=outer,
                        config_yaml=OmegaConf.to_yaml(args, resolve=True),
                        extra={
                            "val_score": val_score,
                            "val_metric": selection_metric_key,
                            "val_metric_base": val_metric_key,
                            "val_tau_list": em_val_tau_list,
                            "val_tau_agg": val_tau_agg,
                            "val_repeats": em_val_repeats,
                            "val_seed_base": em_val_seed_base,
                            "val_scores": val_scores,
                        },
                    )
                    logger.info("Saved best EM checkpoint to %s", ckpt_path)

        if best_outer == 0 and not ckpt_path.is_file():
            save_em_checkpoint(
                ckpt_path,
                ct_model=ct_model,
                planner=planner,
                outer_iter=em_outer_iters,
                config_yaml=OmegaConf.to_yaml(args, resolve=True),
            )
    finally:
        logger.info(
            "EM training done. best_outer=%d best_%s=%.6f ckpt=%s",
            best_outer,
            selection_metric_key,
            best_val,
            ckpt_path,
        )
        mlf.finish(
            artifact_paths=[ckpt_path] if ckpt_path.is_file() else None,
            final_metrics={
                "best/outer_iter": float(best_outer),
                f"best/val_{selection_metric_key}": best_val,
            },
            final_step=best_outer if best_outer > 0 else em_outer_iters,
        )


if __name__ == "__main__":
    main()
