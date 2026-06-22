"""
CT+IQL unified EM training (planning-oriented representation learning).

E-step: fix encoder, Q, V, π; update WeightNet only.
M-step: fix WeightNet; weighted V→Q→π per batch; only Q-step updates encoder.
"""
import logging
import os
import sys
import ast
from pathlib import Path
from typing import Dict

import hydra
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
from src.utils.em_config import (
    empty_replay_error as _empty_replay_error,
    selection_world_from_config as _selection_world_from_config,
    worlds_from_config as _worlds_from_config,
)
from src.utils.mlflow_vcip import VCIPMlflowTracker
from src.utils.utils import repeat_static, set_seed, to_float

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

OmegaConf.register_new_resolver("toint", lambda x: int(x), replace=True)

VAL_METRIC_KEYS = ("mae_uns", "mae_norm", "rmse_uns", "rmse_norm", "gift_rmse", "gift_rmse_percent")


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


@hydra.main(version_base=None, config_name="config.yaml", config_path="../configs/")
def main(args: DictConfig):
    OmegaConf.set_struct(args, False)
    logger.info("\n" + OmegaConf.to_yaml(args, resolve=True))

    set_seed(int(args.exp.seed))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    original_cwd = Path(get_original_cwd())
    args["exp"]["processed_data_dir"] = os.path.join(str(original_cwd), args["exp"]["processed_data_dir"])

    em_outer_iters = int(OmegaConf.select(args, "exp.em_outer_iters", default=50))
    em_m_steps = int(OmegaConf.select(args, "exp.em_m_steps_per_outer", default=200))
    em_encoder_lr = float(OmegaConf.select(args, "exp.em_encoder_lr", default=3e-5))
    em_val_every = int(OmegaConf.select(args, "exp.em_val_every", default=5))
    em_save_every_eval_checkpoint = bool(
        OmegaConf.select(args, "exp.em_save_every_eval_checkpoint", default=False)
    )
    em_save_every_outer_checkpoint = bool(
        OmegaConf.select(args, "exp.em_save_every_outer_checkpoint", default=False)
    )
    em_warmup = int(OmegaConf.select(args, "exp.em_warmup_outer_iters", default=2))
    em_log_m_every = int(OmegaConf.select(args, "exp.em_log_m_every", default=50))
    em_e_epochs = max(1, int(OmegaConf.select(args, "exp.em_e_epochs", default=5)))
    em_encoder_diagnostics = bool(
        OmegaConf.select(args, "exp.em_encoder_diagnostics", default=False)
    )
    em_encoder_diagnostics_every = max(
        1,
        int(
            OmegaConf.select(
                args,
                "exp.em_encoder_diagnostics_every",
                default=em_log_m_every,
            )
        ),
    )
    em_e_refresh_every = int(OmegaConf.select(args, "exp.em_e_refresh_every", default=1))
    em_her_refresh_every = int(OmegaConf.select(args, "exp.em_her_refresh_every", default=1))
    em_her_samples_per_transition = max(
        1,
        int(OmegaConf.select(args, "exp.em_her_samples_per_transition", default=1)),
    )
    iql_target_sampling = str(
        OmegaConf.select(args, "exp.iql_target_sampling", default="horizon_aligned")
    )
    iql_target_horizons = _list_from_config(
        OmegaConf.select(args, "exp.iql_target_horizons", default=None)
    )
    iql_horizon_terminal_done = bool(
        OmegaConf.select(args, "exp.iql_horizon_terminal_done", default=True)
    )

    ct_align = str(OmegaConf.select(args, "exp.ct_align_loss", default="sinkhorn"))
    if ct_align != "sinkhorn":
        raise ValueError(f"EM training requires ct_align_loss=sinkhorn, got {ct_align!r}")
    ct_blur = float(OmegaConf.select(args, "exp.ct_sinkhorn_blur", default=0.01))
    _em_e_w_lr = OmegaConf.select(args, "exp.em_e_w_lr", default=None)
    w_lr = (
        float(_em_e_w_lr)
        if _em_e_w_lr is not None
        else float(OmegaConf.select(args, "exp.ct_w_lr", default=0.1))
    )
    _ct_w_clip = OmegaConf.select(args, "exp.ct_w_clip", default=1.0)
    w_clip = float(_ct_w_clip) if _ct_w_clip is not None else None
    ct_wd = float(OmegaConf.select(args, "exp.ct_weight_decay", default=1e-5))
    ct_batch_size = int(OmegaConf.select(args, "exp.ct_batch_size", default=512))
    m_batch_size = int(OmegaConf.select(args, "exp.iql_batch_size", default=256))

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

    z_dim = int(args.model.z_dim)
    out_dim = int(args.dataset.output_size)
    act_dim = int(args.dataset.treatment_size)
    state_dim = z_dim + out_dim + 1 + act_dim
    goal_adapter_enabled = bool(
        OmegaConf.select(args, "exp.iql_goal_adapter_enabled", default=False)
    )
    goal_adapter_hidden_dim = int(
        OmegaConf.select(args, "exp.iql_goal_adapter_hidden_dim", default=64)
    )
    goal_adapter_init_scale = float(
        OmegaConf.select(args, "exp.iql_goal_adapter_init_scale", default=1e-3)
    )

    max_action = float(OmegaConf.select(args, "exp.iql_max_action", default=1.0))
    max_tau = float(OmegaConf.select(args, "exp.max_tau", default=12.0))
    iql_max_grad = OmegaConf.select(args, "exp.iql_max_grad_norm", default=None)
    iql_max_grad = None if iql_max_grad is None else float(iql_max_grad)
    enc_max_grad = OmegaConf.select(args, "exp.em_encoder_max_grad_norm", default=1.0)
    enc_max_grad = None if enc_max_grad is None else float(enc_max_grad)
    iql_weight_max = OmegaConf.select(args, "exp.iql_weight_max", default=10.0)
    iql_weight_max = None if iql_weight_max is None else float(iql_weight_max)

    planner_cfg = IQLPlannerConfig(
        state_dim=state_dim,
        action_dim=act_dim,
        max_action=max_action,
        hidden_dim=int(OmegaConf.select(args, "exp.iql_hidden_dim", default=256)),
        n_hidden=int(OmegaConf.select(args, "exp.iql_n_hidden", default=2)),
        iql_tau=float(OmegaConf.select(args, "exp.iql_tau", default=0.5)),
        beta=float(OmegaConf.select(args, "exp.iql_beta", default=3.0)),
        adv_max=float(OmegaConf.select(args, "exp.iql_adv_max", default=100.0)),
        weight_max=iql_weight_max,
        discount=float(OmegaConf.select(args, "exp.iql_discount", default=0.99)),
        tau=float(OmegaConf.select(args, "exp.iql_target_tau", default=0.005)),
        actor_lr=float(OmegaConf.select(args, "exp.iql_actor_lr", default=3e-4)),
        qf_lr=float(OmegaConf.select(args, "exp.iql_qf_lr", default=3e-4)),
        vf_lr=float(OmegaConf.select(args, "exp.iql_vf_lr", default=3e-4)),
        max_steps=em_outer_iters * em_m_steps,
        deterministic_actor=bool(OmegaConf.select(args, "exp.iql_deterministic", default=False)),
        actor_dropout=OmegaConf.select(args, "exp.iql_actor_dropout", default=None),
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
        max_patients = OmegaConf.select(args, "exp.iql_max_patients", default=None)
        raw = build_iql_raw_transitions(
            data=dataset_collection.train_f.data,
            reward_type=str(OmegaConf.select(args, "exp.iql_reward_type", default="progress")),
            max_patients=max_patients,
            max_action=max_action,
            dataset_actions_unit_interval=bool(
                OmegaConf.select(args, "exp.iql_dataset_actions_unit_interval", default=True)
            ),
            max_tau=max_tau,
            reward_clip=float(OmegaConf.select(args, "exp.iql_reward_clip", default=3.0)),
            reward_scale=str(OmegaConf.select(args, "exp.iql_reward_scale", default="auto")),
            reward_huber_delta=float(
                OmegaConf.select(args, "exp.iql_reward_huber_delta", default=1.0)
            ),
            samples_per_transition=em_her_samples_per_transition,
            target_sampling=iql_target_sampling,
            target_horizons=iql_target_horizons,
            horizon_terminal_done=iql_horizon_terminal_done,
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
        num_workers=int(OmegaConf.select(args, "exp.ct_num_workers", default=0)),
        collate_fn=collate_ct_estep_batch,
        drop_last=False,
    )

    em_cfg = EMTrainConfig(
        align_mode=ct_align,
        sinkhorn_blur=ct_blur,
        w_clip=w_clip,
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

    _ckpt_override = OmegaConf.select(args, "exp.em_ckpt_dir", default=None)
    if _ckpt_override:
        out_dir = Path(str(_ckpt_override))
        if not out_dir.is_absolute():
            out_dir = original_cwd / out_dir
    else:
        out_dir = (
            original_cwd
            / "em_checkpoints"
            / f"seed_{int(args.exp.seed)}_gamma_{int(args.dataset.coeff)}"
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "ct_iql_em_best.pt"

    val_metric_key = str(OmegaConf.select(args, "exp.em_val_metric", default="mae_uns")).strip().lower()
    if val_metric_key not in VAL_METRIC_KEYS:
        raise ValueError(
            f"exp.em_val_metric must be one of {VAL_METRIC_KEYS}, got {val_metric_key!r}"
        )
    val_worlds = _worlds_from_config(OmegaConf.select(args, "exp.em_val_worlds", default=["sim"]))
    sel_world = _selection_world_from_config(
        OmegaConf.select(args, "exp.em_val_selection_world", default=None),
        val_worlds,
    )
    if "predictor" in val_worlds:
        raise ValueError(
            "End-to-end EM checkpoints do not train/load outcome_predictor; remove 'predictor' "
            "from exp.em_val_worlds or add predictor training before using predictor-world validation."
        )
    eval_tau = int(args.exp.tau)
    em_val_tau_list = _list_from_config(
        OmegaConf.select(args, "exp.em_val_tau_list", default=None)
    )
    if em_val_tau_list is None:
        em_val_tau_list = [eval_tau]
    if len(em_val_tau_list) == 1:
        selection_metric_key = val_metric_key
    else:
        selection_metric_key = (
            f"mean_{val_metric_key}_tau"
            + "_".join(str(int(t)) for t in em_val_tau_list)
        )
    val_bs = int(OmegaConf.select(args, "exp.iql_val_batch_size", default=None) or args.exp.batch_size_val)
    autoreg = bool(OmegaConf.select(args, "exp.iql_eval_autoregressive", default=True))

    inference_model = InferenceModel(args).to(device)
    best_val = float("inf")
    best_outer = 0
    last_e_metrics: Dict[str, float] = {
        "align_pre": 0.0,
        "align_post": 0.0,
        "w_ess_frac": 1.0,
        "w_std": 0.0,
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

            if em_e_refresh_every <= 0 or (outer - 1) % em_e_refresh_every == 0:
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
                logger.info(
                    "E-step full fit outer=%d | n_active=%.0f n_total=%.0f epochs=%d w_lr=%.1e",
                    outer,
                    e_metrics.get("n_samples", 0),
                    e_metrics.get("n_samples_total", e_metrics.get("n_samples", 0)),
                    em_e_epochs,
                    w_lr,
                )
            else:
                e_metrics = last_e_metrics
            m_metrics = run_m_step_steps(
                ct_model, planner, optimizer_enc, replay, em_m_steps, em_cfg, outer_iter=outer
            )
            m_warmup = " (M-warmup)" if outer <= em_warmup else ""
            logger.info(
                "EM outer %d/%d | E(x%d): align_pre=%.4f align_post=%.4f w_ess=%.3f w_std=%.4f | "
                "M%s: q=%.4f v=%.4f pi=%.4f | replay=%d",
                outer,
                em_outer_iters,
                em_e_epochs,
                e_metrics["align_pre"],
                e_metrics["align_post"],
                e_metrics["w_ess_frac"],
                e_metrics.get("w_std", 0.0),
                m_warmup,
                m_metrics["q_loss"],
                m_metrics["value_loss"],
                m_metrics["actor_loss"],
                replay.size,
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
                "e/w_ess_frac": e_metrics["w_ess_frac"],
                "e/w_std": e_metrics.get("w_std", 0.0),
                "e/e_epochs": float(em_e_epochs),
                "e/n_samples": e_metrics.get("n_samples", 0.0),
                "e/n_samples_total": e_metrics.get("n_samples_total", e_metrics.get("n_samples", 0.0)),
                "m/q_loss": m_metrics["q_loss"],
                "m/value_loss": m_metrics["value_loss"],
                "m/actor_loss": m_metrics["actor_loss"],
                "replay/size": float(replay.size),
            }
            for key, value in m_metrics.items():
                if key.startswith("enc_"):
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
                        "e_w_ess_frac": e_metrics["w_ess_frac"],
                        "e_w_std": e_metrics.get("w_std", 0.0),
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
                        worlds=val_worlds,
                    )
                    per_world = metrics.get("per_world", {val_worlds[0]: metrics})
                    tau_score = float(per_world[sel_world][val_metric_key])
                    val_scores.append(tau_score)
                    val_log_metrics[f"val/{sel_world}/tau{int(tau_i)}/{val_metric_key}"] = tau_score
                val_score = float(sum(val_scores) / len(val_scores))
                if len(em_val_tau_list) == 1:
                    logger.info(
                        "EM val outer=%d %s=%.6f (%s, tau=%d)",
                        outer,
                        selection_metric_key,
                        val_score,
                        sel_world,
                        int(em_val_tau_list[0]),
                    )
                    val_log_metrics[f"val/{sel_world}/{val_metric_key}"] = val_score
                else:
                    logger.info(
                        "EM val outer=%d %s=%.6f (%s, taus=%s)",
                        outer,
                        selection_metric_key,
                        val_score,
                        sel_world,
                        em_val_tau_list,
                    )
                    val_log_metrics[f"val/{sel_world}/{selection_metric_key}"] = val_score
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
