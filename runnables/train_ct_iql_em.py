"""
CT+IQL unified EM training (planning-oriented representation learning).

E-step: fix encoder, Q, V, π; update WeightNet only.
M-step: fix WeightNet; weighted V→Q→π per batch; only Q-step updates encoder.
"""
import logging
import os
import sys
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
from src.utils.mlflow_vcip import VCIPMlflowTracker
from src.utils.utils import repeat_static, set_seed, to_float

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

OmegaConf.register_new_resolver("toint", lambda x: int(x), replace=True)


def _state_dict_to_cpu(obj):
    if torch.is_tensor(obj):
        return obj.detach().cpu().clone()
    if isinstance(obj, dict):
        return {k: _state_dict_to_cpu(v) for k, v in obj.items()}
    return obj


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
    em_warmup = int(OmegaConf.select(args, "exp.em_warmup_outer_iters", default=2))
    em_log_m_every = int(OmegaConf.select(args, "exp.em_log_m_every", default=50))
    em_e_epochs = max(1, int(OmegaConf.select(args, "exp.em_e_epochs", default=5)))
    em_e_refresh_every = int(OmegaConf.select(args, "exp.em_e_refresh_every", default=1))
    em_her_refresh_every = int(OmegaConf.select(args, "exp.em_her_refresh_every", default=1))
    em_her_samples_per_transition = max(
        1,
        int(OmegaConf.select(args, "exp.em_her_samples_per_transition", default=1)),
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

    z_dim = int(args.model.z_dim)
    out_dim = int(args.dataset.output_size)
    act_dim = int(args.dataset.treatment_size)
    state_dim = z_dim + out_dim + 1 + act_dim

    max_action = float(OmegaConf.select(args, "exp.iql_max_action", default=1.0))
    max_tau = float(OmegaConf.select(args, "exp.max_tau", default=12.0))
    iql_max_grad = OmegaConf.select(args, "exp.iql_max_grad_norm", default=None)
    iql_max_grad = None if iql_max_grad is None else float(iql_max_grad)
    enc_max_grad = OmegaConf.select(args, "exp.em_encoder_max_grad_norm", default=1.0)
    enc_max_grad = None if enc_max_grad is None else float(enc_max_grad)

    planner_cfg = IQLPlannerConfig(
        state_dim=state_dim,
        action_dim=act_dim,
        max_action=max_action,
        hidden_dim=int(OmegaConf.select(args, "exp.iql_hidden_dim", default=256)),
        n_hidden=int(OmegaConf.select(args, "exp.iql_n_hidden", default=2)),
        iql_tau=float(OmegaConf.select(args, "exp.iql_tau", default=0.5)),
        beta=float(OmegaConf.select(args, "exp.iql_beta", default=3.0)),
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
    )
    planner = IQLPlanner(planner_cfg)

    optimizer_w = torch.optim.Adam(ct_model.weight_net.parameters(), lr=w_lr, weight_decay=ct_wd)
    optimizer_enc = torch.optim.Adam(ct_model.encoder_parameters(), lr=em_encoder_lr, weight_decay=ct_wd)

    def _build_replay(her_seed: int) -> IQLRawReplayBuffer:
        raw = build_iql_raw_transitions(
            data=dataset_collection.train_f.data,
            reward_type=str(OmegaConf.select(args, "exp.iql_reward_type", default="progress")),
            max_patients=OmegaConf.select(args, "exp.iql_max_patients", default=None),
            max_action=max_action,
            dataset_actions_unit_interval=bool(
                OmegaConf.select(args, "exp.iql_dataset_actions_unit_interval", default=True)
            ),
            max_tau=max_tau,
            reward_clip=float(OmegaConf.select(args, "exp.iql_reward_clip", default=3.0)),
            reward_scale=str(OmegaConf.select(args, "exp.iql_reward_scale", default="auto")),
            samples_per_transition=em_her_samples_per_transition,
            seed=her_seed,
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
    val_worlds = tuple(
        str(w).strip()
        for w in OmegaConf.select(args, "exp.em_val_worlds", default=["sim"])
    )
    eval_tau = int(args.exp.tau)
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
                    "E-step full fit outer=%d | n=%.0f epochs=%d w_lr=%.1e",
                    outer,
                    e_metrics.get("n_samples", 0),
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
            mlf.log_metrics(
                {
                    "e/align_pre": e_metrics["align_pre"],
                    "e/align_post": e_metrics["align_post"],
                    "e/w_ess_frac": e_metrics["w_ess_frac"],
                    "e/w_std": e_metrics.get("w_std", 0.0),
                    "e/e_epochs": float(em_e_epochs),
                    "m/q_loss": m_metrics["q_loss"],
                    "m/value_loss": m_metrics["value_loss"],
                    "m/actor_loss": m_metrics["actor_loss"],
                    "replay/size": float(replay.size),
                },
                step=outer,
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
                metrics = aggregate_iql_planner_metrics(
                    planner,
                    inference_model,
                    dataset_collection,
                    dataset_collection.val_f,
                    args,
                    device=device,
                    tau=eval_tau,
                    max_tau=max_tau,
                    autoregressive_eval=autoreg,
                    val_batch_size=val_bs,
                    log_batches=False,
                    worlds=val_worlds,
                )
                per_world = metrics.get("per_world", {val_worlds[0]: metrics})
                sel_world = str(
                    OmegaConf.select(args, "exp.em_val_selection_world", default=val_worlds[0])
                )
                val_score = float(per_world[sel_world][val_metric_key])
                logger.info(
                    "EM val outer=%d %s=%.6f (%s)",
                    outer,
                    val_metric_key,
                    val_score,
                    sel_world,
                )
                mlf.log_metrics({f"val/{sel_world}/{val_metric_key}": val_score}, step=outer)
                if val_score < best_val:
                    best_val = val_score
                    best_outer = outer
                    save_em_checkpoint(
                        ckpt_path,
                        ct_model=ct_model,
                        planner=planner,
                        outer_iter=outer,
                        config_yaml=OmegaConf.to_yaml(args, resolve=True),
                        extra={"val_score": val_score, "val_metric": val_metric_key},
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
            val_metric_key,
            best_val,
            ckpt_path,
        )
        mlf.finish(
            artifact_paths=[ckpt_path] if ckpt_path.is_file() else None,
            final_metrics={
                "best/outer_iter": float(best_outer),
                f"best/val_{val_metric_key}": best_val,
            },
            final_step=best_outer if best_outer > 0 else em_outer_iters,
        )


if __name__ == "__main__":
    main()
