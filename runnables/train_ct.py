"""
Standalone training for Causal Transformer + WeightNet + Predictor (ctd.md).

Joint loss: weighted MSE on Y_{t+1} + alpha * alignment (MMD or Sinkhorn on shuffled A_t).
Early stopping on val loss_pred; checkpoint saves only ``ct_history_encoder`` + ``projection_head``
for downstream ``InferenceModel`` / IQL (``ct_best_encoder.pt``).
"""
import logging
import os
import sys
from pathlib import Path

import hydra
import torch
import torch.nn.functional as F
from hydra.utils import get_original_cwd, instantiate
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.ct_transition_dataset import CTTransitionDataset, collate_ct_batch, _covariate_stream_dim
from src.models.ct_deconfound import CTDeconfoundModel
from src.utils.utils import (
    compute_mmd_weighted,
    compute_weighted_wasserstein_joint_marginal_flat,
    repeat_static,
    set_seed,
    to_float,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

OmegaConf.register_new_resolver("toint", lambda x: int(x), replace=True)


def _alignment_loss(
    Z_t: torch.Tensor,
    A_t: torch.Tensor,
    w: torch.Tensor,
    mode: str,
    sinkhorn_blur: float,
) -> torch.Tensor:
    B = Z_t.size(0)
    perm = torch.randperm(B, device=Z_t.device)
    joint_rep = torch.cat([Z_t, A_t], dim=-1)
    marginal_rep = torch.cat([Z_t, A_t[perm]], dim=-1)
    if mode == "mmd":
        return compute_mmd_weighted(joint_rep, marginal_rep, w)
    if mode == "sinkhorn":
        return compute_weighted_wasserstein_joint_marginal_flat(
            joint_rep, marginal_rep, w, blur=sinkhorn_blur
        )
    raise ValueError(f"Unknown ct_align_loss: {mode}")


def _ct_sched_sampling_p(epoch: int, p_start: float, p_end: float, ramp_epochs: int) -> float:
    if ramp_epochs <= 0:
        return float(p_end)
    t = min(1.0, float(epoch) / float(ramp_epochs))
    return float(p_start) + t * (float(p_end) - float(p_start))


def _ct_horizon_weights(eta: float, k_max: int) -> list:
    raw = [float(eta) ** k for k in range(k_max)]
    s = sum(raw)
    return [x / s for x in raw]


def _scheduled_prev_outputs(
    H_work: dict,
    y_hat_prev: torch.Tensor,
    valid_len_prev: torch.Tensor,
    p: float,
) -> tuple:
    """
    With prob ``p`` per batch row, replace ``prev_outputs[b, idx, :]`` with ``y_hat_prev[b].detach()``,
    where ``idx = valid_len_prev[b] - 1`` (last index of the shorter prefix that produced ``y_hat_prev``).
    """
    # Hm = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in H_work.items()}
    Hm = {}
    for k, v in H_work.items():
        # 克隆张量以避免破坏原始 dataloader 里的数据
        Hm[k] = v.clone() if torch.is_tensor(v) else v
    po = Hm["prev_outputs"]
    B = y_hat_prev.size(0)
    device = y_hat_prev.device
    n_used = 0
    if p <= 0.0:
        return Hm, n_used
    # yd = y_hat_prev.detach()
    yd = y_hat_prev
    for b in range(B):
        if float(torch.rand(1, device=device)) >= p:
            continue
        idx = int(valid_len_prev[b].item()) - 1
        if 0 <= idx < po.size(1):
            po[b, idx, :] = yd[b]
            n_used += 1
    return Hm, n_used


def _run_epoch(
    model,
    loader,
    optimizer_w,
    optimizer_theta,
    device,
    align_mode: str,
    blur: float,
    train: bool,
    *,
    multi_k_max: int = 1,
    horiz_w: list = None,
    ss_p: float = 0.0,
    k_inner: int = 1,
    w_clip: float = 1.0,
    anchor_weight: float = 0.0,
    train_std_cv: float = 1.0,
    dyn_consistency_weight: float = 0.0,
):
    """
    E-step: ``k_inner`` inner updates of WeightNet against ``Z_t.detach()``, minimizing
    ``loss_align(Z_t, A_t, w)``. With k_inner=1 the behaviour matches the legacy 1:1
    E/M interleaving exactly; k_inner>1 is the Plan-A analogue of CTD-NKO's per-epoch
    refit, giving the E-step more iterations to converge before the M-step moves
    Z_t again. The M-step reuses the PRE-loop ``w_fix`` to stay backward-compatible
    with the multi-horizon losses (only the WeightNet parameters advance faster).

    ``anchor_weight``: blend the WeightNet-reweighted MSE (legacy) with an unweighted
    anchor MSE on the M-step: total = (1-a)*weighted + a*anchor. See ct_deconfound
    docstring and vcip.yaml:ct_anchor_weight for motivation. 0.0 = legacy.
    ``train_std_cv``: cancer_volume std from train_scaling_params; used only for
    human-readable logging of un-weighted MAE in unscaled units.
    """
    if horiz_w is None:
        horiz_w = [1.0]
    k_inner_eff = max(1, int(k_inner))
    a_blend = max(0.0, min(1.0, float(anchor_weight)))
    total_pred = 0.0
    total_align = 0.0       # post-loop alignment (what the updated WeightNet achieves)
    total_align_pre = 0.0   # pre-loop alignment (before any E-step update this batch)
    total_dyn = 0.0         # latent dynamics consistency loss (z_next_pred -> z_{t+1})
    sum_l1 = 0.0            # k=1 weighted MSE (legacy)
    sum_l1_anchor = 0.0     # k=1 unweighted MSE (anchor)
    # Un-weighted MAE on normalized y: populated only at k=1 using y_hat1, y_next.
    # This is the primary diagnostic for Predictor offset bias (see debug_predictor_sensitivity.py).
    sum_mae_uw_norm = 0.0
    sum_mae_uw_denom = 0.0
    sum_l2 = 0.0
    sum_l3 = 0.0
    sum_ss = 0
    ss_denom = 0
    n_batches = 0
    # WeightNet health trackers (computed on active samples only, matching the loss mask).
    # Since w = softmax(logits) * B per batch => mean(w)=1 by construction; only var matters.
    sum_ess_frac = 0.0           # batch-averaged ESS fraction (1 = uniform, -> 0 = collapsed)
    sum_w_std = 0.0              # batch-averaged std of active weights
    sum_w_max = 0.0
    sum_w_min = 0.0
    w_samples: list = []         # concat of active weights across the whole epoch (for percentiles)
    if train:
        model.train()
    else:
        model.eval()
    for batch in loader:
        H_t = {k: v.to(device) for k, v in batch["H_t"].items()}
        y_next = batch["y_next"].to(device)
        H_t_next = None
        if "H_t_next" in batch:
            H_t_next = {k: v.to(device) for k, v in batch["H_t_next"].items()}
        elif "H_t_k2" in batch:
            H_t_next = {k: v.to(device) for k, v in batch["H_t_k2"].items()}
        Bsz = H_t["prev_treatments"].size(0)
        if multi_k_max > 1:
            ss_denom += Bsz * (multi_k_max - 1)
        act1 = H_t["active_entries"][:, :, 0] if H_t["active_entries"].dim() == 3 else H_t["active_entries"]
        valid_lens_h1 = act1.sum(dim=1).long().clamp(min=1)
        # Batch-level "is this sample active at the query step" mask, matching
        # CTDeconfoundModel.forward's active_t used in loss_pred. Shape (B,);
        # no .squeeze() so that B=1 edge case still yields a 1-D mask.
        active_t_mask = (H_t["active_entries"][:, -1, 0].detach() > 0.5)

        with torch.set_grad_enabled(train):
            loss_pred1_w, Z_t, A_t, w, _, y_hat1, loss_pred1_anchor = model(H_t, y_next)
            w_fix = w.detach()
            # Blend the weighted and anchor losses for the M-step (Plan A).
            loss_pred1 = (1.0 - a_blend) * loss_pred1_w + a_blend * loss_pred1_anchor
            if dyn_consistency_weight > 0.0 and H_t_next is not None:
                loss_dyn, _, _ = model.latent_dynamics_loss(H_t, H_t_next, detach_target=True)
            else:
                loss_dyn = torch.zeros((), device=device, dtype=loss_pred1.dtype)
            l1f = float(loss_pred1_w.detach())
            l1f_anchor = float(loss_pred1_anchor.detach())
            sum_l1 += l1f
            sum_l1_anchor += l1f_anchor

            # Un-weighted MAE on active samples (normalized y) — bias diagnostic.
            with torch.no_grad():
                mae_per = (y_hat1 - y_next).abs().mean(dim=-1)  # [B]
                act1_last = active_t_mask.float()               # [B], 1 where active
                sum_mae_uw_norm += float((mae_per * act1_last).sum())
                sum_mae_uw_denom += float(act1_last.sum())

            # --------------------- WeightNet health ---------------------
            # Compute stats on active weights only, since inactive samples are
            # masked out in loss_pred but still participate in softmax (which
            # normalizes over all B samples per batch).
            w_det = w.detach()
            w_act = w_det[active_t_mask]
            if w_act.numel() > 0:
                w_sum = float(w_act.sum())
                w_sqsum = float((w_act * w_act).sum())
                n_act = int(w_act.numel())
                # ESS_frac = (Σw)^2 / (Σw^2 * N), in (0, 1]. 1 = uniform, -> 0 = collapsed.
                ess_frac = (w_sum * w_sum) / (w_sqsum * n_act + 1e-12)
                sum_ess_frac += ess_frac
                sum_w_std += float(w_act.std(unbiased=False))
                sum_w_max += float(w_act.max())
                sum_w_min += float(w_act.min())
                # Keep a sample for full-epoch percentiles. Cap size to avoid
                # blowing memory if someone runs huge epochs (~200k values = ~1MB, fine).
                w_samples.append(w_act.cpu())

            # l2f 和 l3f 分别表示在多步(multi-horizon)情况下，第二步(k=2)和第三步(k=3)使用固定样本权重下的预测损失（weighted prediction loss）。
            # 具体地，它们这样计算：
            # - l2f: 针对 k=2（预测 y_{t+2}），用模型第一次E-step返回的权重 w_fix 计算 weighted loss；
            # - l3f: 针对 k=3（预测 y_{t+3}），同样用相同（k=1时求得）的权重 w_fix 计算 weighted loss。
            if multi_k_max <= 1:
                loss_theta = loss_pred1 + float(dyn_consistency_weight) * loss_dyn
                l2f = l3f = 0.0
                ss_batch = 0
            else:
                l2f = l3f = 0.0
                ss_batch = 0
                # k=2
                H2 = {k: v.to(device) for k, v in batch["H_t_k2"].items()}
                y2 = batch["y_next2"].to(device)
                H2m, n2 = _scheduled_prev_outputs(H2, y_hat1, valid_lens_h1, ss_p if train else 0.0)
                ss_batch += n2
                loss_pred2_w, y_hat2, loss_pred2_anchor = model.weighted_prediction_loss(H2m, y2, w_fix)
                loss_pred2 = (1.0 - a_blend) * loss_pred2_w + a_blend * loss_pred2_anchor
                l2f = float(loss_pred2_w.detach())

                loss_theta = horiz_w[0] * loss_pred1 + horiz_w[1] * loss_pred2
                if multi_k_max >= 3:
                    # k=3
                    H3 = {k: v.to(device) for k, v in batch["H_t_k3"].items()}
                    y3 = batch["y_next3"].to(device)
                    act2 = H2["active_entries"][:, :, 0] if H2["active_entries"].dim() == 3 else H2["active_entries"]
                    valid_lens_h2 = act2.sum(dim=1).long().clamp(min=1)
                    H3m, n3 = _scheduled_prev_outputs(H3, y_hat2, valid_lens_h2, ss_p if train else 0.0)
                    ss_batch += n3
                    loss_pred3_w, _, loss_pred3_anchor = model.weighted_prediction_loss(H3m, y3, w_fix)
                    loss_pred3 = (1.0 - a_blend) * loss_pred3_w + a_blend * loss_pred3_anchor
                    l3f = float(loss_pred3_w.detach())
                    loss_theta = loss_theta + horiz_w[2] * loss_pred3
                loss_theta = loss_theta + float(dyn_consistency_weight) * loss_dyn
            # l2f, l3f 的值在下面用于均值统计与日志

            if train:
                # --- E-Step (inner loop of k_inner iterations) ---
                # Freeze the encoder output Z_t for the whole inner loop. Each iteration:
                #   1. recompute logits/w from the CURRENT WeightNet (it just stepped)
                #   2. recompute loss_align with the fresh w
                #   3. backward + step WeightNet only
                # loss_align_pre = iteration-0 alignment (before any E-step update),
                # loss_align (post-loop variable below) = iteration-(k_inner-1) alignment
                # (what the updated WeightNet achieves). Gap = align_pre - align_post
                # measures per-batch E-step progress.
                Z_t_det = Z_t.detach()
                A_t_det = A_t  # A_t is input data, has no grad path anyway
                loss_align_pre = None
                for i in range(k_inner_eff):
                    optimizer_w.zero_grad(set_to_none=True)
                    za_w_i = torch.cat([Z_t_det, A_t_det], dim=-1)
                    logits_i = model.weight_net(za_w_i)
                    w_i = F.softmax(logits_i, dim=0) * float(Z_t_det.size(0))
                    loss_align = _alignment_loss(Z_t_det, A_t_det, w_i, align_mode, blur)
                    if i == 0:
                        loss_align_pre = loss_align.detach()
                    loss_align.backward()
                    # Clip WeightNet grads: with k_inner>1 the E-step receives more
                    # gradient steps per epoch, so prevent occasional blow-ups on
                    # Sinkhorn/MMD loss surfaces. Pass w_clip=None to disable entirely.
                    if w_clip is not None and w_clip > 0:
                        torch.nn.utils.clip_grad_norm_(
                            model.weight_net.parameters(), max_norm=float(w_clip)
                        )
                    optimizer_w.step()

                # --- M-Step: 更新ct_encoder和predictor parameters，仅loss_theta作用 ---
                # Note: loss_theta was built using w_fix (pre-inner-loop w). This matches
                # the legacy M-step semantics exactly when k_inner=1, and for k_inner>1 it
                # keeps the multi-horizon losses consistent with the same w_fix used for
                # k=1. If you want M-step to use the POST-loop w instead, that is a
                # separate design choice (requires a second forward after E-step).
                optimizer_theta.zero_grad(set_to_none=True)
                loss_theta.backward()
                torch.nn.utils.clip_grad_norm_(model.ct_encoder.parameters(), max_norm=1.0)
                torch.nn.utils.clip_grad_norm_(model.predictor.parameters(), max_norm=1.0)
                torch.nn.utils.clip_grad_norm_(model.z_dynamics.parameters(), max_norm=1.0)
                optimizer_theta.step()
            else:
                loss_align = _alignment_loss(Z_t, A_t, w, align_mode, blur)
                loss_align_pre = loss_align.detach()

        total_pred += float(loss_theta.detach())
        total_align += float(loss_align.detach())
        total_dyn += float(loss_dyn.detach())
        if loss_align_pre is not None:
            total_align_pre += float(loss_align_pre)
        else:
            # Should not happen, but keep running-sum consistent for safety.
            total_align_pre += float(loss_align.detach())
        sum_l2 += l2f
        sum_l3 += l3f
        sum_ss += ss_batch
        n_batches += 1

    nb = max(n_batches, 1)
    mae_uw_norm = sum_mae_uw_norm / max(sum_mae_uw_denom, 1e-8)
    extras = {
        "mean_l1": sum_l1 / nb,
        "mean_l1_anchor": sum_l1_anchor / nb,
        # Un-weighted MAE in normalized / unscaled cancer_volume units.
        # Unweighted MAE is the direct diagnostic for predictor offset bias:
        # CT's legacy val_L1 is w-reweighted MSE that MASKS bias, while this
        # un-weighted number reflects true population-level predictor quality.
        "mae_uw_norm": float(mae_uw_norm),
        "mae_uw_uns": float(mae_uw_norm * float(train_std_cv)),
        "mean_l2": sum_l2 / nb,
        "mean_l3": sum_l3 / nb,
        "ss_frac": float(sum_ss) / float(max(1, ss_denom)) if multi_k_max > 1 else 0.0,
        # Pre-inner-loop alignment (what WeightNet would produce BEFORE this batch's
        # E-step updates). Compared against ``total_align/nb`` (post-inner-loop)
        # to quantify how much each batch's k_inner sub-iterations actually reduce
        # alignment. For k_inner=1 these two are equal by construction.
        "align_pre":  total_align_pre / nb,
        "mean_dyn": total_dyn / nb,
        # WeightNet health: batch-averaged scalars + full-epoch percentiles of
        # active weights. Use these to detect whether deconfounding is actually
        # happening (see train_ct.py epoch log and docstring below).
        "w_ess_frac": sum_ess_frac / nb,
        "w_std":     sum_w_std   / nb,
        "w_max":     sum_w_max   / nb,
        "w_min":     sum_w_min   / nb,
    }
    if w_samples:
        W = torch.cat(w_samples)
        # Compute percentiles once per epoch. Cheap: even 100k floats is <1ms.
        qs = torch.tensor([0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
        pct = torch.quantile(W, qs).tolist()
        extras["w_p01"] = pct[0]
        extras["w_p05"] = pct[1]
        extras["w_p25"] = pct[2]
        extras["w_p50"] = pct[3]
        extras["w_p75"] = pct[4]
        extras["w_p95"] = pct[5]
        extras["w_p99"] = pct[6]
    return total_pred / nb, total_align / nb, extras


@hydra.main(version_base=None, config_name="config.yaml", config_path="../configs/")
def main(args: DictConfig):
    OmegaConf.set_struct(args, False)
    logger.info("\n" + OmegaConf.to_yaml(args, resolve=True))

    set_seed(int(args.exp.seed))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ct_align = str(OmegaConf.select(args, "exp.ct_align_loss", default="sinkhorn"))
    # ct_alpha = float(OmegaConf.select(args, "exp.ct_alpha", default=0.1))
    ct_lr = float(OmegaConf.select(args, "exp.ct_lr", default=5e-4))
    ct_epochs = int(OmegaConf.select(args, "exp.ct_epochs", default=200))
    ct_patience = int(OmegaConf.select(args, "exp.ct_patience", default=20))
    ct_wd = float(OmegaConf.select(args, "exp.ct_weight_decay", default=1e-5))
    ct_blur = float(OmegaConf.select(args, "exp.ct_sinkhorn_blur", default=0.01))
    batch_size = int(OmegaConf.select(args, "exp.ct_batch_size", default=512))
    batch_size_val = int(OmegaConf.select(args, "exp.ct_batch_size_val", default=256))
    num_workers = int(OmegaConf.select(args, "exp.ct_num_workers", default=0))
    # How often (in epochs) to log the full weight percentile distribution.
    # The compact ESS/std/max/min line is logged every epoch regardless.
    w_log_every = int(OmegaConf.select(args, "exp.ct_weight_log_every", default=10))
    # E-step inner loop: number of WeightNet updates per batch before the M-step.
    # 1 = legacy 1:1 E/M interleaving; >1 = Plan A (stronger E-step convergence).
    k_inner = max(1, int(OmegaConf.select(args, "exp.ct_k_inner", default=1)))
    # Separate LR for WeightNet (if null, fall back to ct_lr for backward compat).
    _ct_w_lr_cfg = OmegaConf.select(args, "exp.ct_w_lr", default=None)
    w_lr = float(_ct_w_lr_cfg) if _ct_w_lr_cfg is not None else ct_lr
    # Gradient-clip norm for WeightNet. None / <=0 disables clipping.
    _ct_w_clip_cfg = OmegaConf.select(args, "exp.ct_w_clip", default=1.0)
    w_clip = float(_ct_w_clip_cfg) if _ct_w_clip_cfg is not None else None
    # M-step anchor weight (Plan A). Blend: loss = (1-a)*weighted + a*anchor.
    anchor_weight = float(OmegaConf.select(args, "exp.ct_anchor_weight", default=0.0))
    anchor_weight = max(0.0, min(1.0, anchor_weight))
    # Latent dynamics consistency (CTD-NKO-inspired): z_next_pred=g(z_t,a_t) should
    # match the encoder's next-step latent z_{t+1}. 0.0 disables the auxiliary loss.
    dyn_consistency_weight = float(OmegaConf.select(args, "exp.ct_dyn_consistency_weight", default=0.0))
    dyn_consistency_weight = max(0.0, dyn_consistency_weight)

    original_cwd = Path(get_original_cwd())
    args["exp"]["processed_data_dir"] = os.path.join(str(original_cwd), args["exp"]["processed_data_dir"])

    dataset_collection = instantiate(args.dataset, _recursive_=True)
    dataset_collection.process_data_multi()
    dataset_collection = to_float(dataset_collection)
    if int(args.dataset.static_size) > 0:
        dims = len(dataset_collection.train_f.data["static_features"].shape)
        if dims == 2:
            dataset_collection = repeat_static(dataset_collection)

    multi_k_max = int(OmegaConf.select(args, "exp.ct_multi_k_max", default=1))
    ct_multi_eta = float(OmegaConf.select(args, "exp.ct_multi_eta", default=0.5))
    ct_sched_p_start = float(OmegaConf.select(args, "exp.ct_sched_p_start", default=0.0))
    ct_sched_p_end = float(OmegaConf.select(args, "exp.ct_sched_p_end", default=0.3))
    ct_sched_ramp_epochs = int(OmegaConf.select(args, "exp.ct_sched_ramp_epochs", default=100))
    horiz_w = _ct_horizon_weights(ct_multi_eta, max(1, multi_k_max))

    # Early-stop / checkpoint-selection metric.
    #   "weighted" : multi-horizon WeightNet-reweighted MSE (legacy default).
    #   "l1"       : k=1 WeightNet-reweighted MSE (legacy).
    #   "mae_uw"   : k=1 UN-weighted MAE on normalized y. Recommended default
    #                post predictor-bias fix (ct_anchor_weight > 0): this is the
    #                unbiased factual-prediction metric and matches how downstream
    #                IQL uses the outcome_predictor during rollouts. Standard in
    #                CFRNet / Causal Transformer / CTD-NKO model selection.
    ct_es_metric = str(OmegaConf.select(args, "exp.ct_es_metric", default="weighted")).strip().lower()
    if ct_es_metric not in ("weighted", "l1", "mae_uw"):
        logger.warning(f"Unknown exp.ct_es_metric={ct_es_metric!r}; using 'weighted'.")
        ct_es_metric = "weighted"

    need_next_prefix = dyn_consistency_weight > 0.0
    train_ds = CTTransitionDataset(
        dataset_collection.train_f.data,
        multi_k_max=multi_k_max,
        include_next_prefix=need_next_prefix,
    )
    val_ds = CTTransitionDataset(
        dataset_collection.val_f.data,
        multi_k_max=multi_k_max,
        include_next_prefix=need_next_prefix,
    )
    # cancer_volume std used only to rescale MAE for human-readable logs.
    try:
        _, std_ser = dataset_collection.train_scaling_params
        train_std_cv = float(std_ser["cancer_volume"])
    except Exception:
        train_std_cv = 1.0
    logger.info(
        f"CT transitions: train={len(train_ds)}, val={len(val_ds)} | multi_k_max={multi_k_max} "
        f"horiz_w={horiz_w} sched_p=[{ct_sched_p_start}->{ct_sched_p_end} over {ct_sched_ramp_epochs} ep] "
        f"early_stop={ct_es_metric} k_inner={k_inner} anchor_w={anchor_weight:g} "
        f"dyn_w={dyn_consistency_weight:g} next_prefix={need_next_prefix} "
        f"train_std_cv={train_std_cv:.4f}"
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_ct_batch,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size_val,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_ct_batch,
    )

    ds_cfg = OmegaConf.to_container(args.dataset, resolve=True)
    x_dim = _covariate_stream_dim(ds_cfg)
    logger.info(f"Covariate stream x_dim (CT encoder input): {x_dim}")

    model = CTDeconfoundModel(args, x_dim=x_dim).to(device)

    # optimizer = torch.optim.Adam(model.parameters(), lr=ct_lr, weight_decay=ct_wd)
    # 分离网络参数
    w_params = list(model.weight_net.parameters())
    theta_params = list(model.ct_encoder.parameters()) + \
                   list(model.projection.parameters()) + \
                   list(model.predictor.parameters()) + \
                   list(model.z_dynamics.parameters())

    # 创建两个独立的优化器
    # WeightNet uses w_lr (defaults to ct_lr), encoder+predictor uses a smaller fixed
    # 1e-4 so the M-step moves slower than the E-step. Decoupling w_lr makes the
    # [E-inner] per-batch drop tunable without touching encoder dynamics.
    optimizer_w = torch.optim.Adam(w_params, lr=w_lr, weight_decay=ct_wd)
    optimizer_theta = torch.optim.Adam(theta_params, lr=ct_lr, weight_decay=ct_wd)  #TODO 调整 learning rate
    logger.info(
        f"Optimizers: theta_lr={ct_lr:g}, w_lr={w_lr:g} | "
        f"k_inner={k_inner} w_clip={w_clip if w_clip is not None else 'off'} "
        f"anchor_weight={anchor_weight:g} dyn_w={dyn_consistency_weight:g} "
        f"(M-step loss = {1.0 - anchor_weight:g}*weighted + {anchor_weight:g}*anchor + "
        f"{dyn_consistency_weight:g}*loss_z)"
    )

    # Output directory: default to canonical ct_checkpoints/seed_${s}_gamma_${g}/,
    # but allow override via exp.ct_ckpt_dir so grid search / parallel experiments
    # don't clobber the canonical file used by the main pipeline (train_ct_iql.sh).
    _ct_ckpt_dir_override = OmegaConf.select(args, "exp.ct_ckpt_dir", default=None)
    if _ct_ckpt_dir_override:
        out_dir = Path(str(_ct_ckpt_dir_override))
        if not out_dir.is_absolute():
            out_dir = original_cwd / out_dir
    else:
        out_dir = original_cwd / "ct_checkpoints" / f"seed_{int(args.exp.seed)}_gamma_{int(args.dataset.coeff)}"
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "ct_best_encoder.pt"
    logger.info(f"CT encoder checkpoint will be saved to: {ckpt_path}")

    def _es_score(va_extras: dict, va_pred_f: float) -> float:
        """
        Scalar for early stopping / best-ckpt selection. Lower = better.
            - "weighted" : multi-horizon WeightNet-reweighted val MSE (legacy).
            - "l1"       : k=1 WeightNet-reweighted val MSE (legacy, BIASED when
                           WeightNet concentrates mass; see ct_anchor_weight docs).
            - "mae_uw"   : k=1 UN-weighted val MAE on normalized y (recommended).
        """
        if ct_es_metric == "l1":
            return float(va_extras["mean_l1"])
        if ct_es_metric == "mae_uw":
            return float(va_extras["mae_uw_norm"])
        return float(va_pred_f)

    best_es = float("inf")
    patience_left = ct_patience

    for epoch in range(1, ct_epochs + 1):
        ss_p = _ct_sched_sampling_p(epoch, ct_sched_p_start, ct_sched_p_end, ct_sched_ramp_epochs)
        tr_pred, tr_align, tr_ex = _run_epoch(
            model,
            train_loader,
            optimizer_w,
            optimizer_theta,
            device,
            ct_align,
            ct_blur,
            True,
            multi_k_max=multi_k_max,
            horiz_w=horiz_w,
            ss_p=ss_p,
            k_inner=k_inner,
            w_clip=w_clip,
            anchor_weight=anchor_weight,
            train_std_cv=train_std_cv,
            dyn_consistency_weight=dyn_consistency_weight,
        )
        with torch.no_grad():
            # Val never steps the optimizer, so k_inner is irrelevant here; the inner
            # loop body is gated by the ``train`` flag and effectively skipped in eval.
            va_pred, va_align, va_ex = _run_epoch(
                model,
                val_loader,
                None,
                None,
                device,
                ct_align,
                ct_blur,
                False,
                multi_k_max=multi_k_max,
                horiz_w=horiz_w,
                ss_p=0.0,
                k_inner=1,
                anchor_weight=anchor_weight,
                train_std_cv=train_std_cv,
                dyn_consistency_weight=dyn_consistency_weight,
            )

        logger.info(
            f"Epoch {epoch}/{ct_epochs} ss_p={ss_p:.4f} "
            f"train_pred={tr_pred:.6f} train_align={tr_align:.6f} train_ss_frac={tr_ex['ss_frac']:.4f} "
            f"train_L1={tr_ex['mean_l1']:.6f} train_L2={tr_ex['mean_l2']:.6f} train_L3={tr_ex['mean_l3']:.6f} | "
            f"val_pred={va_pred:.6f} val_align={va_align:.6f} "
            f"val_L1={va_ex['mean_l1']:.6f} val_L2={va_ex['mean_l2']:.6f} val_L3={va_ex['mean_l3']:.6f}"
        )
        # Predictor-bias diagnostic: weighted vs un-weighted + MAE in unscaled units.
        # Healthy signs: (a) val_L1_anchor roughly equal to val_L1 (no WeightNet
        # concentration bias); (b) val_MAE_uw_uns < 0.3 on cancer_sim γ=4.
        # Symptoms of the pre-anchor pathology: val_L1 ~1e-5 but val_MAE_uw_uns ~1.7.
        logger.info(
            f"  [Pred-diag] train: L1w={tr_ex['mean_l1']:.3e} L1anc={tr_ex['mean_l1_anchor']:.3e} "
            f"MAE_uw_norm={tr_ex['mae_uw_norm']:.5f} MAE_uw_uns={tr_ex['mae_uw_uns']:.4f} "
            f"Zdyn={tr_ex.get('mean_dyn', float('nan')):.3e} | "
            f"val: L1w={va_ex['mean_l1']:.3e} L1anc={va_ex['mean_l1_anchor']:.3e} "
            f"MAE_uw_norm={va_ex['mae_uw_norm']:.5f} MAE_uw_uns={va_ex['mae_uw_uns']:.4f} "
            f"Zdyn={va_ex.get('mean_dyn', float('nan')):.3e}"
        )
        # When k_inner>1, report the per-batch-averaged pre/post alignment. Interpretation:
        #   * drop = align_pre - align_post > 0 and large   => inner loop is usefully
        #     reducing alignment within each batch (E-step is converging).
        #   * drop ~ 0                                      => k_inner is not helping
        #     (either already converged at i=0, or LR too small, or encoder drifts per-step).
        #   * drop < 0 (rare)                               => WeightNet overshoots;
        #     reduce k_inner or clip harder.
        if k_inner > 1:
            tr_pre = tr_ex.get("align_pre", float("nan"))
            drop = tr_pre - tr_align
            logger.info(
                f"  [E-inner ] k_inner={k_inner} train: align_pre={tr_pre:.6f} align_post={tr_align:.6f} "
                f"drop={drop:+.6f}"
            )
        # WeightNet health: compact line every epoch. Key diagnostic:
        #   ESS_frac ~ 1.0 + w_std ~ 0.0   => WeightNet is uniform (no deconfounding).
        #   ESS_frac ~ 0.3 + w_std large   => WeightNet collapsed onto few samples.
        #   ESS_frac in [0.5, 0.95] with moderate std => healthy reweighting.
        logger.info(
            f"  [W-health] train: ESS={tr_ex.get('w_ess_frac', float('nan')):.3f} "
            f"std={tr_ex.get('w_std', float('nan')):.3f} "
            f"min={tr_ex.get('w_min', float('nan')):.3f} "
            f"max={tr_ex.get('w_max', float('nan')):.3f} | "
            f"val: ESS={va_ex.get('w_ess_frac', float('nan')):.3f} "
            f"std={va_ex.get('w_std', float('nan')):.3f} "
            f"min={va_ex.get('w_min', float('nan')):.3f} "
            f"max={va_ex.get('w_max', float('nan')):.3f}"
        )
        # Full percentile dump every w_log_every epochs (and at epoch 1 + last).
        if w_log_every > 0 and (epoch == 1 or epoch % w_log_every == 0):
            if "w_p50" in tr_ex:
                logger.info(
                    f"  [W-pct  ] train: p01={tr_ex['w_p01']:.3f} p05={tr_ex['w_p05']:.3f} "
                    f"p25={tr_ex['w_p25']:.3f} p50={tr_ex['w_p50']:.3f} p75={tr_ex['w_p75']:.3f} "
                    f"p95={tr_ex['w_p95']:.3f} p99={tr_ex['w_p99']:.3f}"
                )
            if "w_p50" in va_ex:
                logger.info(
                    f"  [W-pct  ]   val: p01={va_ex['w_p01']:.3f} p05={va_ex['w_p05']:.3f} "
                    f"p25={va_ex['w_p25']:.3f} p50={va_ex['w_p50']:.3f} p75={va_ex['w_p75']:.3f} "
                    f"p95={va_ex['w_p95']:.3f} p99={va_ex['w_p99']:.3f}"
                )

        es_score = _es_score(va_ex, va_pred)
        if es_score < best_es:
            best_es = es_score
            patience_left = ct_patience
            torch.save(
                {
                    "ct_history_encoder": model.ct_encoder.state_dict(),
                    "projection_head": model.projection.state_dict(),
                    # Save the OutcomePredictor so downstream IQL can do model-based OPE
                    # (roll out with p(y_{t+1} | z_t, a_t) instead of the cancer simulator).
                    "outcome_predictor": model.predictor.state_dict(),
                    "z_dynamics": model.z_dynamics.state_dict(),
                    "predictor_hidden": int(OmegaConf.select(args, "exp.ct_predictor_hidden", default=64)),
                    "val_loss_pred": va_pred,
                    "val_loss_l1": float(va_ex["mean_l1"]),
                    "val_loss_l1_anchor": float(va_ex["mean_l1_anchor"]),
                    "val_mae_uw_norm": float(va_ex["mae_uw_norm"]),
                    "val_mae_uw_uns": float(va_ex["mae_uw_uns"]),
                    "val_loss_z_dyn": float(va_ex.get("mean_dyn", 0.0)),
                    "val_loss_align": float(va_align),
                    "anchor_weight": float(anchor_weight),
                    "ct_dyn_consistency_weight": float(dyn_consistency_weight),
                    "ct_es_metric": ct_es_metric,
                    "val_es_score": es_score,
                    "epoch": epoch,
                    "x_dim": x_dim,
                    "config": OmegaConf.to_yaml(args, resolve=True),
                },
                ckpt_path,
            )
            logger.info(
                f"Saved encoder checkpoint to {ckpt_path} "
                f"(early_stop_metric={ct_es_metric}, score={es_score:.6f}, val_pred={va_pred:.6f}, "
                f"val_L1={va_ex['mean_l1']:.6f} val_MAE_uw_uns={va_ex['mae_uw_uns']:.4f})"
            )
        else:
            patience_left -= 1
            if patience_left <= 0:
                logger.info(
                    f"Early stopping (no improvement on early_stop_metric={ct_es_metric!r} for {ct_patience} epochs)."
                )
                break

    es_label = {
        "l1": "val_L1 (weighted, biased)",
        "mae_uw": "val_MAE_uw_norm (un-weighted, recommended)",
        "weighted": "val_pred (weighted)",
    }[ct_es_metric]
    logger.info(f"Done. Best {es_label}={best_es:.6f} (early_stop={ct_es_metric}). Encoder-only file: {ckpt_path}")


if __name__ == "__main__":
    main()
