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
):
    """
    E-step: alignment on k=1 (Z,A,w) only. M-step: multi-horizon weighted pred loss; optional SS on prev_outputs for k>=2.
    """
    if horiz_w is None:
        horiz_w = [1.0]
    total_pred = 0.0
    total_align = 0.0
    sum_l1 = 0.0
    sum_l2 = 0.0
    sum_l3 = 0.0
    sum_ss = 0
    ss_denom = 0
    n_batches = 0
    if train:
        model.train()
    else:
        model.eval()
    for batch in loader:
        H_t = {k: v.to(device) for k, v in batch["H_t"].items()}
        y_next = batch["y_next"].to(device)
        Bsz = H_t["prev_treatments"].size(0)
        if multi_k_max > 1:
            ss_denom += Bsz * (multi_k_max - 1)
        act1 = H_t["active_entries"][:, :, 0] if H_t["active_entries"].dim() == 3 else H_t["active_entries"]
        valid_lens_h1 = act1.sum(dim=1).long().clamp(min=1)

        with torch.set_grad_enabled(train):
            loss_pred1, Z_t, A_t, w, _, y_hat1 = model(H_t, y_next)
            w_fix = w.detach()
            l1f = float(loss_pred1.detach())
            sum_l1 += l1f

            # l2f 和 l3f 分别表示在多步(multi-horizon)情况下，第二步(k=2)和第三步(k=3)使用固定样本权重下的预测损失（weighted prediction loss）。
            # 具体地，它们这样计算：
            # - l2f: 针对 k=2（预测 y_{t+2}），用模型第一次E-step返回的权重 w_fix 计算 weighted loss；
            # - l3f: 针对 k=3（预测 y_{t+3}），同样用相同（k=1时求得）的权重 w_fix 计算 weighted loss。
            if multi_k_max <= 1:
                loss_theta = loss_pred1
                l2f = l3f = 0.0
                ss_batch = 0
            else:
                l2f = l3f = 0.0
                ss_batch = 0
                # 计算 k=2 时的输入和目标
                H2 = {k: v.to(device) for k, v in batch["H_t_k2"].items()}
                y2 = batch["y_next2"].to(device)
                # 用 scheduled sampling 更新第k=2步的 prev_outputs
                H2m, n2 = _scheduled_prev_outputs(H2, y_hat1, valid_lens_h1, ss_p if train else 0.0)
                ss_batch += n2
                # loss_pred2：使用模型weight_net在k=1步(E-step)算得的w_fix权重，计算k=2步（y_{t+2}）的weighted MSE
                loss_pred2, y_hat2 = model.weighted_prediction_loss(H2m, y2, w_fix)
                l2f = float(loss_pred2.detach())

                # 总损失：horiz_w允许对不同步次加权
                loss_theta = horiz_w[0] * loss_pred1 + horiz_w[1] * loss_pred2
                if multi_k_max >= 3:
                    # 计算 k=3 时的输入目标和 weighted loss
                    H3 = {k: v.to(device) for k, v in batch["H_t_k3"].items()}
                    y3 = batch["y_next3"].to(device)
                    act2 = H2["active_entries"][:, :, 0] if H2["active_entries"].dim() == 3 else H2["active_entries"]
                    valid_lens_h2 = act2.sum(dim=1).long().clamp(min=1)
                    H3m, n3 = _scheduled_prev_outputs(H3, y_hat2, valid_lens_h2, ss_p if train else 0.0)
                    ss_batch += n3
                    # loss_pred3 计算方式同上，只是用于k=3
                    loss_pred3, _ = model.weighted_prediction_loss(H3m, y3, w_fix)
                    l3f = float(loss_pred3.detach())
                    loss_theta = loss_theta + horiz_w[2] * loss_pred3
            # l2f, l3f 的值在下面用于均值统计与日志

            if train:
                # --- E-Step: 更新WeightNet parameters，仅loss_align作用 ---
                optimizer_w.zero_grad(set_to_none=True)
                loss_align = _alignment_loss(Z_t.detach(), A_t, w, align_mode, blur)  # Z_t.detach()
                loss_align.backward()
                optimizer_w.step()

                # --- M-Step: 更新ct_encoder和predictor parameters，仅loss_theta作用 ---
                optimizer_theta.zero_grad(set_to_none=True)
                loss_theta.backward()
                torch.nn.utils.clip_grad_norm_(model.ct_encoder.parameters(), max_norm=1.0)
                torch.nn.utils.clip_grad_norm_(model.predictor.parameters(), max_norm=1.0)
                optimizer_theta.step()
            else:
                loss_align = _alignment_loss(Z_t, A_t, w, align_mode, blur)

        total_pred += float(loss_theta.detach())
        total_align += float(loss_align.detach())
        sum_l2 += l2f
        sum_l3 += l3f
        sum_ss += ss_batch
        n_batches += 1

    nb = max(n_batches, 1)
    extras = {
        "mean_l1": sum_l1 / nb,
        "mean_l2": sum_l2 / nb,
        "mean_l3": sum_l3 / nb,
        "ss_frac": float(sum_ss) / float(max(1, ss_denom)) if multi_k_max > 1 else 0.0,
    }
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

    ct_es_metric = str(OmegaConf.select(args, "exp.ct_es_metric", default="weighted")).strip().lower()
    if ct_es_metric not in ("weighted", "l1"):
        logger.warning(f"Unknown exp.ct_es_metric={ct_es_metric!r}; using 'weighted'.")
        ct_es_metric = "weighted"

    train_ds = CTTransitionDataset(dataset_collection.train_f.data, multi_k_max=multi_k_max)
    val_ds = CTTransitionDataset(dataset_collection.val_f.data, multi_k_max=multi_k_max)
    logger.info(
        f"CT transitions: train={len(train_ds)}, val={len(val_ds)} | multi_k_max={multi_k_max} "
        f"horiz_w={horiz_w} sched_p=[{ct_sched_p_start}->{ct_sched_p_end} over {ct_sched_ramp_epochs} ep] "
        f"early_stop={ct_es_metric}"
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
                   list(model.predictor.parameters())

    # 创建两个独立的优化器
    optimizer_w = torch.optim.Adam(w_params, lr=ct_lr, weight_decay=ct_wd)
    optimizer_theta = torch.optim.Adam(theta_params, lr=1e-4, weight_decay=ct_wd)  #TODO 调整 learning rate

    out_dir = original_cwd / "ct_checkpoints" / f"seed_{int(args.exp.seed)}_gamma_{int(args.dataset.coeff)}"
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "ct_best_encoder.pt"

    def _es_score(va_extras: dict, va_pred_f: float) -> float:
        """Scalar for early stopping / best ckpt: weighted val_pred or val mean L1 (k=1 loss)."""
        if ct_es_metric == "l1":
            return float(va_extras["mean_l1"])
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
        )
        with torch.no_grad():
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
            )

        logger.info(
            f"Epoch {epoch}/{ct_epochs} ss_p={ss_p:.4f} "
            f"train_pred={tr_pred:.6f} train_align={tr_align:.6f} train_ss_frac={tr_ex['ss_frac']:.4f} "
            f"train_L1={tr_ex['mean_l1']:.6f} train_L2={tr_ex['mean_l2']:.6f} train_L3={tr_ex['mean_l3']:.6f} | "
            f"val_pred={va_pred:.6f} val_align={va_align:.6f} "
            f"val_L1={va_ex['mean_l1']:.6f} val_L2={va_ex['mean_l2']:.6f} val_L3={va_ex['mean_l3']:.6f}"
        )

        es_score = _es_score(va_ex, va_pred)
        if es_score < best_es:
            best_es = es_score
            patience_left = ct_patience
            torch.save(
                {
                    "ct_history_encoder": model.ct_encoder.state_dict(),
                    "projection_head": model.projection.state_dict(),
                    "val_loss_pred": va_pred,
                    "val_loss_l1": float(va_ex["mean_l1"]),
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
                f"(early_stop_metric={ct_es_metric}, score={es_score:.6f}, val_pred={va_pred:.6f}, val_L1={va_ex['mean_l1']:.6f})"
            )
        else:
            patience_left -= 1
            if patience_left <= 0:
                logger.info(
                    f"Early stopping (no improvement on early_stop_metric={ct_es_metric!r} for {ct_patience} epochs)."
                )
                break

    es_label = "val_L1" if ct_es_metric == "l1" else "val_pred(weighted)"
    logger.info(f"Done. Best {es_label}={best_es:.6f} (early_stop={ct_es_metric}). Encoder-only file: {ckpt_path}")


if __name__ == "__main__":
    main()
