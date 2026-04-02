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


def _run_epoch(model, loader, optimizer, device, alpha: float, align_mode: str, blur: float, train: bool):
    total = 0.0
    total_pred = 0.0
    total_align = 0.0
    n_batches = 0
    if train:
        model.train()
    else:
        model.eval()
    for batch in loader:
        H_t = {k: v.to(device) for k, v in batch["H_t"].items()}
        y_next = batch["y_next"].to(device)
        if train:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(train):
            loss_pred, Z_t, A_t, w, _, _ = model(H_t, y_next)
            loss_align = _alignment_loss(Z_t, A_t, w, align_mode, blur)
            loss = loss_pred + alpha * loss_align
        if train:
            loss.backward()
            optimizer.step()
        total += float(loss.detach())
        total_pred += float(loss_pred.detach())
        total_align += float(loss_align.detach())
        n_batches += 1
    return total / max(n_batches, 1), total_pred / max(n_batches, 1), total_align / max(n_batches, 1)


@hydra.main(version_base=None, config_name="config.yaml", config_path="../configs/")
def main(args: DictConfig):
    OmegaConf.set_struct(args, False)
    logger.info("\n" + OmegaConf.to_yaml(args, resolve=True))

    set_seed(int(args.exp.seed))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ct_align = str(OmegaConf.select(args, "exp.ct_align_loss", default="sinkhorn"))
    ct_alpha = float(OmegaConf.select(args, "exp.ct_alpha", default=0.1))
    ct_lr = float(OmegaConf.select(args, "exp.ct_lr", default=1e-3))
    ct_epochs = int(OmegaConf.select(args, "exp.ct_epochs", default=200))
    ct_patience = int(OmegaConf.select(args, "exp.ct_patience", default=20))
    ct_wd = float(OmegaConf.select(args, "exp.ct_weight_decay", default=1e-5))
    ct_blur = float(OmegaConf.select(args, "exp.ct_sinkhorn_blur", default=0.01))
    batch_size = int(OmegaConf.select(args, "exp.ct_batch_size", default=256))
    batch_size_val = int(OmegaConf.select(args, "exp.ct_batch_size_val", default=128))
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

    train_ds = CTTransitionDataset(dataset_collection.train_f.data)
    val_ds = CTTransitionDataset(dataset_collection.val_f.data)
    logger.info(f"CT transitions: train={len(train_ds)}, val={len(val_ds)}")

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
    optimizer = torch.optim.Adam(model.parameters(), lr=ct_lr, weight_decay=ct_wd)

    out_dir = original_cwd / "ct_checkpoints" / f"seed_{int(args.exp.seed)}_gamma_{int(args.dataset.coeff)}"
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "ct_best_encoder.pt"

    best_val_pred = float("inf")
    patience_left = ct_patience

    for epoch in range(1, ct_epochs + 1):
        tr_loss, tr_pred, tr_align = _run_epoch(
            model, train_loader, optimizer, device, ct_alpha, ct_align, ct_blur, train=True
        )
        with torch.no_grad():
            va_loss, va_pred, va_align = _run_epoch(
                model, val_loader, None, device, ct_alpha, ct_align, ct_blur, train=False
            )

        logger.info(
            f"Epoch {epoch}/{ct_epochs} "
            f"train loss={tr_loss:.6f} (pred={tr_pred:.6f} align={tr_align:.6f}) | "
            f"val loss={va_loss:.6f} (pred={va_pred:.6f} align={va_align:.6f})"
        )

        if va_pred < best_val_pred:
            best_val_pred = va_pred
            patience_left = ct_patience
            torch.save(
                {
                    "ct_history_encoder": model.ct_encoder.state_dict(),
                    "projection_head": model.projection.state_dict(),
                    "val_loss_pred": va_pred,
                    "epoch": epoch,
                    "x_dim": x_dim,
                    "config": OmegaConf.to_yaml(args, resolve=True),
                },
                ckpt_path,
            )
            logger.info(f"Saved encoder checkpoint to {ckpt_path} (val_loss_pred={va_pred:.6f})")
        else:
            patience_left -= 1
            if patience_left <= 0:
                logger.info(f"Early stopping (no val_loss_pred improvement for {ct_patience} epochs).")
                break

    logger.info(f"Done. Best val_loss_pred={best_val_pred:.6f}. Encoder-only file: {ckpt_path}")


if __name__ == "__main__":
    main()
