"""
Diagnostic: does the CT-trained ``OutcomePredictor`` actually respond to ``a``?

Motivation
----------
In IQL val, ``predictor:mae_uns`` stays almost constant (~2.72) while the IQL
policy changes significantly. This script probes whether the underlying
``OutcomePredictor(z, a)`` is action-insensitive, which would explain the
"frozen" predictor-world behavior and point to a structural flaw in CT.

Usage
-----
    python runnables/debug_predictor_sensitivity.py \\
        +dataset=cancer_sim_cont +model=vcip \\
        "+model/hparams/cancer=4*" \\
        exp.seed=10 dataset.coeff=4 \\
        exp.iql_inference_ckpt=/absolute/path/to/ct_best_encoder.pt \\
        +exp.debug_num_batches=4 +exp.debug_batch_size=128

What it measures
----------------
1. Counterfactual-action sweep: hold H_t fixed, sweep ``a`` on a grid in the
   training action space ([0, 1]^2 for cancer_sim), record how ``y_pred``
   moves. Report absolute range, std, per-dim slope.
2. Jacobian probe: compare ``||dy/da||`` with ``||dy/dz||`` on the same batch.
   If ``da`` contribution is <5% of the total, predictor is a-insensitive.
3. Training action diversity: compute std / unique count / behavior entropy
   on dataset ``current_treatments``. Low diversity is the root cause.
4. Factual sanity: predict ``y_hat = predictor(z, a_factual)`` on val and
   compare to ``outputs[:, -1]``. Separates "a-insensitive" from "globally
   broken predictor".
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Dict

import hydra
import numpy as np
import torch
from hydra.utils import get_original_cwd, instantiate
from omegaconf import DictConfig, OmegaConf

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.cip_dataset import CIPDataset, get_dataloader
from src.data.iql_dataset_builder import align_h_t_static_to_history
from src.models.inference_model import InferenceModel
from src.utils.inference_ckpt import load_inference_checkpoint
from src.utils.utils import repeat_static, set_seed, to_float

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("debug_predictor")

OmegaConf.register_new_resolver("toint", lambda x: int(x), replace=True)


def _hdr(title: str) -> None:
    logger.info("")
    logger.info("=" * 78)
    logger.info(title)
    logger.info("=" * 78)


@torch.no_grad()
def _collect_z_a_y(
    inference_model: InferenceModel,
    loader,
    device: str,
    max_batches: int,
) -> Dict[str, torch.Tensor]:
    Z, A, Y_target = [], [], []
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        H_t, targets = batch
        H_t = align_h_t_static_to_history(H_t)
        for k in H_t:
            H_t[k] = H_t[k].to(device)
        y_true = targets["outputs"][:, -1, :].to(device)
        z, _, _ = inference_model.ct_hidden_history(H_t)
        a = H_t["current_treatments"][:, -1, :]
        Z.append(z.detach())
        A.append(a.detach())
        Y_target.append(y_true.detach())
    return {
        "z": torch.cat(Z, dim=0),
        "a": torch.cat(A, dim=0),
        "y_true": torch.cat(Y_target, dim=0),
    }


def _test1_action_grid_sweep(
    inference_model: InferenceModel, z: torch.Tensor, a_factual: torch.Tensor, device: str
) -> None:
    _hdr("TEST 1 — counterfactual action-grid sweep (predictor mean output)")
    B = z.size(0)
    grid_vals = [0.0, 0.25, 0.5, 0.75, 1.0]
    rows = []
    for v0 in grid_vals:
        for v1 in grid_vals:
            a = torch.tensor([v0, v1], device=device).unsqueeze(0).expand(B, -1)
            za = torch.cat([z, a], dim=-1)
            with torch.no_grad():
                y = inference_model.outcome_predictor(za).squeeze(-1)
            rows.append((v0, v1, float(y.mean()), float(y.std())))

    y_means = np.asarray([r[2] for r in rows])
    logger.info("a_grid=[v0, v1] in [0,1]^2 (25 points), computed on %d (z,H_t) samples:", B)
    for v0, v1, m, s in rows:
        logger.info("  a=(%.2f, %.2f)  y_pred mean=%+.5f  std=%.5f", v0, v1, m, s)

    y_min, y_max = y_means.min(), y_means.max()
    y_rng = y_max - y_min
    y_abs_ctr = float(np.mean(np.abs(y_means)))
    logger.info("-" * 78)
    logger.info("y_pred mean across action grid:")
    logger.info("  min=%+.5f  max=%+.5f  range=%.5f", y_min, y_max, y_rng)
    logger.info("  mean|y_mean|=%.5f  →  relative range = %.2f%%",
                y_abs_ctr, 100.0 * y_rng / (y_abs_ctr + 1e-8))

    a_fmin = a_factual.min(dim=0).values.tolist()
    a_fmax = a_factual.max(dim=0).values.tolist()
    a_fstd = a_factual.std(dim=0).tolist()
    logger.info("-" * 78)
    logger.info("Factual a in this batch:  min=%s  max=%s  std=%s",
                [round(v, 4) for v in a_fmin],
                [round(v, 4) for v in a_fmax],
                [round(v, 4) for v in a_fstd])

    logger.info("")
    if y_rng < 1e-3:
        logger.info("VERDICT-1: ❌ predictor is effectively CONSTANT across action grid "
                    "(range %.2e < 1e-3)", y_rng)
    elif 100.0 * y_rng / (y_abs_ctr + 1e-8) < 5.0:
        logger.info("VERDICT-1: ⚠️  predictor responds WEAKLY to a "
                    "(relative range < 5%%) — root cause of frozen predictor world")
    else:
        logger.info("VERDICT-1: ✅ predictor responds to a with relative range "
                    "%.1f%% — investigate other causes", 100.0 * y_rng / (y_abs_ctr + 1e-8))


def _test2_jacobian_probe(
    inference_model: InferenceModel, z: torch.Tensor, a_factual: torch.Tensor, device: str
) -> None:
    _hdr("TEST 2 — Jacobian ratio ||dy/da|| vs ||dy/dz||")
    B = z.size(0)

    z_req = z.clone().detach().requires_grad_(True)
    a_req = a_factual.clone().detach().requires_grad_(True)
    za = torch.cat([z_req, a_req], dim=-1)
    y = inference_model.outcome_predictor(za).sum()
    y.backward()

    grad_a = a_req.grad.detach()
    grad_z = z_req.grad.detach()
    norm_a = float(grad_a.abs().mean())
    norm_z = float(grad_z.abs().mean())
    ratio = norm_a / (norm_z + 1e-12)

    norm_a_per_dim = grad_a.abs().mean(dim=0).tolist()
    norm_z_per_dim = grad_z.abs().mean(dim=0).tolist()

    logger.info("Evaluated on a_factual (real data actions), B=%d", B)
    logger.info("  mean |dy/da|   = %.6e   (per-dim = %s)",
                norm_a, [f"{v:.3e}" for v in norm_a_per_dim])
    logger.info("  mean |dy/dz|   = %.6e   (per-dim top-5 = %s)",
                norm_z, [f"{v:.3e}" for v in sorted(norm_z_per_dim, reverse=True)[:5]])
    logger.info("  ratio |dy/da| / |dy/dz| = %.4e", ratio)

    logger.info("")
    if ratio < 1e-2:
        logger.info("VERDICT-2: ❌ action gradient is <1%% of state gradient — predictor "
                    "IGNORES a (structural flaw)")
    elif ratio < 5e-2:
        logger.info("VERDICT-2: ⚠️  action gradient is %.2f%% of state gradient — predictor "
                    "only weakly uses a", 100.0 * ratio)
    else:
        logger.info("VERDICT-2: ✅ action gradient is %.2f%% of state gradient — predictor "
                    "uses a meaningfully", 100.0 * ratio)


def _test3_training_action_diversity(dataset_collection) -> None:
    _hdr("TEST 3 — action diversity in CT training data")
    a = dataset_collection.train_f.data["current_treatments"]
    a_np = a.cpu().numpy() if hasattr(a, "cpu") else np.asarray(a)
    flat = a_np.reshape(-1, a_np.shape[-1])

    mean = flat.mean(axis=0)
    std = flat.std(axis=0)
    mn = flat.min(axis=0)
    mx = flat.max(axis=0)
    uniq = np.unique(np.round(flat, 4), axis=0).shape[0]

    logger.info("train_f.data['current_treatments'] shape=%s", tuple(a_np.shape))
    logger.info("  per-dim mean  = %s", [round(v, 4) for v in mean.tolist()])
    logger.info("  per-dim std   = %s", [round(v, 4) for v in std.tolist()])
    logger.info("  per-dim min   = %s", [round(v, 4) for v in mn.tolist()])
    logger.info("  per-dim max   = %s", [round(v, 4) for v in mx.tolist()])
    logger.info("  #unique rows (rounded to 4 dp) = %d  (of %d total rows)",
                uniq, flat.shape[0])

    frac_zero = float(np.mean(np.all(flat == 0.0, axis=1)))
    logger.info("  fraction of all-zero action rows = %.2f%%", 100.0 * frac_zero)


def _test4_factual_sanity(
    inference_model: InferenceModel,
    z: torch.Tensor,
    a_factual: torch.Tensor,
    y_true: torch.Tensor,
    train_std_cv: float,
) -> None:
    _hdr("TEST 4 — predictor factual sanity (predict under real actions)")
    with torch.no_grad():
        za = torch.cat([z, a_factual], dim=-1)
        y_hat = inference_model.outcome_predictor(za)

    mae_norm = float((y_hat - y_true).abs().mean())
    rmse_norm = float(((y_hat - y_true) ** 2).mean().sqrt())
    mae_uns = mae_norm * train_std_cv
    rmse_uns = rmse_norm * train_std_cv

    logger.info("on val_f, B=%d (1-step ahead ground-truth outputs[:, -1])", z.size(0))
    logger.info("  MAE  (normalized) = %.6f   MAE  (unscaled) = %.4f", mae_norm, mae_uns)
    logger.info("  RMSE (normalized) = %.6f   RMSE (unscaled) = %.4f", rmse_norm, rmse_uns)
    logger.info("  y_true norm:  mean=%+.4f std=%.4f", float(y_true.mean()), float(y_true.std()))
    logger.info("  y_hat  norm:  mean=%+.4f std=%.4f", float(y_hat.mean()), float(y_hat.std()))

    logger.info("")
    if mae_uns < 0.3:
        logger.info("VERDICT-4: ✅ predictor is ACCURATE on factual a. "
                    "If Test 1/2 fail, root cause is OOD action extrapolation.")
    elif mae_uns < 1.0:
        logger.info("VERDICT-4: ⚠️  predictor has moderate factual error (MAE_uns %.3f). "
                    "Both OOD and capacity issues possible.", mae_uns)
    else:
        logger.info("VERDICT-4: ❌ predictor is INACCURATE even on factual a (MAE_uns %.3f). "
                    "CT's predictor is globally broken.", mae_uns)


@hydra.main(version_base=None, config_name="config.yaml", config_path="../configs/")
def main(args: DictConfig):
    OmegaConf.set_struct(args, False)
    set_seed(int(args.exp.seed))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt = OmegaConf.select(args, "exp.iql_inference_ckpt", default="")
    num_batches = int(OmegaConf.select(args, "exp.debug_num_batches", default=4))
    batch_size = int(OmegaConf.select(args, "exp.debug_batch_size", default=128))

    original_cwd = get_original_cwd()
    args["exp"]["processed_data_dir"] = os.path.join(original_cwd, args["exp"]["processed_data_dir"])

    dataset_collection = instantiate(args.dataset, _recursive_=True)
    dataset_collection.process_data_multi()
    dataset_collection = to_float(dataset_collection)
    if args["dataset"]["static_size"] > 0:
        dims = len(dataset_collection.train_f.data["static_features"].shape)
        if dims == 2:
            dataset_collection = repeat_static(dataset_collection)

    inference_model = InferenceModel(args).to(device)
    load_inference_checkpoint(inference_model, ckpt, device)
    inference_model.eval()

    if not getattr(inference_model, "_outcome_predictor_loaded", False):
        logger.error("outcome_predictor weights did NOT load from %s — abort.", ckpt)
        sys.exit(1)

    for p in inference_model.outcome_predictor.parameters():
        p.requires_grad_(False)

    val_loader = get_dataloader(
        CIPDataset(dataset_collection.val_f.data, args, train=False),
        batch_size=batch_size,
        shuffle=False,
    )

    logger.info("Collecting %d val batches (bs=%d) from val_f ...", num_batches, batch_size)
    bundle = _collect_z_a_y(inference_model, val_loader, device, num_batches)
    z, a_factual, y_true = bundle["z"], bundle["a"], bundle["y_true"]
    logger.info("Collected z=%s  a=%s  y_true=%s",
                tuple(z.shape), tuple(a_factual.shape), tuple(y_true.shape))

    try:
        mean_ser, std_ser = dataset_collection.train_scaling_params
        train_std_cv = float(std_ser["cancer_volume"])
    except Exception:
        train_std_cv = 1.0
    logger.info("train_std_cv = %.4f  (used to rescale norm → unscaled)", train_std_cv)

    _test1_action_grid_sweep(inference_model, z, a_factual, device)
    _test2_jacobian_probe(inference_model, z, a_factual, device)
    _test3_training_action_diversity(dataset_collection)
    _test4_factual_sanity(inference_model, z, a_factual, y_true, train_std_cv)

    _hdr("OVERALL READING GUIDE")
    logger.info("Test 1 (grid sweep, relative range):")
    logger.info("   < 5%%  →  predictor a-insensitive  (structural flaw in CT)")
    logger.info("Test 2 (|dy/da| / |dy/dz|):")
    logger.info("   < 1e-2 →  action gradients almost zero  (confirms Test 1)")
    logger.info("Test 3 (action diversity):")
    logger.info("   low std / few unique rows → behavior policy heavily confounded")
    logger.info("Test 4 (factual MAE_uns):")
    logger.info("   accurate + Test 1/2 fail → pure OOD-action problem, fixable in eval")
    logger.info("   inaccurate → CT's predictor is globally broken, retrain needed")


if __name__ == "__main__":
    main()
