"""Evaluate held-out representation-intervention balance for one EM checkpoint."""
from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

import hydra
import numpy as np
import torch
from hydra.utils import get_original_cwd, instantiate
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.ct_transition_dataset import (  # noqa: E402
    CTEstepDataset,
    _covariate_stream_dim,
    collate_ct_estep_batch,
)
from src.evaluation.weightnet_balance import stratified_balance_metrics  # noqa: E402
from src.models.ct_encoder_weight import (  # noqa: E402
    CTEncoderWeightModel,
    normalize_log_weights_by_stratum,
)
from src.utils.em_ckpt import load_em_ct_model  # noqa: E402
from src.utils.stable_iql_em_defaults import stable_select  # noqa: E402
from src.utils.utils import set_seed, to_float  # noqa: E402

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
OmegaConf.register_new_resolver("toint", lambda x: int(x), replace=True)


def _resolve_path(raw: str, original_cwd: Path) -> Path:
    path = Path(str(raw))
    return path if path.is_absolute() else original_cwd / path


@hydra.main(version_base=None, config_name="config.yaml", config_path="../configs/")
def main(args: DictConfig) -> None:
    OmegaConf.set_struct(args, False)
    original_cwd = Path(get_original_cwd())
    args["exp"]["processed_data_dir"] = os.path.join(
        str(original_cwd), args["exp"]["processed_data_dir"]
    )
    seed = int(args.exp.seed)
    set_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = _resolve_path(
        str(OmegaConf.select(args, "exp.balance_ckpt", default="")), original_cwd
    )
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Balance checkpoint not found: {checkpoint}")
    output_path = _resolve_path(
        str(OmegaConf.select(args, "exp.balance_out", default="balance.json")),
        original_cwd,
    )
    split_raw = str(OmegaConf.select(args, "exp.balance_split", default="both"))
    splits = ["val", "test"] if split_raw == "both" else [split_raw]
    if any(split not in {"val", "test"} for split in splits):
        raise ValueError(f"balance_split must be val, test, or both; got {split_raw!r}")

    dataset_collection = instantiate(args.dataset, _recursive_=True)
    for fold_name in ("train_f", "val_f", "test_f"):
        fold = getattr(dataset_collection, fold_name, None)
        if fold is not None:
            fold.process_data(dataset_collection.train_scaling_params)
    dataset_collection = to_float(dataset_collection)
    if int(args.dataset.static_size) > 0:
        for fold_name in ("train_f", "val_f", "test_f"):
            fold = getattr(dataset_collection, fold_name, None)
            if fold is None or "static_features" not in fold.data:
                continue
            static = fold.data["static_features"]
            if static.ndim == 2:
                fold.data["static_features"] = np.repeat(
                    static[:, None, :],
                    fold.data["outputs"].shape[1],
                    axis=1,
                )

    dataset_cfg = OmegaConf.to_container(args.dataset, resolve=True)
    model = CTEncoderWeightModel(
        args, _covariate_stream_dim(dataset_cfg)
    ).to(device)
    checkpoint_obj = load_em_ct_model(model, str(checkpoint), device)
    model.eval()
    use_weight_net = bool(stable_select(args, "exp.ct_use_weight_net"))
    weight_max = stable_select(args, "exp.iql_weight_max")
    weight_max = None if weight_max is None else float(weight_max)
    batch_size = int(
        OmegaConf.select(
            args,
            "exp.balance_batch_size",
            default=stable_select(args, "exp.ct_batch_size"),
        )
    )
    min_samples = int(
        OmegaConf.select(args, "exp.balance_min_samples", default=8)
    )

    result = {
        "schema": "cripo_weightnet_balance_v1",
        "seed": seed,
        "dataset_seed": int(args.dataset.seed),
        "kappa": int(args.dataset.coeff),
        "checkpoint": str(checkpoint),
        "best_outer": int(checkpoint_obj["outer_iter"]),
        "use_weight_net": use_weight_net,
        "align_loss": str(stable_select(args, "exp.ct_align_loss")),
        "weight_max": weight_max,
        "splits": {},
    }
    for split in splits:
        fold = (
            dataset_collection.val_f
            if split == "val"
            else dataset_collection.test_f
        )
        loader = DataLoader(
            CTEstepDataset(fold.data),
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=collate_ct_estep_batch,
            drop_last=False,
        )
        with torch.no_grad():
            z, action, active, _, time_index = model._encode_full_dataset(
                loader, device
            )
            valid = active > 0.5
            z = z[valid]
            action = action[valid]
            time_index = time_index[valid]
            if use_weight_net:
                logits = model.weight_net(torch.cat([z, action], dim=-1))
                weights = normalize_log_weights_by_stratum(
                    logits, time_index, weight_max=weight_max
                )
            else:
                weights = torch.ones_like(time_index, dtype=z.dtype)
        aggregate, per_time = stratified_balance_metrics(
            z,
            action,
            weights,
            time_index,
            min_samples=min_samples,
            shuffle_seed=seed + (17011 if split == "val" else 29009),
        )
        result["splits"][split] = {
            "aggregate": aggregate,
            "per_time": per_time,
        }
        logger.info(
            "Balance %s | dCor %.5f->%.5f shuffled=%.5f | "
            "nHSIC %.5f->%.5f | ESS mean/min %.3f/%.3f",
            split,
            aggregate["uniform_distance_correlation"],
            aggregate["weighted_distance_correlation"],
            aggregate["shuffled_distance_correlation"],
            aggregate["uniform_normalized_hsic"],
            aggregate["weighted_normalized_hsic"],
            aggregate["ess_fraction_mean"],
            aggregate["ess_fraction_min"],
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True))
    logger.info("Wrote balance diagnostics to %s", output_path)


if __name__ == "__main__":
    main()
