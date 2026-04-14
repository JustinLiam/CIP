"""
Print distribution of targets["outputs"][:, -1] from CIPDataset (same as eval_iql_planner).

Run from repo root, same overrides as eval/train:
  python runnables/inspect_cip_targets.py +dataset=cancer_sim_cont +model=vcip
  python runnables/inspect_cip_targets.py +dataset=cancer_sim_cont +model=vcip exp.test=true
"""
import logging
import os
import sys
from pathlib import Path

import hydra
import numpy as np
from hydra.utils import get_original_cwd, instantiate
from omegaconf import DictConfig, OmegaConf

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.cip_dataset import CIPDataset, get_dataloader
from src.utils.utils import repeat_static, set_seed, to_float

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
OmegaConf.register_new_resolver("toint", lambda x: int(x), replace=True)


@hydra.main(version_base=None, config_name="config.yaml", config_path="../configs/")
def main(args: DictConfig):
    OmegaConf.set_struct(args, False)
    set_seed(args.exp.seed)
    original_cwd = Path(get_original_cwd())
    args["exp"]["processed_data_dir"] = os.path.join(str(original_cwd), args["exp"]["processed_data_dir"])

    dataset_collection = instantiate(args.dataset, _recursive_=True)
    dataset_collection.process_data_multi()
    dataset_collection = to_float(dataset_collection)
    if args["dataset"]["static_size"] > 0:
        dims = len(dataset_collection.train_f.data["static_features"].shape)
        if dims == 2:
            dataset_collection = repeat_static(dataset_collection)

    if args.exp.test:
        data = dataset_collection.test_f.data
        split_name = "test"
    else:
        data = dataset_collection.val_f.data
        split_name = "val"

    batch_size = int(OmegaConf.select(args, "exp.batch_size_val", default=512))
    loader = get_dataloader(CIPDataset(data, args, train=False), batch_size=batch_size, shuffle=False)

    mean_ser, std_ser = dataset_collection.train_scaling_params
    m = float(mean_ser["cancer_volume"])
    s = float(std_ser["cancer_volume"])

    all_last = []
    for _, targets in loader:
        y = targets["outputs"][:, -1, :].detach().cpu().numpy().reshape(-1)
        all_last.append(y)
    all_last = np.concatenate(all_last, axis=0)

    unscaled = all_last * s + m
    logger.info("=== targets['outputs'][:, -1] (factual outcome at end of projection, %s) ===", split_name)
    logger.info("Normalized space (train cancer_volume mean/std)")
    logger.info("  n=%d", all_last.size)
    logger.info("  min=%.6f  max=%.6f  mean=%.6f  std=%.6f", all_last.min(), all_last.max(), all_last.mean(), all_last.std())
    logger.info("  fraction exactly 0: %.6f", float(np.mean(all_last == 0.0)))
    logger.info("  quantiles p5,p25,p50,p75,p95: %s", np.percentile(all_last, [5, 25, 50, 75, 95]).round(6).tolist())
    logger.info("Unscaled (y_norm * std + mean)")
    logger.info("  min=%.6f  max=%.6f  mean=%.6f  std=%.6f", unscaled.min(), unscaled.max(), unscaled.mean(), unscaled.std())
    logger.info("  fraction exactly 0: %.6f", float(np.mean(unscaled == 0.0)))
    logger.info("Train scaling: mean[cancer_volume]=%.6f  std[cancer_volume]=%.6f", m, s)
    logger.info("First 30 values (normalized): %s", np.round(all_last[:30], 6).tolist())


if __name__ == "__main__":
    main()
