from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.epi_abm import EpiABMDatasetCollection


def main() -> None:
    parser = argparse.ArgumentParser(description="Build fixed 01045 EpiABM VCIP cache.")
    parser.add_argument("--seed", type=int, default=20260705)
    parser.add_argument("--max-seq-length", type=int, default=182)
    parser.add_argument("--num-random-policies", type=int, default=0)
    parser.add_argument("--processed-data-dir", default="data/processed/epi_abm/01045")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--intervention-mode", default="continuous_freeze")
    parser.add_argument("--random-policy-mode", default="continuous_weekly")
    parser.add_argument("--cache-version", default="daily_v2_continuous_factual")
    parser.add_argument("--force-regenerate", action="store_true")
    args = parser.parse_args()

    collection = EpiABMDatasetCollection(
        seed=args.seed,
        county="01045",
        counties=["01045"],
        processed_data_dir=args.processed_data_dir,
        max_seq_length=args.max_seq_length,
        projection_horizon=14,
        num_random_policies=args.num_random_policies,
        behavior_policy_subset="factual_only",
        treatment_mode="continuous",
        intervention_mode=args.intervention_mode,
        random_policy_mode=args.random_policy_mode,
        cache_version=args.cache_version,
        force_regenerate=args.force_regenerate,
        generate_if_missing=True,
        device=args.device,
        split={"val": 0.25, "test": 0.25},
    )
    collection.process_data_multi()

    cache_path = Path(args.processed_data_dir) / f"01045_202010-202104_{args.cache_version}.pkl"
    manifest_path = cache_path.with_suffix(".manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    summary = {
        "cache_path": str(cache_path),
        "manifest_path": str(manifest_path),
        "train_shape": collection.train_f.data["outputs"].shape,
        "val_shape": collection.val_f.data["outputs"].shape,
        "test_shape": collection.test_f.data["outputs"].shape,
        "split_indices": manifest["split_indices"],
        "policy_names": manifest.get("policy_names", []),
        "output_means": collection.train_scaling_params["output_means"].tolist(),
        "output_stds": collection.train_scaling_params["output_stds"].tolist(),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
