from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.epi_abm.audit_epi_abm_assets import audit_county, counties_from_arg, counties_from_csv
from src.data.epi_abm import EpiABMDatasetCollection


def main() -> None:
    parser = argparse.ArgumentParser(description="Build fixed multi-county EpiABM VCIP cache.")
    parser.add_argument("--seed", type=int, default=20260705)
    parser.add_argument("--max-seq-length", type=int, default=182)
    parser.add_argument("--num-random-policies", type=int, default=0)
    parser.add_argument("--epi-root", default="data_generation/epi_diff_abm")
    parser.add_argument("--date-tag", default="202010-202104")
    parser.add_argument("--processed-data-dir", default="data/processed/epi_abm/multi_county")
    parser.add_argument("--from-epicf-csv", default="data_generation/epi_diff_abm/data/multi_policy_data.csv")
    parser.add_argument("--counties", nargs="*", default=None)
    parser.add_argument("--include-unready", action="store_true", help="Try all requested counties instead of filtering to ready assets.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--intervention-mode", default="continuous_freeze")
    parser.add_argument("--behavior-policy-subset", default="factual_only")
    parser.add_argument("--random-policy-mode", default="continuous_weekly")
    parser.add_argument("--cache-version", default="daily_v2_continuous_factual")
    parser.add_argument("--split-by", choices=["county", "episode"], default="county")
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--test-frac", type=float, default=0.15)
    parser.add_argument("--force-regenerate", action="store_true")
    args = parser.parse_args()

    epi_root = Path(args.epi_root)
    counties = counties_from_arg(args.counties) if args.counties else counties_from_csv(Path(args.from_epicf_csv))
    audit_rows = [audit_county(epi_root, county, args.date_tag, args.max_seq_length) for county in counties]
    ready = [str(row["county"]) for row in audit_rows if row["ready_for_cache"]]

    selected = counties if args.include_unready else ready
    if not selected:
        raise RuntimeError("No counties are ready for cache generation. Run upstream prep/calibration first.")

    collection = EpiABMDatasetCollection(
        seed=args.seed,
        county=selected[0],
        counties=selected,
        date_tag=args.date_tag,
        epi_root=args.epi_root,
        processed_data_dir=args.processed_data_dir,
        max_seq_length=args.max_seq_length,
        projection_horizon=14,
        num_random_policies=args.num_random_policies,
        behavior_policy_subset=args.behavior_policy_subset,
        split_by=args.split_by,
        treatment_mode="continuous",
        intervention_mode=args.intervention_mode,
        random_policy_mode=args.random_policy_mode,
        cache_version=args.cache_version,
        force_regenerate=args.force_regenerate,
        generate_if_missing=True,
        device=args.device,
        split={"val": args.val_frac, "test": args.test_frac},
    )
    collection.process_data_multi()

    manifest_path = collection._manifest_path()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    summary = {
        "cache_path": str(collection._cache_path()),
        "manifest_path": str(manifest_path),
        "requested_counties": len(counties),
        "ready_counties": len(ready),
        "selected_counties": len(selected),
        "train_shape": collection.train_f.data["outputs"].shape,
        "val_shape": collection.val_f.data["outputs"].shape,
        "test_shape": collection.test_f.data["outputs"].shape,
        "split_by_effective": manifest.get("split_by_effective"),
        "split_counties": manifest.get("split_counties", {}),
        "output_means": collection.train_scaling_params["output_means"].tolist(),
        "output_stds": collection.train_scaling_params["output_stds"].tolist(),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
