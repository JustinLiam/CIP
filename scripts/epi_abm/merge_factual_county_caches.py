from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.epi_abm.audit_epi_abm_assets import counties_from_arg


def cache_label(counties: List[str]) -> str:
    label = counties[0] if len(counties) == 1 else "multi_" + "_".join(counties[:3])
    if len(counties) > 3:
        label = f"{label}_plus{len(counties) - 3}"
    return label


def make_county_split(data: Dict[str, np.ndarray], *, seed: int, val_frac: float, test_frac: float):
    row_counties = np.asarray([f"{int(float(x)):05d}" for x in data["sim_county_id"][:, 0, 0]])
    counties = np.asarray(sorted(set(row_counties.tolist())))
    if counties.size < 3:
        raise ValueError("county split requires at least three counties.")

    rng = np.random.RandomState(seed)
    shuffled = counties.copy()
    rng.shuffle(shuffled)
    n_counties = int(shuffled.size)
    n_test = max(1, int(round(n_counties * float(test_frac))))
    n_val = max(1, int(round(n_counties * float(val_frac))))
    if n_test + n_val >= n_counties:
        n_test = 1
        n_val = 1
    n_train = n_counties - n_val - n_test

    train_counties = set(shuffled[:n_train].tolist())
    val_counties = set(shuffled[n_train : n_train + n_val].tolist())
    test_counties = set(shuffled[n_train + n_val :].tolist())

    split_indices = {
        "train": np.where(np.isin(row_counties, list(train_counties)))[0].astype(int).tolist(),
        "val": np.where(np.isin(row_counties, list(val_counties)))[0].astype(int).tolist(),
        "test": np.where(np.isin(row_counties, list(test_counties)))[0].astype(int).tolist(),
    }
    split_counties = {
        "train": sorted(train_counties),
        "val": sorted(val_counties),
        "test": sorted(test_counties),
    }
    return split_indices, split_counties


def concat_data(rows: List[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
    out = {}
    keys = rows[0].keys()
    for key in keys:
        values = [row[key] for row in rows]
        if hasattr(values[0], "shape") and values[0].shape[:1] == (1,):
            out[key] = np.concatenate(values, axis=0)
        else:
            out[key] = values[0]
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge one-county factual EpiABM caches into a county-split cache.")
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--processed-data-dir", required=True)
    parser.add_argument("--counties", nargs="+", required=True)
    parser.add_argument("--date-tag", default="202010-202104")
    parser.add_argument("--cache-version", default="daily_v2_continuous_factual")
    parser.add_argument("--seed", type=int, default=20260705)
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--test-frac", type=float, default=0.15)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.processed_data_dir)
    counties = counties_from_arg(args.counties)
    data_rows: List[Dict[str, np.ndarray]] = []
    episode_actions: Dict[int, np.ndarray] = {}
    policy_names: List[str] = []

    for episode_id, county in enumerate(counties):
        path = input_dir / f"{county}_{args.date_tag}_{args.cache_version}.pkl"
        if not path.exists():
            raise FileNotFoundError(f"Missing one-county cache for {county}: {path}")
        with path.open("rb") as f:
            bundle = pickle.load(f)
        data = {k: np.asarray(v).copy() for k, v in bundle["data"].items()}
        if data["active_entries"].shape[0] != 1:
            raise ValueError(f"Expected one row in {path}, got {data['active_entries'].shape[0]}.")
        data["sim_episode_id"][:] = float(episode_id)
        data_rows.append(data)
        action = bundle["episode_actions"].get(0)
        if action is None:
            action = next(iter(bundle["episode_actions"].values()))
        episode_actions[episode_id] = np.asarray(action, dtype=np.float32)
        policy_names.append(f"{county}:factual")

    data = concat_data(data_rows)
    split_indices, split_counties = make_county_split(
        data,
        seed=args.seed,
        val_frac=args.val_frac,
        test_frac=args.test_frac,
    )
    metadata = {
        "policy_names": policy_names,
        "num_episodes": len(policy_names),
        "policy_name_by_episode": {str(i): name for i, name in enumerate(policy_names)},
        "counties": counties,
        "behavior_policy_subset": "factual_only",
        "split_by_requested": "county",
        "split_by_effective": "county",
        "split_seed": args.seed,
        "split_counties": split_counties,
    }
    bundle = {
        "data": data,
        "episode_actions": episode_actions,
        "metadata": metadata,
        "split_indices": split_indices,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{cache_label(counties)}_{args.date_tag}_{args.cache_version}.pkl"
    with out_path.open("wb") as f:
        pickle.dump(bundle, f)

    manifest = dict(metadata)
    manifest.update(
        {
            "county": counties[0],
            "date_tag": args.date_tag,
            "seed": args.seed,
            "cache_version": args.cache_version,
            "cache_path": str(out_path),
            "split_indices": split_indices,
        }
    )
    manifest_path = out_path.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    summary = {
        "cache_path": str(out_path),
        "manifest_path": str(manifest_path),
        "num_episodes": len(policy_names),
        "split_counties": {k: len(v) for k, v in split_counties.items()},
        "outputs_shape": list(data["unscaled_outputs"].shape),
        "non_factual_policy_names": [name for name in policy_names if not name.endswith(":factual")],
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
