"""Aggregate selected-checkpoint county-major RMSE against an explicit target."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Iterable, List


def _absolute(path: str, base: Path) -> Path:
    value = Path(path).expanduser()
    return value if value.is_absolute() else (base / value).resolve()


def _load_jsonl(path: Path) -> List[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _rmse(values: Iterable[float]) -> float:
    materialized = list(values)
    if not materialized:
        raise ValueError("Cannot calculate RMSE for an empty group")
    return math.sqrt(sum(value * value for value in materialized) / len(materialized))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_csv(path: Path, rows: List[dict]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--split", choices=["val", "test"], required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--expected-counties", type=int, default=23)
    parser.add_argument("--expected-seeds", type=int, default=5)
    parser.add_argument("--merged-name", default="parallel_merged")
    parser.add_argument("--taus", nargs="+", type=int, default=list(range(1, 22)))
    args = parser.parse_args()

    manifest_path = Path(args.manifest).expanduser().resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    expected_taus = [int(tau) for tau in args.taus]

    seed_rows = []
    target_protocols = set()
    for job in manifest["jobs"]:
        out_dir = _absolute(job["out_dir"], manifest_path.parent)
        metrics_path = out_dir / args.merged_name / "county_metrics.jsonl"
        if not metrics_path.is_file():
            raise FileNotFoundError(f"Missing merged output for {job['id']}: {metrics_path}")
        rows = _load_jsonl(metrics_path)
        labels = sorted({str(row["label"]) for row in rows})
        if len(labels) != 1:
            raise ValueError(f"Expected one selected label for {job['id']}: {labels}")
        if {str(row["split"]) for row in rows} != {args.split}:
            raise ValueError(f"Unexpected split in {metrics_path}")
        target_protocols.update(
            (
                str(row.get("target_reference")),
                str(row.get("target_mode")),
                float(row.get("target_scale", 1.0)),
                str(row.get("external_target_sha256")),
            )
            for row in rows
        )
        for tau in expected_taus:
            group = [row for row in rows if int(row["tau"]) == tau]
            counties = {str(row["county"]) for row in group}
            if len(group) != int(args.expected_counties) or len(counties) != int(
                args.expected_counties
            ):
                raise ValueError(
                    f"{job['id']} tau={tau} has rows={len(group)} "
                    f"counties={len(counties)}, expected={args.expected_counties}"
                )
            seed_rows.append(
                {
                    "method": str(job.get("method", "cripo")),
                    "seed": int(job["seed"]),
                    "split": args.split,
                    "tau": tau,
                    "checkpoint_label": labels[0],
                    "n_counties": len(counties),
                    "target_RMSE_per_10k": _rmse(
                        float(row["target_distance_per_10k"]) for row in group
                    ),
                }
            )

    if len(target_protocols) != 1:
        raise ValueError(f"Mixed target protocols: {sorted(target_protocols)}")

    grouped = defaultdict(list)
    for row in seed_rows:
        grouped[int(row["tau"])].append(float(row["target_RMSE_per_10k"]))
    aggregate_rows = []
    for tau in expected_taus:
        values = grouped[tau]
        if len(values) != int(args.expected_seeds):
            raise ValueError(
                f"tau={tau} has {len(values)} seeds, expected {args.expected_seeds}"
            )
        aggregate_rows.append(
            {
                "split": args.split,
                "tau": tau,
                "n_seeds": len(values),
                "target_RMSE_per_10k_mean": statistics.mean(values),
                "target_RMSE_per_10k_std": statistics.stdev(values),
                "std_ddof": 1,
                "seed_values": {
                    str(row["seed"]): row["target_RMSE_per_10k"]
                    for row in seed_rows
                    if int(row["tau"]) == tau
                },
            }
        )

    seed_json = output_dir / "rmse_per_10k_by_seed_tau.json"
    aggregate_json = output_dir / "rmse_per_10k_across_seeds.json"
    seed_csv = output_dir / "rmse_per_10k_by_seed_tau.csv"
    aggregate_csv = output_dir / "rmse_per_10k_across_seeds.csv"
    table_path = output_dir / "rmse_per_10k_tau1_21.md"
    seed_json.write_text(json.dumps(seed_rows, indent=2, sort_keys=True) + "\n")
    aggregate_json.write_text(
        json.dumps(aggregate_rows, indent=2, sort_keys=True) + "\n"
    )
    _write_csv(seed_csv, seed_rows)
    _write_csv(
        aggregate_csv,
        [{key: value for key, value in row.items() if key != "seed_values"} for row in aggregate_rows],
    )
    lines = [
        f"# {args.split} target RMSE per 10k",
        "",
        "| tau | RMSE per 10k (mean +/- sample std) |",
        "| ---: | ---: |",
    ]
    lines.extend(
        f"| {row['tau']} | {row['target_RMSE_per_10k_mean']:.6f} +/- "
        f"{row['target_RMSE_per_10k_std']:.6f} |"
        for row in aggregate_rows
    )
    table_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    protocol = next(iter(target_protocols))
    manifest_out = output_dir / "manifest.json"
    manifest_out.write_text(
        json.dumps(
            {
                "schema": "epi_abm_county_major_target_aggregate_v1",
                "status": "complete",
                "source_manifest": str(manifest_path),
                "split": args.split,
                "taus": expected_taus,
                "target_reference": protocol[0],
                "target_mode": protocol[1],
                "target_scale": protocol[2],
                "external_target_sha256": protocol[3],
                "std_ddof": 1,
                "outputs": {
                    path.name: {"path": str(path), "sha256": _sha256(path)}
                    for path in [seed_json, aggregate_json, seed_csv, aggregate_csv, table_path]
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    print(
        json.dumps(
            {
                "event": "county_major_target_aggregate_complete",
                "split": args.split,
                "output_dir": str(output_dir),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
