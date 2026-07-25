"""Strictly aggregate county-major tests against same-device factual replay."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List


def _absolute(path: str, base: Path) -> Path:
    value = Path(path).expanduser()
    return value if value.is_absolute() else (base / value).resolve()


def _load_jsonl(path: Path) -> List[dict]:
    return [
        json.loads(line)
        for line in path.read_text().splitlines()
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


def _write_csv(path: Path, rows: List[dict], fieldnames: List[str]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-manifest", required=True)
    parser.add_argument("--selection", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--expected-counties", type=int, required=True)
    parser.add_argument("--expected-seeds", type=int, default=5)
    parser.add_argument("--merged-name", default="parallel_merged")
    args = parser.parse_args()

    manifest_path = Path(args.test_manifest).expanduser().resolve()
    selection_path = Path(args.selection).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = json.loads(manifest_path.read_text())
    selection = json.loads(selection_path.read_text())
    if selection.get("target_reference") != "factual_replay":
        raise ValueError("Selection does not use factual_replay as its target reference")
    selected_by_job = {row["job_id"]: row for row in selection["jobs"]}
    if len(selected_by_job) != len(selection["jobs"]):
        raise ValueError("Duplicate job IDs in selection")

    seed_rows: List[dict] = []
    factual_reference: Dict[tuple, float] = {}
    merged_manifests = []
    expected_taus = None
    for job in manifest["jobs"]:
        job_id = str(job["id"])
        selected = selected_by_job[job_id]
        out_dir = _absolute(job["out_dir"], manifest_path.parent)
        merged_dir = out_dir / args.merged_name
        metrics_path = merged_dir / "county_metrics.jsonl"
        merged_manifest_path = merged_dir / "manifest.json"
        if not metrics_path.is_file() or not merged_manifest_path.is_file():
            raise FileNotFoundError(f"Missing merged test output for {job_id}")
        rows = _load_jsonl(metrics_path)
        merged_manifest = json.loads(merged_manifest_path.read_text())
        merged_manifests.append(str(merged_manifest_path))

        labels = {row["label"] for row in rows}
        expected_label = selected["best_label"]
        if labels != {expected_label}:
            raise ValueError(
                f"{job_id} labels {sorted(labels)} do not match {expected_label}"
            )
        if {row["split"] for row in rows} != {"test"}:
            raise ValueError(f"{job_id} contains a non-test row")
        taus = sorted({int(row["tau"]) for row in rows})
        if expected_taus is None:
            expected_taus = taus
        elif taus != expected_taus:
            raise ValueError(f"{job_id} tau set differs from other jobs")
        counties = {row["county"] for row in rows}
        if len(counties) != int(args.expected_counties):
            raise ValueError(
                f"{job_id} has {len(counties)} counties, "
                f"expected {args.expected_counties}"
            )
        expected_rows = int(args.expected_counties) * len(taus)
        if len(rows) != expected_rows:
            raise ValueError(
                f"{job_id} has {len(rows)} rows, expected {expected_rows}"
            )
        keys = {(row["county"], int(row["tau"])) for row in rows}
        if len(keys) != expected_rows:
            raise ValueError(f"{job_id} has duplicate county/tau keys")

        for row in rows:
            key = (row["county"], int(row["tau"]))
            value = float(row["factual_final_per_10k"])
            previous = factual_reference.setdefault(key, value)
            if value != previous:
                raise ValueError(
                    f"GPU factual reference mismatch at {key}: "
                    f"{value} != {previous}"
                )

        for tau in taus:
            group = [row for row in rows if int(row["tau"]) == tau]
            score = _rmse(
                float(row["policy_vs_factual_final_improvement_per_10k"])
                for row in group
            )
            seed_rows.append({
                "method": job["method"],
                "seed": int(job["seed"]),
                "best_label": expected_label,
                "tau": tau,
                "factual_reference_RMSE_per_10k": score,
                "n_counties": len(group),
            })

    jobs_in_manifest = {str(job["id"]) for job in manifest["jobs"]}
    if jobs_in_manifest != set(selected_by_job):
        raise ValueError("Test manifest and selection job IDs differ")

    grouped = defaultdict(list)
    labels_by_method = defaultdict(dict)
    for row in seed_rows:
        grouped[(row["method"], row["tau"])].append(
            float(row["factual_reference_RMSE_per_10k"])
        )
        labels_by_method[row["method"]][int(row["seed"])] = row["best_label"]

    aggregate_rows = []
    for (method, tau), values in sorted(grouped.items()):
        if len(values) != int(args.expected_seeds):
            raise ValueError(
                f"{method} tau={tau} has {len(values)} seeds, "
                f"expected {args.expected_seeds}"
            )
        aggregate_rows.append({
            "method": method,
            "tau": tau,
            "factual_reference_RMSE_per_10k_mean": statistics.mean(values),
            "factual_reference_RMSE_per_10k_std": statistics.stdev(values),
            "std_ddof": 1,
            "n_seeds": len(values),
        })

    seed_csv = output_dir / "test_factual_reference_seed_metrics.csv"
    aggregate_csv = output_dir / "test_factual_reference_aggregate.csv"
    _write_csv(
        seed_csv,
        seed_rows,
        [
            "method",
            "seed",
            "best_label",
            "tau",
            "factual_reference_RMSE_per_10k",
            "n_counties",
        ],
    )
    _write_csv(
        aggregate_csv,
        aggregate_rows,
        [
            "method",
            "tau",
            "factual_reference_RMSE_per_10k_mean",
            "factual_reference_RMSE_per_10k_std",
            "std_ddof",
            "n_seeds",
        ],
    )

    summary_path = output_dir / "test_factual_reference_summary.json"
    summary_path.write_text(json.dumps({
        "schema": "epi_abm_gpu_factual_reference_summary_v1",
        "metric": {
            "name": "factual_reference_RMSE_per_10k",
            "per_seed_per_tau": (
                "sqrt(mean_over_counties("
                "policy_vs_factual_final_improvement_per_10k^2))"
            ),
            "across_seeds": "mean and sample standard deviation",
            "std_ddof": 1,
        },
        "target_reference": "same-device GPU factual replay",
        "common_random_numbers": True,
        "common_reference_keys": len(factual_reference),
        "expected_counties": int(args.expected_counties),
        "taus": expected_taus,
        "jobs": len(manifest["jobs"]),
        "best_labels_by_method": {
            method: {str(seed): label for seed, label in sorted(labels.items())}
            for method, labels in sorted(labels_by_method.items())
        },
        "seed_metrics": seed_rows,
        "aggregate": aggregate_rows,
    }, indent=2, sort_keys=True) + "\n")

    results_path = output_dir / "RESULTS_GPU_FACTUAL_REFERENCE.md"
    lines = [
        "# GPU factual-reference EpiABM results",
        "",
        "All methods use the same GPU factual replay and common random numbers.",
        "",
        "## Best checkpoints",
        "",
        "| method | seed 10 | seed 101 | seed 1010 | seed 10101 | seed 101010 |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for method, labels in sorted(labels_by_method.items()):
        lines.append(
            "| " + method + " | "
            + " | ".join(labels[seed] for seed in [10, 101, 1010, 10101, 101010])
            + " |"
        )
    lines.extend([
        "",
        "## Test RMSE at tau=21",
        "",
        "| method | RMSE /10k mean +/- std |",
        "| --- | ---: |",
    ])
    for row in aggregate_rows:
        if int(row["tau"]) == 21:
            lines.append(
                f"| {row['method']} | "
                f"{row['factual_reference_RMSE_per_10k_mean']:.6f} +/- "
                f"{row['factual_reference_RMSE_per_10k_std']:.6f} |"
            )
    results_path.write_text("\n".join(lines) + "\n")

    final_manifest_path = output_dir / "manifest.json"
    final_manifest_path.write_text(json.dumps({
        "schema": "epi_abm_gpu_factual_reference_aggregate_manifest_v1",
        "status": "complete",
        "test_manifest": str(manifest_path),
        "selection": str(selection_path),
        "target_reference": "same-device GPU factual replay",
        "common_random_numbers": True,
        "common_reference_keys": len(factual_reference),
        "jobs": len(manifest["jobs"]),
        "methods": sorted(labels_by_method),
        "merged_manifests": merged_manifests,
        "outputs": {
            path.name: {"path": str(path), "sha256": _sha256(path)}
            for path in [seed_csv, aggregate_csv, summary_path, results_path]
        },
    }, indent=2, sort_keys=True) + "\n")

    print(json.dumps({
        "event": "gpu_factual_reference_aggregate_complete",
        "jobs": len(manifest["jobs"]),
        "common_reference_keys": len(factual_reference),
        "output_dir": str(output_dir),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
