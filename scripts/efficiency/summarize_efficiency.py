"""Aggregate the KDD'26 efficiency runs into per-seed and summary CSV files."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path


FIELDS = (
    "params_deploy",
    "train_min",
    "latency_ms",
    "episode_ms_tau6",
    "episode_ms_tau12",
    "peak_gb",
)
INFERENCE_FIELDS = (
    "latency_ms",
    "episode_ms_tau6",
    "episode_ms_tau12",
)


def read_kv(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for line in path.read_text(errors="replace").splitlines():
        if "\t" in line:
            key, value = line.split("\t", 1)
            values[key] = value
    return values


def to_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def median_timing(path: Path, tau: int, field: str) -> float:
    values = []
    if not path.exists():
        return math.nan
    for line in path.read_text(errors="replace").splitlines():
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if int(record.get("tau", -1)) == tau:
            value = to_float(record.get(field))
            if math.isfinite(value):
                values.append(value)
    return statistics.median(values) if values else math.nan


def last_complexity_row(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    with path.open(newline="", errors="replace") as stream:
        rows = list(csv.DictReader(stream))
    return rows[-1] if rows else {}


def collect_cripo(task: Path, dataset: str, seed: int) -> dict:
    metadata = read_kv(task / "metadata.tsv")
    profile_path = task / "inference_profile.json"
    profile = json.loads(profile_path.read_text()) if profile_path.exists() else {}
    timing = {int(row["tau"]): row for row in profile.get("timing", [])}
    status_path = task / "status.txt"
    status = status_path.read_text().strip() if status_path.exists() else "MISSING"
    return {
        "dataset": dataset,
        "model": "cripo",
        "seed": seed,
        "status": status,
        "params_deploy": to_float(profile.get("params_deploy")),
        "train_min": to_float(metadata.get("elapsed_ms")) / 60000.0,
        "latency_ms": to_float(timing.get(1, {}).get("decision_ms")),
        "episode_ms_tau6": to_float(timing.get(6, {}).get("episode_ms")),
        "episode_ms_tau12": to_float(timing.get(12, {}).get("episode_ms")),
        "peak_gb": to_float(metadata.get("peak_train_gpu_mib")) / 1024.0,
    }


def collect_baseline(task: Path, dataset: str, model: str, seed: int) -> dict:
    metadata = read_kv(task / "metadata.tsv")
    complexity = last_complexity_row(task / "complexity_info.csv")
    status_path = task / "status.txt"
    status = status_path.read_text().strip() if status_path.exists() else "MISSING"
    params_deploy = median_timing(
        task / "inference_timing.jsonl", 1, "params_deploy"
    )
    if not math.isfinite(params_deploy):
        params_deploy = to_float(complexity.get("params"))
    return {
        "dataset": dataset,
        "model": model,
        "seed": seed,
        "status": status,
        "params_deploy": params_deploy,
        "train_min": to_float(complexity.get("train_time")) / 60.0,
        "latency_ms": median_timing(task / "inference_timing.jsonl", 1, "decision_ms"),
        "episode_ms_tau6": median_timing(
            task / "inference_timing.jsonl", 6, "episode_ms"
        ),
        "episode_ms_tau12": median_timing(
            task / "inference_timing.jsonl", 12, "episode_ms"
        ),
        "peak_gb": to_float(metadata.get("peak_train_gpu_mib")) / 1024.0,
    }


def finite(values):
    return [value for value in values if math.isfinite(value)]


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_root", type=Path)
    args = parser.parse_args()
    root = args.run_root.resolve()
    manifest = json.loads((root / "run_manifest.json").read_text())
    seeds = tuple(int(seed) for seed in manifest["seeds"])
    datasets = tuple(manifest["datasets"])
    methods = tuple(manifest["methods"])
    inference_seed = int(manifest["protocol"]["inference_seed"])
    baseline_models = tuple(model for model in methods if model != "cripo")

    rows = []
    for dataset in datasets:
        for seed in seeds:
            rows.append(collect_cripo(root / dataset / "cripo" / f"seed_{seed}", dataset, seed))
        for model in baseline_models:
            for seed in seeds:
                rows.append(
                    collect_baseline(
                        root / dataset / model / f"seed_{seed}", dataset, model, seed
                    )
                )

    per_seed_path = root / "efficiency_per_seed.csv"
    write_csv(
        per_seed_path,
        rows,
        ["dataset", "model", "seed", "status", *FIELDS],
    )

    summaries = []
    for dataset in datasets:
        for model in methods:
            group = [
                row for row in rows
                if row["dataset"] == dataset and row["model"] == model
            ]
            summary = {
                "dataset": dataset,
                "model": model,
                "n_completed": sum(row["status"] == "COMPLETED" for row in group),
                "n_total": len(group),
            }
            for field in FIELDS:
                selected = (
                    [row for row in group if row["seed"] == inference_seed]
                    if field in INFERENCE_FIELDS
                    else group
                )
                values = finite([row[field] for row in selected])
                summary[f"{field}_mean"] = statistics.mean(values) if values else math.nan
                summary[f"{field}_std"] = (
                    statistics.stdev(values) if len(values) > 1 else 0.0
                    if values else math.nan
                )
                summary[f"{field}_n"] = len(values)
            summaries.append(summary)

    summary_fields = ["dataset", "model", "n_completed", "n_total"]
    for field in FIELDS:
        summary_fields.extend((f"{field}_mean", f"{field}_std", f"{field}_n"))
    write_csv(root / "efficiency_summary.csv", summaries, summary_fields)

    missing = []
    for row in rows:
        missing_fields = [field for field in FIELDS if not math.isfinite(row[field])]
        if row["status"] != "COMPLETED" or missing_fields:
            missing.append(
                {
                    "dataset": row["dataset"],
                    "model": row["model"],
                    "seed": row["seed"],
                    "status": row["status"],
                    "missing_fields": missing_fields,
                }
            )
    report = {
        "expected_tasks": len(rows),
        "completed_tasks": sum(row["status"] == "COMPLETED" for row in rows),
        "complete_metric_rows": sum(
            row["status"] == "COMPLETED"
            and all(math.isfinite(row[field]) for field in FIELDS)
            for row in rows
        ),
        "missing_or_incomplete": missing,
    }
    (root / "completion_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps({k: v for k, v in report.items() if k != "missing_or_incomplete"}))


if __name__ == "__main__":
    main()
