"""Strict completion audit for the KDD'26 efficiency experiment."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path


METRICS = (
    "params_deploy",
    "train_min",
    "latency_ms",
    "episode_ms_tau6",
    "episode_ms_tau12",
    "peak_gb",
)


def finite(value: str) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def positive(value: str) -> bool:
    try:
        return math.isfinite(float(value)) and float(value) > 0
    except (TypeError, ValueError):
        return False


def read_json_lines(path: Path) -> list[dict]:
    records = []
    if not path.exists():
        return records
    for line in path.read_text(errors="replace").splitlines():
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(record, dict):
            records.append(record)
    return records


def has_tau(record: dict, tau: int) -> bool:
    try:
        return int(record.get("tau", -1)) == tau
    except (TypeError, ValueError):
        return False


def integer(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def metadata(path: Path) -> dict[str, str]:
    result = {}
    if path.exists():
        for line in path.read_text(errors="replace").splitlines():
            if "\t" in line:
                key, value = line.split("\t", 1)
                result[key] = value
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_root", type=Path)
    args = parser.parse_args()
    root = args.run_root.resolve()
    manifest_path = root / "run_manifest.json"
    per_seed_path = root / "efficiency_per_seed.csv"

    failures: list[str] = []
    if not manifest_path.exists():
        failures.append("missing run_manifest.json")
        manifest = {}
    else:
        manifest = json.loads(manifest_path.read_text())
    if not per_seed_path.exists():
        failures.append("missing efficiency_per_seed.csv")
        rows = []
    else:
        with per_seed_path.open(newline="") as stream:
            rows = list(csv.DictReader(stream))

    expected = {
        (dataset, method, str(seed))
        for dataset in manifest.get("datasets", [])
        for method in manifest.get("methods", [])
        for seed in manifest.get("seeds", [])
    }
    actual = {(row["dataset"], row["model"], row["seed"]) for row in rows}
    if actual != expected:
        failures.append(
            f"task-key mismatch: missing={sorted(expected - actual)}, "
            f"unexpected={sorted(actual - expected)}"
        )

    backup_commit = manifest.get("backup", {}).get("commit")
    protocol = manifest.get("protocol", {})
    expected_batches = {
        "tumor": int(protocol.get("inference_batch_size_tumor", 0)),
        "mimic": int(protocol.get("inference_batch_size_mimic", 0)),
    }
    for row in rows:
        key = f"{row['dataset']}/{row['model']}/seed_{row['seed']}"
        task = root / row["dataset"] / row["model"] / f"seed_{row['seed']}"
        if row["status"] != "COMPLETED":
            failures.append(f"{key}: status={row['status']}")
        missing_metrics = [metric for metric in METRICS if not finite(row.get(metric))]
        if missing_metrics:
            failures.append(f"{key}: missing metrics {missing_metrics}")
        nonpositive_metrics = [
            metric for metric in METRICS if finite(row.get(metric))
            and not positive(row.get(metric))
        ]
        if nonpositive_metrics:
            failures.append(f"{key}: non-positive metrics {nonpositive_metrics}")
        if finite(row.get("params_deploy")):
            params = float(row["params_deploy"])
            if not params.is_integer():
                failures.append(f"{key}: non-integer params_deploy={params}")

        meta = metadata(task / "metadata.tsv")
        if meta.get("exit_code") != "0":
            failures.append(f"{key}: exit_code={meta.get('exit_code')!r}")
        if backup_commit and meta.get("git_commit") != backup_commit:
            failures.append(
                f"{key}: git_commit={meta.get('git_commit')!r}, expected={backup_commit}"
            )
        resource = task / "logs" / "resources.tsv"
        if not resource.exists() or len(resource.read_text(errors="replace").splitlines()) < 2:
            failures.append(f"{key}: missing resource samples")
        errors = task / "logs" / "error_signatures.txt"
        if errors.exists() and errors.read_text(errors="replace").strip():
            failures.append(f"{key}: non-empty error_signatures.txt")

        if row["model"] == "cripo":
            profile_path = task / "inference_profile.json"
            required = (
                task / "checkpoints" / "ct_iql_em_best.pt",
                profile_path,
            )
            if profile_path.exists():
                try:
                    profile = json.loads(profile_path.read_text())
                    timing = {
                        int(record["tau"]): record
                        for record in profile.get("timing", [])
                    }
                except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                    profile = {}
                    timing = {}
                expected_batch = expected_batches.get(row["dataset"], 0)
                if integer(profile.get("batch_size")) != expected_batch:
                    failures.append(
                        f"{key}: profile batch_size={profile.get('batch_size')}, "
                        f"expected={expected_batch}"
                    )
                for tau, field in (
                    (1, "decision_ms"),
                    (6, "episode_ms"),
                    (12, "episode_ms"),
                ):
                    if tau not in timing or not positive(timing[tau].get(field)):
                        failures.append(
                            f"{key}: missing positive raw timing {field} at tau={tau}"
                        )
        else:
            timing_path = task / "inference_timing.jsonl"
            required = (
                task / "complexity_info.csv",
                timing_path,
            )
            timing = read_json_lines(timing_path)
            expected_batch = expected_batches.get(row["dataset"], 0)
            for tau, field in (
                (1, "decision_ms"),
                (6, "episode_ms"),
                (12, "episode_ms"),
            ):
                matching = [
                    record for record in timing
                    if has_tau(record, tau) and positive(record.get(field))
                ]
                if not matching:
                    failures.append(
                        f"{key}: missing positive raw timing {field} at tau={tau}"
                    )
                observed_batches = {
                    integer(record.get("batch_size"))
                    for record in matching
                    if positive(record.get("batch_size"))
                }
                if observed_batches != {expected_batch}:
                    failures.append(
                        f"{key}: raw timing batch sizes at tau={tau} are "
                        f"{sorted(observed_batches)}, expected={[expected_batch]}"
                    )
        for path in required:
            if not path.exists() or path.stat().st_size == 0:
                failures.append(f"{key}: missing {path.name}")

    for dataset in manifest.get("datasets", []):
        for method in manifest.get("methods", []):
            group = [
                row for row in rows
                if row["dataset"] == dataset and row["model"] == method
            ]
            params = {
                int(float(row["params_deploy"]))
                for row in group
                if finite(row.get("params_deploy"))
            }
            if len(group) == len(manifest.get("seeds", [])) and len(params) != 1:
                failures.append(
                    f"{dataset}/{method}: inconsistent params_deploy={sorted(params)}"
                )

    report = {
        "passed": not failures,
        "expected_tasks": len(expected),
        "observed_tasks": len(rows),
        "failures": failures,
    }
    output = root / "audit_report.json"
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "failures"}))
    if failures:
        print(f"failures={len(failures)}; see {output}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
