#!/usr/bin/env python3
"""Derive training-only peak GPU memory from timestamped resource samples."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path


TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S"


def read_metadata(path: Path) -> dict[str, str]:
    result = {}
    for line in path.read_text(errors="replace").splitlines():
        if "\t" in line:
            key, value = line.split("\t", 1)
            result[key] = value
    return result


def set_metadata(path: Path, key: str, value: str) -> None:
    lines = path.read_text(errors="replace").splitlines()
    output = []
    replaced = False
    for line in lines:
        if line.startswith(f"{key}\t"):
            output.append(f"{key}\t{value}")
            replaced = True
        else:
            output.append(line)
    if not replaced:
        output.append(f"{key}\t{value}")
    path.write_text("\n".join(output) + "\n")


def parse_timestamp(line: str) -> float | None:
    try:
        stamp = datetime.strptime(line[:19], TIMESTAMP_FORMAT)
    except ValueError:
        return None
    return stamp.astimezone().timestamp()


def evaluation_start(log_path: Path) -> float:
    last_timestamp = None
    for line in log_path.read_text(errors="replace").splitlines():
        current = parse_timestamp(line)
        if current is not None:
            last_timestamp = current
        if "Starting evaluation..." in line:
            if last_timestamp is None:
                break
            return last_timestamp
        if "评估模型" in line and "tau=1" in line:
            if last_timestamp is None:
                break
            return last_timestamp
    raise ValueError(f"missing evaluation boundary in {log_path}")


def peak_before(resource_path: Path, cutoff: float | None) -> int:
    values = []
    with resource_path.open(newline="", errors="replace") as stream:
        for row in csv.DictReader(stream, delimiter="\t"):
            try:
                timestamp = float(row["unix_time"])
                used_mib = int(row["gpu_used_mib"])
            except (KeyError, TypeError, ValueError):
                continue
            if cutoff is None or timestamp < cutoff:
                values.append(used_mib)
    if not values:
        raise ValueError(f"no training resource samples in {resource_path}")
    return max(values)


def derive(task: Path) -> int:
    metadata_path = task / "metadata.tsv"
    metadata = read_metadata(metadata_path)
    resource_path = task / "logs" / "resources.tsv"
    model = task.parent.name
    if model == "cripo":
        cutoff = None
    else:
        complexity_source = Path(metadata["complexity_source"])
        log_dir = complexity_source.parent.parent / "logs"
        seed = task.name[len("seed_"):]
        candidates = sorted(
            log_dir.glob(f"seed_{seed}_*.log"),
            key=lambda path: path.stat().st_mtime,
        )
        if not candidates:
            raise ValueError(f"missing timestamped training log for {task}")
        cutoff = evaluation_start(candidates[-1])
    peak = peak_before(resource_path, cutoff)
    set_metadata(metadata_path, "peak_train_gpu_mib", str(peak))
    return peak


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_root", type=Path)
    args = parser.parse_args()
    tasks = sorted(args.run_root.resolve().glob("*/*/seed_*"))
    completed = [
        task for task in tasks
        if (task / "status.txt").read_text(errors="replace").strip() == "COMPLETED"
    ]
    failures = []
    for task in completed:
        try:
            derive(task)
        except (KeyError, OSError, ValueError) as error:
            failures.append(str(error))
    print(f"derived_training_peaks={len(completed) - len(failures)}")
    if failures:
        raise SystemExit("\n".join(failures))


if __name__ == "__main__":
    main()
