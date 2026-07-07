#!/usr/bin/env python3
"""Run epi-diff-abm county calibration in isolated worker directories.

Each worker gets its own source/config directory to avoid races on
``covid_abm/yamls/config.yaml``. Heavy generated assets are shared through
symlinks to the canonical epi-diff-abm ``data/``, ``populations/``, and
``result_graphs/`` directories.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time
from multiprocessing import Process, Queue
from pathlib import Path
from typing import Dict, List, Sequence, Set

import yaml


STATE_BY_FIPS = {
    "01": "AL", "02": "AK", "04": "AZ", "05": "AR", "06": "CA", "08": "CO",
    "09": "CT", "10": "DE", "11": "DC", "12": "FL", "13": "GA", "15": "HI",
    "16": "ID", "17": "IL", "18": "IN", "19": "IA", "20": "KS", "21": "KY",
    "22": "LA", "23": "ME", "24": "MD", "25": "MA", "26": "MI", "27": "MN",
    "28": "MS", "29": "MO", "30": "MT", "31": "NE", "32": "NV", "33": "NH",
    "34": "NJ", "35": "NM", "36": "NY", "37": "NC", "38": "ND", "39": "OH",
    "40": "OK", "41": "OR", "42": "PA", "44": "RI", "45": "SC", "46": "SD",
    "47": "TN", "48": "TX", "49": "UT", "50": "VT", "51": "VA", "53": "WA",
    "54": "WV", "55": "WI", "56": "WY", "72": "PR",
}


def norm_county(value: object) -> str:
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    return text.zfill(5)


def load_counties(csv_path: Path) -> List[str]:
    counties = set()
    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            counties.add(norm_county(row["county"]))
    return sorted(counties)


def calibrated_param_path(epi_root: Path, county: str, date_tag: str) -> Path:
    return (
        epi_root
        / "result_graphs"
        / county
        / date_tag
        / "0.0005_3_5_True_False"
        / "calibrated_params.txt"
    )


def county_ready(epi_root: Path, county: str, date_tag: str, num_steps: int) -> bool:
    state = STATE_BY_FIPS[county[:2]]
    checks = [
        epi_root / "data" / "delphi_county_data" / f"{county}_data.csv",
        epi_root / "data" / "state_data" / state / county / "agents_ages.csv",
        epi_root / "data" / "state_data" / state / county / "agents_household_sizes.csv",
        epi_root / "data" / "state_data" / state / county / "agents_occupations.csv",
        epi_root / "data" / "population_data" / f"{state}_population_data" / f"{county}_population.csv",
        epi_root / "data" / "processed_data" / county / date_tag / "daily_data.csv",
        epi_root / "populations" / f"pop{county}" / "age.pickle",
        epi_root / "populations" / f"pop{county}" / "disease_stages.csv",
        epi_root / "populations" / f"pop{county}" / "intervention.csv",
        epi_root / "data" / "networks" / "covid_output_causal" / county / "mobility_networks" / "HOUSEHOLD_NETWORK.pkl",
    ]
    for t in range(min(int(num_steps), 3)):
        checks.extend(
            [
                epi_root / "data" / "networks" / "covid_output_causal" / county / "mobility_networks" / "schoolnets" / f"{t}.pkl",
                epi_root / "data" / "networks" / "covid_output_causal" / county / "mobility_networks" / "occnets" / f"{t}.pkl",
                epi_root / "data" / "networks" / "covid_output_causal" / county / "mobility_networks" / "randnets" / f"{t}.pkl",
            ]
        )
    return all(path.exists() for path in checks)


def ensure_population_package(epi_root: Path, county: str) -> None:
    pop_root = epi_root / "populations"
    pop_dir = pop_root / f"pop{county}"
    pop_dir.mkdir(parents=True, exist_ok=True)
    (pop_root / "__init__.py").touch()
    (pop_dir / "__init__.py").touch()
    mapping = pop_dir / "mapping.json"
    population_mapping = pop_dir / "population_mapping.json"
    if mapping.exists() and not population_mapping.exists():
        shutil.copyfile(mapping, population_mapping)


def select_counties(args: argparse.Namespace) -> List[str]:
    all_counties = load_counties(args.county_csv)
    ready = []
    for county in all_counties:
        if args.max_counties > 0 and len(ready) >= args.max_counties:
            break
        if args.skip_calibrated and calibrated_param_path(args.epi_root, county, args.date_tag).exists():
            continue
        if county_ready(args.epi_root, county, args.date_tag, args.num_steps):
            ready.append(county)
    return ready


def copy_worker_source(epi_root: Path, worker_epi: Path, force: bool = False) -> None:
    if worker_epi.exists() and force:
        shutil.rmtree(worker_epi)
    if worker_epi.exists():
        return

    def ignore(dir_path: str, names: Sequence[str]) -> Set[str]:
        ignored = {"data", "populations", "result_graphs", "reproduction_runs", "online_rollout_runs"}
        ignored.update(name for name in names if name == "__pycache__" or name.endswith(".pyc"))
        return ignored.intersection(names)

    shutil.copytree(epi_root, worker_epi, ignore=ignore)
    for name in ("data", "populations", "result_graphs"):
        target = epi_root / name
        link = worker_epi / name
        if link.exists() or link.is_symlink():
            link.unlink()
        link.symlink_to(target, target_is_directory=True)


def write_config(worker_epi: Path, county: str, args: argparse.Namespace, gpu: str) -> None:
    config_path = worker_epi / "covid_abm" / "yamls" / "config.yaml"
    with config_path.open("r") as f:
        cfg = yaml.safe_load(f)
    meta = cfg["simulation_metadata"]
    meta["POPULATION"] = county
    meta["DATE"] = args.date_tag
    meta["num_steps_per_episode"] = int(args.num_steps)
    meta["NUM_WEEKS"] = int(args.num_weeks)
    meta["NUM_WEEKS_TO_EVAL"] = int(args.num_weeks_to_eval)
    meta["device"] = "cuda" if gpu != "cpu" else "cpu"
    meta["GENERATING_COUNTERFACTUAL"] = False
    meta["calibration"] = True
    with config_path.open("w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)


def append_jsonl(path: Path, row: Dict[str, object]) -> None:
    with path.open("a") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")


def worker_loop(
    worker_id: int,
    gpu: str,
    tasks: Queue,
    args: argparse.Namespace,
    status_path: Path,
) -> None:
    worker_root = args.run_dir / "workers" / f"worker_{worker_id}"
    worker_epi = worker_root / "epi-diff-abm"
    worker_root.mkdir(parents=True, exist_ok=True)
    copy_worker_source(args.epi_root, worker_epi, force=args.force_worker_refresh)

    while True:
        county = tasks.get()
        if county is None:
            return

        param_path = calibrated_param_path(args.epi_root, county, args.date_tag)
        if args.skip_calibrated and param_path.exists():
            append_jsonl(status_path, {"county": county, "worker": worker_id, "status": "skip_calibrated", "time": time.time()})
            continue
        if not county_ready(args.epi_root, county, args.date_tag, args.num_steps):
            append_jsonl(status_path, {"county": county, "worker": worker_id, "status": "skip_not_ready", "time": time.time()})
            continue

        ensure_population_package(args.epi_root, county)
        write_config(worker_epi, county, args, gpu)
        log_path = args.run_dir / "logs" / f"calibrate_{county}_worker{worker_id}.log"
        env = os.environ.copy()
        if gpu != "cpu":
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        env["PYTHONPATH"] = f"{worker_epi}:{env.get('PYTHONPATH', '')}"
        start = time.time()
        append_jsonl(status_path, {"county": county, "worker": worker_id, "gpu": gpu, "status": "start", "time": start})
        with log_path.open("w") as log:
            proc = subprocess.run(
                [sys.executable, "main.py"],
                cwd=str(worker_epi),
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
            )
        elapsed = time.time() - start
        if proc.returncode == 0 and param_path.exists():
            status = "done"
        elif proc.returncode == 0:
            status = "missing_params"
        else:
            status = "failed"
        append_jsonl(
            status_path,
            {
                "county": county,
                "worker": worker_id,
                "gpu": gpu,
                "status": status,
                "returncode": proc.returncode,
                "elapsed_sec": round(elapsed, 3),
                "log": str(log_path),
                "time": time.time(),
            },
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epi-root", type=Path, default=Path("external_repos/epi-diff-abm"))
    parser.add_argument("--county-csv", type=Path, default=Path("external_repos/epi-diff-abm/data/multi_policy_data.csv"))
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--gpus", default="0,1", help="Comma-separated GPU ids, or cpu.")
    parser.add_argument("--date-tag", default="202010-202104")
    parser.add_argument("--num-steps", type=int, default=182)
    parser.add_argument("--num-weeks", type=int, default=26)
    parser.add_argument("--num-weeks-to-eval", type=int, default=24)
    parser.add_argument("--max-counties", type=int, default=0)
    parser.add_argument("--skip-calibrated", action="store_true", default=True)
    parser.add_argument("--force-worker-refresh", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.epi_root = args.epi_root.resolve()
    args.county_csv = args.county_csv.resolve()
    args.run_dir = args.run_dir.resolve()
    (args.run_dir / "logs").mkdir(parents=True, exist_ok=True)
    (args.run_dir / "workers").mkdir(parents=True, exist_ok=True)

    counties = select_counties(args)
    manifest = {
        "epi_root": str(args.epi_root),
        "county_csv": str(args.county_csv),
        "run_dir": str(args.run_dir),
        "workers": args.workers,
        "gpus": args.gpus,
        "date_tag": args.date_tag,
        "num_steps": args.num_steps,
        "selected_counties": counties,
        "selected_county_count": len(counties),
        "dry_run": args.dry_run,
        "time": time.time(),
    }
    (args.run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
    print(json.dumps(manifest, indent=2, sort_keys=True))
    if args.dry_run or not counties:
        return

    gpu_list = ["cpu"] if args.gpus.strip().lower() == "cpu" else [x.strip() for x in args.gpus.split(",") if x.strip()]
    tasks: Queue = Queue()
    for county in counties:
        tasks.put(county)
    for _ in range(max(1, args.workers)):
        tasks.put(None)
    status_path = args.run_dir / "status.jsonl"
    processes: List[Process] = []
    for worker_id in range(max(1, args.workers)):
        gpu = gpu_list[worker_id % len(gpu_list)]
        proc = Process(target=worker_loop, args=(worker_id, gpu, tasks, args, status_path))
        proc.start()
        processes.append(proc)
    for proc in processes:
        proc.join()
    failed = [proc.exitcode for proc in processes if proc.exitcode != 0]
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
