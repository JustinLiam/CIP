#!/usr/bin/env python3
"""End-to-end smoke test for the released CRIPO EpiABM data path.

This script is intentionally small but checks the full reviewer-facing path:
asset placement, isolated runtime creation, online action injection, and cache
generation. It assumes the upstream assets have already been downloaded or
generated under ``data_generation/epi_diff_abm``.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable, List


ROOT = Path(__file__).resolve().parents[2]


def norm_county(value: str) -> str:
    return str(value).strip().zfill(5)


def required_assets(epi_root: Path, county: str, date_tag: str, num_steps: int) -> List[Path]:
    paths = [
        epi_root / "data" / "multi_policy_data.csv",
        epi_root / "data" / "delphi_county_data" / f"{county}_data.csv",
        epi_root / "data" / "processed_data" / county / date_tag / "daily_data.csv",
        epi_root / "data" / "processed_data" / county / date_tag / "weekly_data.csv",
        epi_root / "populations" / "__init__.py",
        epi_root / "populations" / f"pop{county}" / "__init__.py",
        epi_root / "populations" / f"pop{county}" / "age.pickle",
        epi_root / "populations" / f"pop{county}" / "disease_stages.csv",
        epi_root / "populations" / f"pop{county}" / "intervention.csv",
        epi_root / "data" / "networks" / "covid_output_causal" / county / "mobility_networks" / "HOUSEHOLD_NETWORK.pkl",
        epi_root
        / "result_graphs"
        / county
        / date_tag
        / "0.0005_3_5_True_False"
        / "calibrated_params.txt",
    ]
    for t in range(min(int(num_steps), 3)):
        paths.extend(
            [
                epi_root / "data" / "networks" / "covid_output_causal" / county / "mobility_networks" / "schoolnets" / f"{t}.pkl",
                epi_root / "data" / "networks" / "covid_output_causal" / county / "mobility_networks" / "occnets" / f"{t}.pkl",
                epi_root / "data" / "networks" / "covid_output_causal" / county / "mobility_networks" / "randnets" / f"{t}.pkl",
            ]
        )
    return paths


def run_step(name: str, cmd: List[str], run_dir: Path, *, cwd: Path = ROOT) -> dict:
    log_path = run_dir / f"{name}.log"
    started = time.time()
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.run(
            cmd,
            cwd=str(cwd),
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
    row = {
        "name": name,
        "cmd": cmd,
        "returncode": proc.returncode,
        "elapsed_sec": round(time.time() - started, 3),
        "log": str(log_path),
    }
    print(json.dumps(row, sort_keys=True), flush=True)
    if proc.returncode != 0:
        raise RuntimeError(f"Smoke step failed: {name}; see {log_path}")
    return row


def parse_stages(raw: str) -> List[str]:
    stages = [x.strip() for x in raw.split(",") if x.strip()]
    allowed = {"assets", "isolate", "rollout", "cache"}
    unknown = [x for x in stages if x not in allowed]
    if unknown:
        raise ValueError(f"Unknown smoke stages: {unknown}; allowed={sorted(allowed)}")
    return stages


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a reviewer-facing EpiABM smoke test.")
    parser.add_argument("--epi-root", default="data_generation/epi_diff_abm")
    parser.add_argument("--county", default="01045")
    parser.add_argument("--date-tag", default="202010-202104")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--rollout-days", type=int, default=21)
    parser.add_argument("--cache-days", type=int, default=28)
    parser.add_argument("--seed", type=int, default=20260705)
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--stages", default="assets,isolate,rollout,cache")
    args = parser.parse_args()

    county = norm_county(args.county)
    epi_root = Path(args.epi_root)
    if not epi_root.is_absolute():
        epi_root = ROOT / epi_root
    run_dir = Path(args.run_dir) if args.run_dir else (
        ROOT / "results" / "epi_abm" / "smoke" / f"{county}_{time.strftime('%Y%m%d_%H%M%S')}"
    )
    if not run_dir.is_absolute():
        run_dir = ROOT / run_dir
    cache_dir = Path(args.cache_dir) if args.cache_dir else (
        ROOT / "data" / "processed" / "epi_abm" / "smoke" / run_dir.name
    )
    if not cache_dir.is_absolute():
        cache_dir = ROOT / cache_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    stages = parse_stages(args.stages)

    summary = {
        "county": county,
        "date_tag": args.date_tag,
        "epi_root": str(epi_root),
        "run_dir": str(run_dir),
        "cache_dir": str(cache_dir),
        "stages": stages,
        "steps": [],
    }

    if "assets" in stages:
        missing = [str(path) for path in required_assets(epi_root, county, args.date_tag, max(args.rollout_days, args.cache_days)) if not path.exists()]
        asset_summary = {"name": "assets", "missing": missing, "missing_count": len(missing)}
        (run_dir / "asset_check.json").write_text(json.dumps(asset_summary, indent=2, sort_keys=True), encoding="utf-8")
        print(json.dumps(asset_summary, sort_keys=True), flush=True)
        summary["steps"].append(asset_summary)
        if missing:
            raise FileNotFoundError(f"Missing {len(missing)} EpiABM assets; see {run_dir / 'asset_check.json'}")

    if "isolate" in stages:
        summary["steps"].append(
            run_step(
                "create_isolated_runtime",
                [
                    sys.executable,
                    "scripts/epi_abm/create_isolated_runtime.py",
                    "--source",
                    str(epi_root),
                    "--dest",
                    str(run_dir / "runtime"),
                    "--force",
                ],
                run_dir,
            )
        )

    if "rollout" in stages:
        burn_in = max(1, min(7, int(args.rollout_days) // 2))
        summary["steps"].append(
            run_step(
                "online_injection",
                [
                    sys.executable,
                    "scripts/epi_abm/validate_01045_online_injection.py",
                    "--county",
                    county,
                    "--date-tag",
                    args.date_tag,
                    "--epi-root",
                    str(epi_root),
                    "--device",
                    args.device,
                    "--num-steps",
                    str(args.rollout_days),
                    "--burn-in-days",
                    str(burn_in),
                    "--seed",
                    str(args.seed),
                    "--run-root",
                    str(run_dir / "online_injection"),
                ],
                run_dir,
            )
        )

    if "cache" in stages:
        summary["steps"].append(
            run_step(
                "build_01045_cache",
                [
                    sys.executable,
                    "scripts/epi_abm/build_01045_cache.py",
                    "--seed",
                    str(args.seed),
                    "--max-seq-length",
                    str(args.cache_days),
                    "--num-random-policies",
                    "0",
                    "--processed-data-dir",
                    str(cache_dir),
                    "--device",
                    args.device,
                    "--cache-version",
                    "smoke_daily_v2_continuous_factual",
                    "--force-regenerate",
                ],
                run_dir,
            )
        )

    summary_path = run_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"smoke_summary": str(summary_path)}, sort_keys=True))


if __name__ == "__main__":
    main()
