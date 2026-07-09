#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import json
import sys
import time
import types
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from src.data.epi_abm import EpiABMWeeklyEnv  # noqa: E402
except ModuleNotFoundError as exc:
    if exc.name != "pytorch_lightning":
        raise
    package_name = "epi_abm_standalone"
    package = types.ModuleType(package_name)
    package.__path__ = [str(ROOT / "src" / "data" / "epi_abm")]
    sys.modules[package_name] = package
    EpiABMWeeklyEnv = importlib.import_module(f"{package_name}.weekly_env").EpiABMWeeklyEnv


def parse_args():
    parser = argparse.ArgumentParser(description="Validate direct online action injection for EpiABM 01045.")
    parser.add_argument("--county", default="01045")
    parser.add_argument("--date-tag", default="202010-202104")
    parser.add_argument("--epi-root", default=str(ROOT / "data_generation" / "epi_diff_abm"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-steps", type=int, default=182)
    parser.add_argument("--burn-in-days", type=int, default=28)
    parser.add_argument("--action-hold-days", type=int, default=7)
    parser.add_argument("--seed", type=int, default=20260704)
    parser.add_argument("--run-root", default=None)
    return parser.parse_args()


def make_weekly_hold(actions: np.ndarray, hold_days: int, start_day: int) -> np.ndarray:
    out = np.asarray(actions, dtype=np.float32).copy()
    for start in range(int(start_day), out.shape[0], int(hold_days)):
        out[start : start + int(hold_days)] = out[start]
    return out


def run_daily(env: EpiABMWeeklyEnv, actions: np.ndarray, seed: int) -> np.ndarray:
    env.reset(decision_day=0, seed=seed)
    ys = []
    for day in range(actions.shape[0]):
        y, _ = env.step_day(actions[day])
        ys.append(y)
    return np.asarray(ys, dtype=np.float32)


def run_segmented(env: EpiABMWeeklyEnv, actions: np.ndarray, burn_in_days: int, seed: int) -> np.ndarray:
    env.reset(decision_day=burn_in_days, seed=seed, burn_in_actions=actions)
    prefix_env = EpiABMWeeklyEnv(
        county=env.county,
        date_tag=env.date_tag,
        epi_root=str(env.epi_root),
        device=env.device,
        action_hold_days=env.action_hold_days,
        num_steps=env.num_steps,
    )
    prefix_env.reset(decision_day=0, seed=seed)
    prefix = []
    for day in range(burn_in_days):
        y, _ = prefix_env.step_day(actions[day])
        prefix.append(y)

    suffix = []
    while env.current_day < env.num_steps:
        y_week, _ = env.step_week(actions[env.current_day])
        suffix.extend(y_week.reshape(-1).tolist())
    return np.asarray(prefix + suffix, dtype=np.float32)


def main():
    args = parse_args()
    run_root = Path(args.run_root) if args.run_root else (
        ROOT / "results" / "epi_abm" / "smoke" / f"01045_online_injection_{time.strftime('%Y%m%d_%H%M%S')}"
    )
    run_root.mkdir(parents=True, exist_ok=True)

    env = EpiABMWeeklyEnv(
        county=args.county,
        date_tag=args.date_tag,
        epi_root=args.epi_root,
        device=args.device,
        action_hold_days=args.action_hold_days,
        num_steps=args.num_steps,
    )
    factual = env.read_factual_interventions()
    effective = make_weekly_hold(factual[: args.num_steps], args.action_hold_days, args.burn_in_days)

    full_daily = run_daily(env, effective, args.seed)
    segmented_env = EpiABMWeeklyEnv(
        county=args.county,
        date_tag=args.date_tag,
        epi_root=args.epi_root,
        device=args.device,
        action_hold_days=args.action_hold_days,
        num_steps=args.num_steps,
    )
    segmented = run_segmented(segmented_env, effective, args.burn_in_days, args.seed)

    daily_df = pd.DataFrame({
        "day": np.arange(args.num_steps),
        "full_daily_cases": full_daily,
        "segmented_cases": segmented,
        "abs_diff": np.abs(full_daily - segmented),
    })
    daily_path = run_root / "daily_comparison.csv"
    daily_df.to_csv(daily_path, index=False)

    n_weeks = args.num_steps // args.action_hold_days
    trim = n_weeks * args.action_hold_days
    full_weekly = full_daily[:trim].reshape(n_weeks, args.action_hold_days).sum(axis=1)
    seg_weekly = segmented[:trim].reshape(n_weeks, args.action_hold_days).sum(axis=1)
    weekly_df = pd.DataFrame({
        "week": np.arange(n_weeks),
        "full_weekly_cases": full_weekly,
        "segmented_weekly_cases": seg_weekly,
        "abs_diff": np.abs(full_weekly - seg_weekly),
    })
    weekly_path = run_root / "weekly_comparison.csv"
    weekly_df.to_csv(weekly_path, index=False)

    summary = {
        "county": args.county,
        "date_tag": args.date_tag,
        "seed": args.seed,
        "num_steps": args.num_steps,
        "burn_in_days": args.burn_in_days,
        "action_hold_days": args.action_hold_days,
        "daily_max_abs_diff": float(daily_df["abs_diff"].max()),
        "daily_mean_abs_diff": float(daily_df["abs_diff"].mean()),
        "weekly_max_abs_diff": float(weekly_df["abs_diff"].max()),
        "weekly_mean_abs_diff": float(weekly_df["abs_diff"].mean()),
        "daily_comparison_csv": str(daily_path),
        "weekly_comparison_csv": str(weekly_path),
    }
    (run_root / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
