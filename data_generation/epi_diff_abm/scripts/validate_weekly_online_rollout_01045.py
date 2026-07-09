#!/usr/bin/env python3
"""Validate a burn-in + weekly-step ABM rollout against a full daily replay.

This script is intentionally a wrapper around the upstream ABM mechanics:
it does not patch the transition function. For this first smoke test, weekly
actions are supplied through a temporary intervention.csv schedule, because
the current upstream transition reads interventions from disk each day.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import os
import random
import shutil
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent_torch.core.dataloader import LoadPopulation  # noqa: E402
from agent_torch.core.executor import Executor  # noqa: E402
from agent_torch.core.helpers import to_cpu  # noqa: E402
from abm_nets import map_and_replace_tensor  # noqa: E402
import covid_abm  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run 01045 burn-in + weekly ABM rollout and compare it with a full daily replay."
    )
    parser.add_argument("--county", default="01045")
    parser.add_argument("--date-tag", default="202010-202104")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-steps", type=int, default=182)
    parser.add_argument("--num-weeks", type=int, default=26)
    parser.add_argument("--num-weeks-to-eval", type=int, default=24)
    parser.add_argument("--burn-in-days", type=int, default=28)
    parser.add_argument("--seed", type=int, default=20260704)
    parser.add_argument("--run-root", default=None)
    parser.add_argument(
        "--policy-mode",
        choices=["factual_weekly_hold", "all_open", "all_closed"],
        default="factual_weekly_hold",
        help=(
            "Post burn-in policy schedule. factual_weekly_hold uses the first "
            "factual action in each rollout week and holds it for seven days."
        ),
    )
    parser.add_argument("--school-action", type=int, choices=[0, 1], default=0)
    parser.add_argument("--occ-action", type=int, choices=[0, 1], default=0)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def deep_clone(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.clone()
    if isinstance(value, dict):
        return {k: deep_clone(v) for k, v in value.items()}
    if isinstance(value, list):
        return [deep_clone(v) for v in value]
    if isinstance(value, tuple):
        return tuple(deep_clone(v) for v in value)
    return value


@contextmanager
def temporary_config(args: argparse.Namespace):
    config_path = ROOT / "covid_abm" / "yamls" / "config.yaml"
    backup_text = config_path.read_text()
    cfg = yaml.safe_load(backup_text)
    meta = cfg["simulation_metadata"]
    meta["POPULATION"] = str(args.county)
    meta["DATE"] = str(args.date_tag)
    meta["num_steps_per_episode"] = int(args.num_steps)
    meta["NUM_WEEKS"] = int(args.num_weeks)
    meta["NUM_WEEKS_TO_EVAL"] = int(args.num_weeks_to_eval)
    meta["device"] = str(args.device)
    meta["GENERATING_COUNTERFACTUAL"] = False
    meta["calibration"] = True
    config_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    try:
        yield config_path
    finally:
        config_path.write_text(backup_text)


def make_weekly_intervention(
    original: pd.DataFrame,
    *,
    burn_in_days: int,
    num_steps: int,
    policy_mode: str,
    school_action: int,
    occ_action: int,
) -> pd.DataFrame:
    required = {"t", "school_intervention", "occ_intervention"}
    missing = required - set(original.columns)
    if missing:
        raise ValueError(f"intervention.csv missing required columns: {sorted(missing)}")

    df = original.copy()
    df["t"] = df["t"].astype(int)
    df = df.sort_values("t").reset_index(drop=True)
    if len(df) < num_steps:
        raise ValueError(f"intervention.csv has {len(df)} rows, need at least {num_steps}")

    df = df.iloc[:num_steps].copy()
    for start in range(burn_in_days, num_steps, 7):
        end = min(start + 7, num_steps)
        if policy_mode == "factual_weekly_hold":
            school = int(df.loc[df["t"] == start, "school_intervention"].iloc[0])
            occ = int(df.loc[df["t"] == start, "occ_intervention"].iloc[0])
        elif policy_mode == "all_open":
            school, occ = 0, 0
        elif policy_mode == "all_closed":
            school, occ = 1, 1
        else:
            school, occ = int(school_action), int(occ_action)
        mask = (df["t"] >= start) & (df["t"] < end)
        df.loc[mask, "school_intervention"] = school
        df.loc[mask, "occ_intervention"] = occ
    return df


@contextmanager
def temporary_intervention(args: argparse.Namespace, run_root: Path):
    pop_dir = ROOT / "populations" / f"pop{args.county}"
    intervention_path = pop_dir / "intervention.csv"
    backup_path = run_root / "intervention.csv.original"
    effective_path = run_root / "intervention.csv.weekly_effective"
    if not intervention_path.exists():
        raise FileNotFoundError(intervention_path)

    original = pd.read_csv(intervention_path)
    shutil.copyfile(intervention_path, backup_path)
    effective = make_weekly_intervention(
        original,
        burn_in_days=args.burn_in_days,
        num_steps=args.num_steps,
        policy_mode=args.policy_mode,
        school_action=args.school_action,
        occ_action=args.occ_action,
    )
    effective.to_csv(effective_path, index=False)
    effective.to_csv(intervention_path, index=False)
    try:
        yield intervention_path, backup_path, effective_path
    finally:
        shutil.copyfile(backup_path, intervention_path)


def find_param_name(named_parameters: list[tuple[str, torch.nn.Parameter]], suffix: str, fallback_idx: int) -> str:
    matches = [name for name, _ in named_parameters if name.split(".")[-1] == suffix]
    if len(matches) == 1:
        return matches[0]
    if matches:
        return matches[0]
    return named_parameters[fallback_idx][0]


def load_calibrated_params(sim, runner, param_path: Path | None, device: str) -> Path:
    meta = sim.config["simulation_metadata"]
    if param_path is None:
        output_dir = (
            ROOT
            / "result_graphs"
            / str(meta["POPULATION"])
            / str(meta["DATE"])
            / (
                f"{meta['INITIAL_INFECTION_RATE']}_"
                f"{meta['EXPOSED_TO_INFECTED_TIME']}_"
                f"{meta['INFECTED_TO_RECOVERED_TIME']}_"
                f"{meta['WITH_K']}_"
                f"{meta['WITH_VACC']}"
            )
        )
        param_path = output_dir / "calibrated_params.txt"
    if not param_path.exists():
        raise FileNotFoundError(param_path)

    values = np.loadtxt(param_path)
    param_tensor = torch.tensor(values, dtype=torch.float, device=device)
    if param_tensor.ndim == 1:
        param_tensor = param_tensor[:, None]

    num_weeks = int(meta["NUM_WEEKS"])
    named = [(name, param) for name, param in runner.named_parameters()]
    if len(named) < 5:
        raise RuntimeError(f"Expected learnable ABM parameters, found only {len(named)}")

    r2_name = find_param_name(named, "R2", 1)
    infected_name = find_param_name(named, "infected_proportion", 3)
    k_name = find_param_name(named, "k", 4)

    map_and_replace_tensor(r2_name)(runner, True, param_tensor[:num_weeks], mode_calibrate=True)
    map_and_replace_tensor(infected_name)(runner, True, param_tensor[-2], mode_calibrate=True)
    map_and_replace_tensor(k_name)(runner, True, param_tensor[-1], mode_calibrate=True)
    return param_path


def step_global_range(runner, start_day: int, end_day: int) -> None:
    assert runner.state is not None
    for time_step in range(start_day, end_day):
        runner.state["current_step"] = time_step
        runner.state_trajectory.append([])

        for substep in runner.config["substeps"].keys():
            observation_profile, action_profile = {}, {}
            for agent_type in runner.config["substeps"][substep]["active_agents"]:
                assert substep == runner.state["current_substep"]
                assert time_step == runner.state["current_step"]
                observation_profile[agent_type] = runner.controller.observe(
                    runner.state,
                    runner.initializer.observation_function,
                    agent_type,
                )
                action_profile[agent_type] = runner.controller.act(
                    runner.state,
                    observation_profile[agent_type],
                    runner.initializer.policy_function,
                    agent_type,
                )

            next_state = runner.controller.progress(
                runner.state,
                action_profile,
                runner.initializer.transition_function,
            )
            runner.state = next_state
            runner.state_trajectory[-1].append(to_cpu(runner.state))


def run_full_replay(runner, initial_state: dict[str, Any], num_steps: int, seed: int) -> np.ndarray:
    set_seed(seed)
    runner.state = deep_clone(initial_state)
    runner.state_trajectory = []
    runner.step(num_steps)
    return (
        runner.state_trajectory[-1][-1]["environment"]["daily_infected"]
        .detach()
        .cpu()
        .numpy()
        .astype(float)
    )


def run_weekly_rollout(
    runner,
    initial_state: dict[str, Any],
    *,
    num_steps: int,
    burn_in_days: int,
    seed: int,
) -> np.ndarray:
    set_seed(seed)
    runner.state = deep_clone(initial_state)
    runner.state_trajectory = []

    if burn_in_days > 0:
        step_global_range(runner, 0, burn_in_days)

    for start in range(burn_in_days, num_steps, 7):
        step_global_range(runner, start, min(start + 7, num_steps))

    return (
        runner.state_trajectory[-1][-1]["environment"]["daily_infected"]
        .detach()
        .cpu()
        .numpy()
        .astype(float)
    )


def write_comparisons(
    run_root: Path,
    full_daily: np.ndarray,
    weekly_daily: np.ndarray,
    *,
    burn_in_days: int,
) -> dict[str, Any]:
    if full_daily.shape != weekly_daily.shape:
        raise ValueError(f"shape mismatch: full={full_daily.shape}, weekly={weekly_daily.shape}")

    days = np.arange(full_daily.shape[0])
    daily_df = pd.DataFrame(
        {
            "day": days,
            "week": days // 7,
            "phase": np.where(days < burn_in_days, "burn_in", "weekly_rollout"),
            "full_daily_cases": full_daily,
            "weekly_rollout_daily_cases": weekly_daily,
            "abs_diff": np.abs(full_daily - weekly_daily),
        }
    )
    daily_path = run_root / "daily_comparison.csv"
    daily_df.to_csv(daily_path, index=False)

    n_full_weeks = full_daily.shape[0] // 7
    trim = n_full_weeks * 7
    full_weekly = full_daily[:trim].reshape(n_full_weeks, 7).sum(axis=1)
    rollout_weekly = weekly_daily[:trim].reshape(n_full_weeks, 7).sum(axis=1)
    week_ids = np.arange(n_full_weeks)
    weekly_df = pd.DataFrame(
        {
            "week": week_ids,
            "start_day": week_ids * 7,
            "end_day_exclusive": week_ids * 7 + 7,
            "phase": np.where(week_ids * 7 < burn_in_days, "burn_in", "weekly_rollout"),
            "full_weekly_cases": full_weekly,
            "weekly_rollout_cases": rollout_weekly,
            "abs_diff": np.abs(full_weekly - rollout_weekly),
        }
    )
    weekly_path = run_root / "weekly_comparison.csv"
    weekly_df.to_csv(weekly_path, index=False)

    metrics = {
        "daily_max_abs_diff": float(daily_df["abs_diff"].max()),
        "daily_mean_abs_diff": float(daily_df["abs_diff"].mean()),
        "weekly_max_abs_diff": float(weekly_df["abs_diff"].max()),
        "weekly_mean_abs_diff": float(weekly_df["abs_diff"].mean()),
        "num_days": int(full_daily.shape[0]),
        "num_full_weeks": int(n_full_weeks),
        "daily_comparison_csv": str(daily_path),
        "weekly_comparison_csv": str(weekly_path),
    }
    return metrics


def main() -> None:
    args = parse_args()
    os.chdir(ROOT)
    run_root = Path(args.run_root) if args.run_root else (
        ROOT / "online_rollout_runs" / f"{args.county}_weekly_{time.strftime('%Y%m%d_%H%M%S')}"
    )
    run_root.mkdir(parents=True, exist_ok=True)

    config_restored = False
    with temporary_config(args), temporary_intervention(args, run_root) as (_, _, effective_intervention):
        population_module = importlib.import_module(f"populations.pop{args.county}")
        sim = Executor(covid_abm, pop_loader=LoadPopulation(population_module))
        runner = sim._get_runner(sim.config)
        runner.init()
        param_path = load_calibrated_params(sim, runner, None, args.device)
        initial_state = deep_clone(runner.state)

        full_daily = run_full_replay(runner, initial_state, args.num_steps, args.seed)
        weekly_daily = run_weekly_rollout(
            runner,
            initial_state,
            num_steps=args.num_steps,
            burn_in_days=args.burn_in_days,
            seed=args.seed,
        )

        metrics = write_comparisons(
            run_root,
            full_daily,
            weekly_daily,
            burn_in_days=args.burn_in_days,
        )
        summary = {
            "county": args.county,
            "date_tag": args.date_tag,
            "device": args.device,
            "seed": args.seed,
            "num_steps": args.num_steps,
            "num_weeks": args.num_weeks,
            "burn_in_days": args.burn_in_days,
            "policy_mode": args.policy_mode,
            "calibrated_params": str(param_path),
            "effective_intervention_csv": str(effective_intervention),
            **metrics,
        }
        summary_path = run_root / "summary.json"
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True))

        with (run_root / "summary.txt").open("w", newline="") as f:
            writer = csv.writer(f)
            for key, value in summary.items():
                writer.writerow([key, value])

        print(json.dumps(summary, indent=2, sort_keys=True))
        config_restored = True

    if config_restored:
        print("Restored config.yaml and intervention.csv after validation.")


if __name__ == "__main__":
    main()
