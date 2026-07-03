#!/usr/bin/env python3
"""Prepare Tumor gamma=1/2/3 source CSVs for shift comparison figures."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


METHOD_ORDER = ["rmsn", "crn", "ct", "actin", "vcip", "scrl", "gift", "cripo"]
METHOD_LABELS = {
    "rmsn": "RMSN",
    "crn": "CRN",
    "ct": "CT",
    "actin": "ACTIN",
    "vcip": "VCIP",
    "scrl": "SCRL",
    "gift": "GIFT",
    "cripo": "CRIPO (Ours)",
}
SEEDS = "10 101 1010 10101 101010"
GAMMAS = [1, 2, 3]
TAUS = range(1, 7)


def add_gift_family_rows(rows: list[dict], summary_path: Path, shift: str) -> None:
    summary = pd.read_csv(summary_path)
    summary = summary[(summary["gamma"].isin(GAMMAS)) & (summary["shift"] == shift)]
    for _, rec in summary.iterrows():
        method = str(rec["model"]).lower()
        if method not in METHOD_ORDER or method == "cripo":
            continue
        for tau in TAUS:
            rows.append(
                {
                    "gamma": int(rec["gamma"]),
                    "method": method,
                    "method_label": METHOD_LABELS[method],
                    "tau": tau,
                    "rmse_mean": float(rec[f"tau_{tau}_mean"]),
                    "rmse_std": float(rec[f"tau_{tau}_std"]),
                    "n_seeds": 5,
                    "seeds": SEEDS,
                    "source_metric_file": str(summary_path),
                }
            )


def add_cripo_gamma1_rows(rows: list[dict], stats_path: Path, split: str) -> None:
    stats = pd.read_csv(stats_path)
    stats = stats[(stats["gamma"] == 1) & (stats["split"] == split)]
    for _, rec in stats.iterrows():
        tau = int(rec["tau"])
        if tau not in TAUS:
            continue
        rows.append(
            {
                "gamma": 1,
                "method": "cripo",
                "method_label": METHOD_LABELS["cripo"],
                "tau": tau,
                "rmse_mean": float(rec["mean_rmse"]),
                "rmse_std": float(rec["std_rmse"]),
                "n_seeds": int(rec["n"]),
                "seeds": SEEDS,
                "source_metric_file": str(stats_path),
            }
        )


def add_cripo_gamma23_shift_false_rows(rows: list[dict], source_path: Path) -> None:
    source = pd.read_csv(source_path)
    source = source[(source["method"] == "cripo") & (source["gamma"].isin([2, 3]))]
    source = source[source["tau"].isin(list(TAUS))]
    for _, rec in source.iterrows():
        rows.append(
            {
                "gamma": int(rec["gamma"]),
                "method": "cripo",
                "method_label": METHOD_LABELS["cripo"],
                "tau": int(rec["tau"]),
                "rmse_mean": float(rec["rmse_mean"]),
                "rmse_std": float(rec["rmse_std"]),
                "n_seeds": int(rec.get("n_seeds", 5)),
                "seeds": rec.get("seeds", SEEDS),
                "source_metric_file": str(source_path),
            }
        )


def add_cripo_gamma23_shift_true_rows(rows: list[dict], gamma_to_summary: dict[int, Path]) -> None:
    for gamma, summary_path in gamma_to_summary.items():
        summary = pd.read_csv(summary_path)
        summary = summary[(summary["split"] == "test") & (summary["eval_tau"].isin(list(TAUS)))]
        for tau, group in summary.groupby("eval_tau", sort=True):
            values = group["rmse_uns"].astype(float).to_numpy()
            seeds = " ".join(str(seed) for seed in sorted(group["seed"].astype(int).unique()))
            rows.append(
                {
                    "gamma": gamma,
                    "method": "cripo",
                    "method_label": METHOD_LABELS["cripo"],
                    "tau": int(tau),
                    "rmse_mean": float(np.mean(values)),
                    "rmse_std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                    "n_seeds": int(len(values)),
                    "seeds": seeds,
                    "source_metric_file": str(summary_path),
                }
            )


def write_source(rows: list[dict], output_path: Path) -> None:
    source = pd.DataFrame(rows)
    source["method"] = pd.Categorical(source["method"], METHOD_ORDER, ordered=True)
    source = source.sort_values(["gamma", "method", "tau"]).reset_index(drop=True)

    expected = {(gamma, method, tau) for gamma in GAMMAS for method in METHOD_ORDER for tau in TAUS}
    observed = {(int(row.gamma), str(row.method), int(row.tau)) for row in source.itertuples()}
    missing = sorted(expected - observed)
    if missing:
        raise ValueError(f"Missing source rows: {missing}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    source.to_csv(output_path, index=False)
    print(f"source={output_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gift-family-summary",
        type=Path,
        default=Path("external_repos/GIFT/tumor_tau12_resume_scrl_dualgpu_20260703_021354/tumor_tau12_summary.csv"),
    )
    parser.add_argument(
        "--cripo-gamma1-stats",
        type=Path,
        default=Path("grid_results/cripo_noise001_seq60_dseed10_gamma1_20260702_205358/gamma1_val_test_tau1_6_stats.csv"),
    )
    parser.add_argument(
        "--cripo-gamma23-shift-false-source",
        type=Path,
        default=Path("results/figures/tumor_gamma_shift_false/tumor_episode_rmse_tau1_6_source.csv"),
    )
    parser.add_argument(
        "--cripo-gamma2-shift-true-summary",
        type=Path,
        default=Path("grid_results/noise0_seq60_fixed_dseed10_gamma23_20260627_214813/gamma2/summary.csv"),
    )
    parser.add_argument(
        "--cripo-gamma3-shift-true-summary",
        type=Path,
        default=Path("grid_results/noise0_seq60_fixed_dseed10_gamma23_20260627_214813/gamma3/summary.csv"),
    )
    parser.add_argument(
        "--shift-false-output",
        type=Path,
        default=Path("results/figures/tumor_gamma_123_shift_false/tumor_gamma_123_shift_false_tau1_6_rmse_source.csv"),
    )
    parser.add_argument(
        "--shift-true-output",
        type=Path,
        default=Path("results/figures/tumor_gamma_123_shift_true/tumor_gamma_123_shift_true_tau1_6_rmse_source.csv"),
    )
    args = parser.parse_args()

    shift_false_rows: list[dict] = []
    add_gift_family_rows(shift_false_rows, args.gift_family_summary, "shift_False")
    add_cripo_gamma1_rows(shift_false_rows, args.cripo_gamma1_stats, "val")
    add_cripo_gamma23_shift_false_rows(shift_false_rows, args.cripo_gamma23_shift_false_source)
    write_source(shift_false_rows, args.shift_false_output)

    shift_true_rows: list[dict] = []
    add_gift_family_rows(shift_true_rows, args.gift_family_summary, "shift_True")
    add_cripo_gamma1_rows(shift_true_rows, args.cripo_gamma1_stats, "test")
    add_cripo_gamma23_shift_true_rows(
        shift_true_rows,
        {
            2: args.cripo_gamma2_shift_true_summary,
            3: args.cripo_gamma3_shift_true_summary,
        },
    )
    write_source(shift_true_rows, args.shift_true_output)


if __name__ == "__main__":
    main()
