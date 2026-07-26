"""Render the completed efficiency summary as a paper-ready LaTeX table."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


MODELS = (
    ("rmsn", "RMSN"),
    ("crn", "CRN"),
    ("ct", "CT"),
    ("actin", "ACTIN"),
    ("vcip", "VCIP"),
    ("scrl", "SCRL"),
    ("gift", "GIFT"),
    ("cripo", r"\textbf{CRIPO}"),
)


def number(row: dict[str, str], key: str) -> float:
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        return math.nan


def mean_std_five(row: dict[str, str], field: str, digits: int) -> str:
    mean = number(row, f"{field}_mean")
    std = number(row, f"{field}_std")
    count = int(number(row, f"{field}_n")) if math.isfinite(number(row, f"{field}_n")) else 0
    if int(number(row, "n_completed")) != 5 or count != 5:
        return "--"
    return f"${mean:.{digits}f} \\\\pm {std:.{digits}f}$"


def single_seed(row: dict[str, str], field: str, digits: int) -> str:
    value = number(row, f"{field}_mean")
    count = int(number(row, f"{field}_n")) if math.isfinite(number(row, f"{field}_n")) else 0
    if int(number(row, "n_completed")) != 5 or count != 1:
        return "--"
    return f"{value:.{digits}f}"


def params(row: dict[str, str]) -> str:
    value = number(row, "params_deploy_mean")
    count = int(number(row, "params_deploy_n")) if math.isfinite(number(row, "params_deploy_n")) else 0
    if int(number(row, "n_completed")) != 5 or count != 5:
        return "--"
    return f"{int(round(value)):,}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("summary_csv", type=Path)
    parser.add_argument("output_tex", type=Path)
    args = parser.parse_args()

    with args.summary_csv.open(newline="") as stream:
        rows = {
            (row["dataset"], row["model"]): row
            for row in csv.DictReader(stream)
        }

    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Computational efficiency on a single NVIDIA RTX 4090. "
        r"Training time and training-only peak GPU memory are reported as mean $\pm$ "
        r"standard deviation over five training seeds; deployment parameter "
        r"counts are verified to be seed-invariant. Inference latency and "
        r"episode time are reported for the prespecified seed 10. Inference "
        r"times are synchronized per-trajectory averages at batch size 200 "
        r"for Tumor and 100 for MIMIC-III. Latency denotes one "
        r"closed-loop intervention decision; episode time includes the "
        r"complete autoregressive rollout.}",
        r"\label{tab:efficiency}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{llrrrrrr}",
        r"\toprule",
        r"Dataset & Method & Params (deploy) & Train (min/seed) & Latency (ms) "
        r"& Episode (ms, $\tau=6$) & Episode (ms, $\tau=12$) & Peak train (GB) \\",
        r"\midrule",
    ]
    for dataset, label in (("tumor", "Tumor"), ("mimic", "MIMIC-III")):
        for index, (model, model_label) in enumerate(MODELS):
            row = rows.get((dataset, model), {})
            dataset_cell = label if index == 0 else ""
            lines.append(
                " & ".join(
                    (
                        dataset_cell,
                        model_label,
                        params(row),
                        mean_std_five(row, "train_min", 2),
                        single_seed(row, "latency_ms", 2),
                        single_seed(row, "episode_ms_tau6", 2),
                        single_seed(row, "episode_ms_tau12", 2),
                        mean_std_five(row, "peak_gb", 2),
                    )
                )
                + r" \\"
            )
        if dataset == "tumor":
            lines.append(r"\midrule")
    lines.extend(
        (
            r"\bottomrule",
            r"\end{tabular}%",
            r"}",
            r"\end{table*}",
            "",
        )
    )
    args.output_tex.parent.mkdir(parents=True, exist_ok=True)
    args.output_tex.write_text("\n".join(lines))


if __name__ == "__main__":
    main()
