#!/usr/bin/env python3
"""Create a manuscript-style RMSE figure for tumor gamma comparisons."""

import argparse
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt


METHOD_ORDER = ["rmsn", "crn", "ct", "actin", "vcip", "scrl", "gift", "cripo"]
METHOD_LABELS = {
    "rmsn": "RMSN",
    "crn": "CRN",
    "ct": "CT",
    "actin": "ACTIN",
    "vcip": "VCIP",
    "scrl": "SCRL",
    "gift": "GIFT",
    "cripo": "CRIPO",
}

COLORS = {
    "rmsn": "#A5A5A5",
    "crn": "#AB47BC",
    "ct": "#7CB342",
    "actin": "#FB8C00",
    "vcip": "#1E88E5",
    "scrl": "#26A69A",
    "gift": "#FF7043",
    "cripo": "#D81B60",
}

LINEWIDTH = {}

LINESTYLE = {
    "rmsn": "-",
    "crn": (0, (3.7, 1.6)),
    "ct": (0, (6.4, 1.6, 1.0, 1.6)),
    "actin": (0, (5.0, 10.0, 3.0)),
    "vcip": (0, (5.0, 10.0)),
    "scrl": (0, (5.0, 1.0)),
    "gift": (0, (1.0, 1.65)),
    "cripo": "-",
}

MARKER = {
    "rmsn": "v",
    "crn": "s",
    "ct": "^",
    "actin": "D",
    "vcip": "X",
    "scrl": "P",
    "gift": "p",
    "cripo": "o",
}


def apply_publication_style() -> None:
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Linux Libertine O", "Libertinus Serif", "Times New Roman", "DejaVu Serif"]
    plt.rcParams["svg.fonttype"] = "none"
    mpl.rcParams.update(
        {
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.size": 7.0,
            "mathtext.fontset": "custom",
            "mathtext.rm": "Linux Libertine O",
            "mathtext.it": "Linux Libertine O:italic",
            "mathtext.bf": "Linux Libertine O:bold",
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.major.size": 3.0,
            "ytick.major.size": 3.0,
            "legend.frameon": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def make_long_source(summary: pd.DataFrame, max_tau: int = 6) -> pd.DataFrame:
    rows = []
    for _, rec in summary.iterrows():
        gamma = int(rec["gamma"])
        method = str(rec["model"])
        if method not in METHOD_ORDER:
            continue
        for tau in range(1, max_tau + 1):
            rows.append(
                {
                    "gamma": gamma,
                    "method": method,
                    "method_label": METHOD_LABELS[method],
                    "tau": tau,
                    "rmse_mean": float(rec[f"tau_{tau}_mean"]),
                    "rmse_std": float(rec[f"tau_{tau}_std"]),
                    "n_seeds": int(rec["n_seeds"]),
                    "seeds": rec.get("seeds", ""),
                }
            )
    source = pd.DataFrame(rows)
    source["method"] = pd.Categorical(source["method"], METHOD_ORDER, ordered=True)
    return source.sort_values(["gamma", "method", "tau"]).reset_index(drop=True)


def load_source(path: Path, max_tau: int = 6) -> pd.DataFrame:
    table = pd.read_csv(path)
    long_columns = {"gamma", "method", "tau", "rmse_mean", "rmse_std"}
    if long_columns.issubset(table.columns):
        source = table.copy()
        source = source[source["method"].isin(METHOD_ORDER)]
        source = source[source["tau"].between(1, max_tau)]
        source["method_label"] = source["method"].map(METHOD_LABELS)
        source["method"] = pd.Categorical(source["method"], METHOD_ORDER, ordered=True)
        return source.sort_values(["gamma", "method", "tau"]).reset_index(drop=True)
    return make_long_source(table, max_tau=max_tau)


def draw_figure(
    source: pd.DataFrame,
    output_prefix: Path,
    gammas: Optional[List[int]] = None,
    method_labels: Optional[Dict[str, str]] = None,
) -> None:
    apply_publication_style()

    gammas = [2, 3, 4] if gammas is None else gammas
    if len(gammas) != 3:
        raise ValueError("This manuscript layout expects exactly three gamma panels.")
    method_labels = METHOD_LABELS if method_labels is None else method_labels

    width_in = 501.37985 / 72.0
    height_in = 147.564378 / 72.0
    fig, axes = plt.subplots(1, 3, figsize=(width_in, height_in), sharex=True)
    legend_handles = []
    legend_labels = []

    for idx, (ax, gamma) in enumerate(zip(axes, gammas)):
        panel = source[source["gamma"] == gamma]
        y_upper = 0.0
        for method in METHOD_ORDER:
            dat = panel[panel["method"] == method]
            if dat.empty:
                continue
            tau = dat["tau"].to_numpy(dtype=float)
            mean = dat["rmse_mean"].to_numpy(dtype=float)
            std = dat["rmse_std"].to_numpy(dtype=float)
            color = COLORS[method]
            lw = LINEWIDTH.get(method, 1.35)
            alpha_band = 0.11
            z = 6 if method == "cripo" else 5 if method == "gift" else 4 if method == "vcip" else 2
            lower = np.maximum(mean - std, 0)
            upper = mean + std
            y_upper = max(y_upper, float(np.max(upper)))
            ax.fill_between(tau, lower, upper, color=color, alpha=alpha_band, linewidth=0, zorder=z - 1)
            (line,) = ax.plot(
                tau,
                mean,
                color=color,
                linewidth=lw,
                linestyle=LINESTYLE[method],
                marker=MARKER[method],
                markersize=3.3,
                markerfacecolor=color,
                markeredgecolor=color,
                markeredgewidth=0.9,
                label=method_labels[method],
                zorder=z,
            )
            if idx == 0:
                legend_handles.append(line)
                legend_labels.append(method_labels[method])

        ax.set_xlim(0.75, 6.25)
        ax.set_xticks(np.arange(1, 7))
        ax.set_xlabel(r"Horizon $\tau$", labelpad=1.5, fontsize=8)
        ax.set_ylim(0, y_upper * 1.10)
        ax.grid(True, which="major", axis="both", color="#B0B0B0", linestyle="--", linewidth=0.58, alpha=0.36)
        if idx == 0:
            ax.set_ylabel("Normalized RMSE", labelpad=2, fontsize=8)
        else:
            ax.set_ylabel("")
        ax.text(
            0.5,
            -0.32,
            rf"({chr(ord('a') + idx)}) Tumor dataset ($\kappa={gamma}$)",
            transform=ax.transAxes,
            fontsize=8,
            va="top",
            ha="center",
        )

    fig.legend(
        legend_handles,
        legend_labels,
        loc="upper center",
        ncol=len(legend_handles),
        bbox_to_anchor=(0.5, 1.015),
        fontsize=6.7,
        handlelength=1.1,
        handletextpad=0.3,
        columnspacing=0.7,
        borderaxespad=0.0,
        frameon=False,
    )

    fig.subplots_adjust(
        left=25.87025 / 501.37985,
        right=500.51585 / 501.37985,
        top=(147.564378 - 19.8916) / 147.564378,
        bottom=(147.564378 - 111.0004) / 147.564378,
        wspace=0.28,
    )

    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_prefix.with_suffix(".svg"))
    fig.savefig(output_prefix.with_suffix(".pdf"))
    fig.savefig(output_prefix.with_suffix(".tiff"), dpi=600)
    fig.savefig(output_prefix.with_suffix(".png"), dpi=600)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("results/figures/tumor_gamma_shift_false/tumor_rmse_tau1_6_source.csv"),
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=Path("results/figures/tumor_gamma_shift_false/tumor_gamma_shift_false_tau1_6_rmse"),
    )
    parser.add_argument(
        "--gammas",
        default="2,3,4",
        help="Comma-separated gamma/kappa values for the three panels.",
    )
    parser.add_argument(
        "--cripo-label",
        default="CRIPO",
        help="Legend label for the CRIPO method.",
    )
    args = parser.parse_args()

    gammas = [int(item.strip()) for item in args.gammas.split(",") if item.strip()]
    method_labels = dict(METHOD_LABELS)
    method_labels["cripo"] = args.cripo_label
    source = load_source(args.summary, max_tau=6)
    source = source[source["gamma"].isin(gammas)]
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    source_path = args.output_prefix.with_name(args.output_prefix.name + "_source.csv")
    source.to_csv(source_path, index=False)
    draw_figure(source, args.output_prefix, gammas=gammas, method_labels=method_labels)
    print(f"source={source_path}")
    print(f"figure={args.output_prefix}")


if __name__ == "__main__":
    main()
