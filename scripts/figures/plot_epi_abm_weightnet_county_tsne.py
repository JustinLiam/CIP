#!/usr/bin/env python3
"""Single-county EpiABM representation-balance diagnostic."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.lines import Line2D
from torch.utils.data import DataLoader

from scripts.figures.plot_epi_abm_weightnet_tsne import (
    effective_sample_size_fraction,
    encode_checkpoint,
    fit_tsne,
    load_dataset_and_config,
    mean_absolute_smd,
    plot_density_overlap,
)
from src.data.ct_transition_dataset import (
    CTEstepDataset,
    _covariate_stream_dim,
    collate_ct_estep_batch,
)
from src.models.ct_encoder_weight import CTEncoderWeightModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--county", help="Five-digit county FIPS; default selects a representative county.")
    parser.add_argument("--min-group-size", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--tsne-seed", type=int, default=10)
    parser.add_argument("--perplexity", type=float, default=25.0)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def county_rows(
    transition_dataset: CTEstepDataset,
    train_data: dict,
    z: np.ndarray,
    actions: np.ndarray,
    weights: np.ndarray,
) -> pd.DataFrame:
    row_indices = np.asarray([row for row, _day in transition_dataset.index], dtype=np.int64)
    records = []
    for row_idx in sorted(set(row_indices.tolist())):
        selected = row_indices == row_idx
        county = f"{int(train_data['sim_county_id'][row_idx, 0, 0]):05d}"
        county_actions = actions[selected]
        county_z = z[selected]
        county_weights = weights[selected]
        unit_weights = np.ones(selected.sum(), dtype=np.float32)
        unweighted_smd = mean_absolute_smd(county_z, county_actions, unit_weights)
        weighted_smd = mean_absolute_smd(county_z, county_actions, county_weights)
        records.append(
            {
                "county": county,
                "row_index": int(row_idx),
                "n_transitions": int(selected.sum()),
                "school_open": int((county_actions[:, 0] < 0.5).sum()),
                "school_closed": int((county_actions[:, 0] >= 0.5).sum()),
                "workplace_open": int((county_actions[:, 1] < 0.5).sum()),
                "workplace_closed": int((county_actions[:, 1] >= 0.5).sum()),
                "unweighted_smd": unweighted_smd,
                "weighted_smd": weighted_smd,
                "smd_improvement": unweighted_smd - weighted_smd,
                "ess_fraction": effective_sample_size_fraction(county_weights),
                "weight_std": float(np.std(county_weights)),
                "weight_max": float(np.max(county_weights)),
            }
        )
    return pd.DataFrame(records).sort_values("county").reset_index(drop=True)


def choose_county(metrics: pd.DataFrame, requested: str | None, min_group_size: int):
    if requested is not None:
        county = str(requested).zfill(5)
        selected = metrics[metrics.county == county]
        if selected.empty:
            raise ValueError(f"County {county} is not in the train split.")
        return selected.iloc[0], "explicit"

    support_columns = ["school_open", "school_closed", "workplace_open", "workplace_closed"]
    eligible = metrics[(metrics[support_columns] >= int(min_group_size)).all(axis=1)].copy()
    if eligible.empty:
        raise ValueError(
            f"No train county has at least {min_group_size} observations in all treatment groups."
        )
    median_improvement = float(eligible.smd_improvement.median())
    eligible["distance_to_median_improvement"] = np.abs(
        eligible.smd_improvement - median_improvement
    )
    selected = eligible.sort_values(
        ["distance_to_median_improvement", "county"], ascending=[True, True]
    ).iloc[0]
    return selected, f"eligible-median-improvement (median={median_improvement:.6f})"


def make_figure(
    output_path: Path,
    county: str,
    coordinates: np.ndarray,
    actions: np.ndarray,
    weights: np.ndarray,
    selected_outer: int,
    unweighted_smd: float,
    weighted_smd: float,
    ess_fraction: float,
) -> pd.DataFrame:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "legend.fontsize": 7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig = plt.figure(figsize=(8.4, 5.3), constrained_layout=True)
    grid = fig.add_gridspec(2, 3, width_ratios=[1.0, 1.0, 0.75])
    axes = [[fig.add_subplot(grid[row, col]) for col in range(2)] for row in range(2)]
    metric_grid = grid[:, 2].subgridspec(2, 1, hspace=0.32)
    ax_overlap = fig.add_subplot(metric_grid[0, 0])
    ax_smd = fig.add_subplot(metric_grid[1, 0])

    unit_weights = np.ones(len(actions), dtype=np.float32)
    row_names = ("School intervention", "Workplace intervention")
    overlap_records = []
    for treatment_idx, row_name in enumerate(row_names):
        observed_overlap = plot_density_overlap(
            axes[treatment_idx][0],
            coordinates,
            actions[:, treatment_idx],
            unit_weights,
        )
        weighted_overlap = plot_density_overlap(
            axes[treatment_idx][1],
            coordinates,
            actions[:, treatment_idx],
            weights,
            observed_overlap=observed_overlap,
        )
        overlap_records.extend(
            [
                {"treatment": row_name.split()[0].lower(), "distribution": "observed", "overlap": observed_overlap},
                {"treatment": row_name.split()[0].lower(), "distribution": "reweighted", "overlap": weighted_overlap},
            ]
        )
        axes[treatment_idx][0].set_ylabel(row_name, fontweight="bold", labelpad=7)
    axes[0][0].set_title(f"(a) Outer {selected_outer}: observed", loc="left", fontweight="bold")
    axes[0][1].set_title(f"(b) Outer {selected_outer}: reweighted", loc="left", fontweight="bold")

    overlap_df = pd.DataFrame(overlap_records)
    treatment_names = ["school", "workplace"]
    x = np.arange(2)
    observed_overlap = [
        float(overlap_df[(overlap_df.treatment == name) & (overlap_df.distribution == "observed")].overlap.iloc[0])
        for name in treatment_names
    ]
    weighted_overlap = [
        float(overlap_df[(overlap_df.treatment == name) & (overlap_df.distribution == "reweighted")].overlap.iloc[0])
        for name in treatment_names
    ]
    width = 0.34
    ax_overlap.bar(x - width / 2, observed_overlap, width, color="#AEB4BC", label="Observed")
    ax_overlap.bar(x + width / 2, weighted_overlap, width, color="#55A868", label="Reweighted")
    ax_overlap.set_xticks(x, ["School", "Workplace"])
    ax_overlap.set_ylim(0.0, 1.0)
    ax_overlap.set_ylabel("Density overlap")
    ax_overlap.set_title("(c) Within-county balance", loc="left", fontweight="bold")
    ax_overlap.legend(frameon=False, loc="upper left")
    ax_overlap.grid(axis="y", color="#E4E7EB", lw=0.6)

    bars = ax_smd.bar(
        [0, 1], [unweighted_smd, weighted_smd], color=["#AEB4BC", "#CC79A7"], width=0.62
    )
    ax_smd.set_xticks([0, 1], ["Observed", "WeightNet\nreweighted"])
    ax_smd.tick_params(axis="x", labelsize=7)
    ax_smd.set_ylabel("Mean absolute SMD")
    ax_smd.set_ylim(0.0, max(unweighted_smd, weighted_smd) * 1.65)
    ax_smd.grid(axis="y", color="#E4E7EB", lw=0.6)
    for bar, value in zip(bars, [unweighted_smd, weighted_smd]):
        ax_smd.text(bar.get_x() + bar.get_width() / 2, value + 0.012, f"{value:.3f}", ha="center")
    reduction = 100.0 * (unweighted_smd - weighted_smd) / max(unweighted_smd, 1e-12)
    ax_smd.text(
        0.5,
        0.96,
        f"SMD reduction: {reduction:.1f}%\nESS fraction: {ess_fraction:.3f}",
        transform=ax_smd.transAxes,
        ha="center",
        va="top",
        bbox={"facecolor": "#F3F5F7", "edgecolor": "#D2D7DD", "pad": 4.0},
    )
    for axis in (ax_overlap, ax_smd):
        axis.spines[["top", "right"]].set_visible(False)

    fig.suptitle(
        f"EpiCF-ABM WeightNet representation balance within county {county}",
        fontsize=11,
        fontweight="bold",
    )
    legend_handles = [
        Line2D([0], [0], color="#3572B0", lw=1.8, label="Open density"),
        Line2D([0], [0], color="#E69F00", lw=1.8, label="Closed density"),
        Line2D([0], [0], color="#55A868", lw=7, alpha=0.45, label="Shared density"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="outside lower center",
        ncol=3,
        frameon=False,
        title="Treatment-conditional representation densities",
    )
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".png"), dpi=320, bbox_inches="tight")
    plt.close(fig)
    return overlap_df


def main() -> None:
    args = parse_args()
    run_root = args.run_root.resolve()
    output_dir = (
        args.output_dir
        or run_root / "figures" / f"weightnet_county_tsne_seed_{args.seed}"
    ).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg, collection, config_path = load_dataset_and_config(run_root, args.seed)
    train_data = collection.train_f.data
    transition_dataset = CTEstepDataset(train_data)
    loader = DataLoader(
        transition_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_ct_estep_batch,
    )
    checkpoint_path = (
        run_root
        / "train"
        / f"seed_{args.seed}"
        / "em_ckpt"
        / "selected_best_by_val_target_rmse.pt"
    )
    checkpoint_header = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    selected_outer = int(checkpoint_header["outer_iter"])
    weight_max = checkpoint_header["iql"]["cfg"].get("weight_max")
    x_dim = _covariate_stream_dim(cfg.dataset)
    model = CTEncoderWeightModel(cfg, x_dim).to(args.device)
    z, actions, weights, _checkpoint = encode_checkpoint(
        model, loader, checkpoint_path, args.device, weight_max
    )

    metrics = county_rows(transition_dataset, train_data, z, actions, weights)
    metrics.to_csv(output_dir / "county_selection_metrics.csv", index=False)
    selected_row, selection_rule = choose_county(metrics, args.county, args.min_group_size)
    county = str(selected_row.county).zfill(5)
    row_indices = np.asarray([row for row, _day in transition_dataset.index], dtype=np.int64)
    selected = row_indices == int(selected_row.row_index)
    county_z = z[selected]
    county_actions = actions[selected]
    county_weights = weights[selected]
    coordinates = fit_tsne(county_z, args.tsne_seed, args.perplexity)
    overlap_df = make_figure(
        output_dir / f"epi_abm_weightnet_county_{county}_tsne_seed{args.seed}",
        county,
        coordinates,
        county_actions,
        county_weights,
        selected_outer,
        float(selected_row.unweighted_smd),
        float(selected_row.weighted_smd),
        float(selected_row.ess_fraction),
    )
    overlap_df.to_csv(output_dir / f"county_{county}_density_overlap.csv", index=False)
    np.savez_compressed(
        output_dir / f"county_{county}_tsne_data.npz",
        coordinates=coordinates,
        actions=county_actions,
        weights=county_weights,
        z=county_z,
    )
    summary = {
        "seed": args.seed,
        "selected_outer": selected_outer,
        "county": county,
        "selection_rule": selection_rule,
        "min_group_size": args.min_group_size,
        "config": str(config_path),
        "county_metrics": {
            key: float(selected_row[key])
            for key in (
                "n_transitions",
                "school_open",
                "school_closed",
                "workplace_open",
                "workplace_closed",
                "unweighted_smd",
                "weighted_smd",
                "smd_improvement",
                "ess_fraction",
                "weight_std",
                "weight_max",
            )
        },
        "density_overlap": overlap_df.to_dict(orient="records"),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
