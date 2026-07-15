#!/usr/bin/env python3
"""Visualize EpiABM WeightNet reweighting across EM outer checkpoints."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from hydra.utils import instantiate
from matplotlib.lines import Line2D
from omegaconf import OmegaConf
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from scipy.stats import gaussian_kde
from torch.utils.data import DataLoader, Subset

from src.data.ct_transition_dataset import (
    CTEstepDataset,
    _covariate_stream_dim,
    collate_ct_estep_batch,
)
from src.models.ct_encoder_weight import CTEncoderWeightModel
from src.planners.iql_planner import _cap_renormalize_weights
from src.utils.utils import repeat_static, set_seed, to_float


REGIME_COLORS = {
    "open/open": "#3572B0",
    "open/closed": "#009E73",
    "closed/open": "#E69F00",
    "closed/closed": "#CC79A7",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--samples-per-county", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--tsne-seed", type=int, default=10)
    parser.add_argument("--perplexity", type=float, default=40.0)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def select_balanced_indices(
    dataset: CTEstepDataset,
    samples_per_county: int,
    seed: int,
) -> np.ndarray:
    by_row: Dict[int, list[int]] = {}
    for dataset_idx, (row_idx, _day_idx) in enumerate(dataset.index):
        by_row.setdefault(int(row_idx), []).append(dataset_idx)
    rng = np.random.RandomState(seed)
    selected = []
    for row_idx in sorted(by_row):
        candidates = np.asarray(by_row[row_idx], dtype=np.int64)
        count = min(int(samples_per_county), len(candidates))
        chosen = rng.choice(candidates, size=count, replace=False)
        selected.extend(sorted(chosen.tolist()))
    return np.asarray(selected, dtype=np.int64)


def load_dataset_and_config(run_root: Path, seed: int):
    config_path = run_root / "train" / f"seed_{seed}" / "hydra" / ".hydra" / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(config_path)
    cfg = OmegaConf.load(config_path)
    collection = instantiate(cfg.dataset, _recursive_=True)
    collection.process_data_multi()
    collection = to_float(collection)
    if int(cfg.dataset.static_size) > 0 and collection.train_f.data["static_features"].ndim == 2:
        collection = repeat_static(collection)
    return cfg, collection, config_path


@torch.no_grad()
def encode_checkpoint(
    model: CTEncoderWeightModel,
    loader: DataLoader,
    checkpoint_path: Path,
    device: str,
    weight_max: float | None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict_encoder(
        {
            "ct_history_encoder": checkpoint["ct_history_encoder"],
            "projection_head": checkpoint["projection_head"],
            "weight_net": checkpoint["weight_net"],
        }
    )
    model.eval()
    z_parts, action_parts, logit_parts = [], [], []
    for batch in loader:
        history = {key: value.to(device) for key, value in batch["H_t"].items()}
        z_t, action_t = model.encode(history)
        logits = model.weight_net(torch.cat([z_t, action_t], dim=-1))
        z_parts.append(z_t.cpu())
        action_parts.append(action_t.cpu())
        logit_parts.append(logits.reshape(-1).cpu())
    z = torch.cat(z_parts).numpy()
    actions = torch.cat(action_parts).numpy()
    logits = torch.cat(logit_parts)
    weights = torch.softmax(logits, dim=0) * float(logits.numel())
    weights = _cap_renormalize_weights(weights, weight_max).numpy()
    return z, actions, weights, checkpoint


def effective_sample_size_fraction(weights: np.ndarray) -> float:
    weights = np.asarray(weights, dtype=np.float64)
    return float(weights.sum() ** 2 / (len(weights) * np.square(weights).sum() + 1e-12))


def mean_absolute_smd(z: np.ndarray, actions: np.ndarray, weights: np.ndarray) -> float:
    values = []
    eps = 1e-8
    for treatment_idx in range(actions.shape[1]):
        group_one = actions[:, treatment_idx] >= 0.5
        for selected in (group_one,):
            other = ~selected
            if selected.sum() < 2 or other.sum() < 2:
                continue
            w1 = weights[selected].astype(np.float64)
            w0 = weights[other].astype(np.float64)
            w1 /= w1.sum()
            w0 /= w0.sum()
            z1 = z[selected].astype(np.float64)
            z0 = z[other].astype(np.float64)
            mean1 = np.sum(z1 * w1[:, None], axis=0)
            mean0 = np.sum(z0 * w0[:, None], axis=0)
            var1 = np.sum(np.square(z1 - mean1) * w1[:, None], axis=0)
            var0 = np.sum(np.square(z0 - mean0) * w0[:, None], axis=0)
            pooled = np.sqrt(0.5 * (var1 + var0))
            valid = pooled > eps
            values.extend(np.abs((mean1[valid] - mean0[valid]) / pooled[valid]).tolist())
    return float(np.mean(values)) if values else float("nan")


def treatment_regimes(actions: np.ndarray) -> np.ndarray:
    school = actions[:, 0] >= 0.5
    workplace = actions[:, 1] >= 0.5
    labels = np.empty(len(actions), dtype=object)
    labels[(~school) & (~workplace)] = "open/open"
    labels[(~school) & workplace] = "open/closed"
    labels[school & (~workplace)] = "closed/open"
    labels[school & workplace] = "closed/closed"
    return labels.astype(str)


def fit_tsne(z: np.ndarray, seed: int, perplexity: float) -> np.ndarray:
    scaled = StandardScaler().fit_transform(z)
    pca_dim = min(16, scaled.shape[1], scaled.shape[0] - 1)
    reduced = PCA(n_components=pca_dim, random_state=seed).fit_transform(scaled)
    kwargs = dict(
        n_components=2,
        perplexity=min(float(perplexity), max(5.0, (len(z) - 1) / 3.0)),
        init="pca",
        learning_rate="auto",
        random_state=seed,
        method="barnes_hut",
    )
    try:
        return TSNE(max_iter=1500, **kwargs).fit_transform(reduced)
    except TypeError:
        return TSNE(n_iter=1500, **kwargs).fit_transform(reduced)


def scatter_regimes(
    ax: plt.Axes,
    coordinates: np.ndarray,
    regimes: np.ndarray,
    weights: np.ndarray,
    weighted: bool,
) -> None:
    if weighted:
        q95 = max(float(np.quantile(weights, 0.95)), 1e-8)
        scaled_weight = np.clip(weights / q95, 0.05, 1.0)
        sizes = 5.0 + 24.0 * scaled_weight
        alphas = 0.12 + 0.58 * scaled_weight
    else:
        sizes = np.full(len(weights), 8.0)
        alphas = np.full(len(weights), 0.38)
    for regime, color in REGIME_COLORS.items():
        selected = regimes == regime
        if not np.any(selected):
            continue
        rgba = np.tile(np.asarray(plt.matplotlib.colors.to_rgba(color)), (selected.sum(), 1))
        rgba[:, 3] = alphas[selected]
        ax.scatter(
            coordinates[selected, 0],
            coordinates[selected, 1],
            s=sizes[selected],
            c=rgba,
            edgecolors="none",
            rasterized=True,
        )
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color("#C7CDD4")
        spine.set_linewidth(0.7)


def density_level_for_mass(density: np.ndarray, mass: float) -> float:
    flat = np.sort(np.asarray(density, dtype=np.float64).reshape(-1))[::-1]
    total = float(flat.sum())
    if total <= 0.0:
        return float("inf")
    cumulative = np.cumsum(flat) / total
    index = min(int(np.searchsorted(cumulative, mass, side="left")), len(flat) - 1)
    return float(flat[index])


def treatment_density_overlap(
    coordinates: np.ndarray,
    treatment: np.ndarray,
    weights: np.ndarray,
    grid_size: int = 180,
):
    lower = np.percentile(coordinates, 1.0, axis=0)
    upper = np.percentile(coordinates, 99.0, axis=0)
    padding = np.maximum((upper - lower) * 0.06, 1e-3)
    lower -= padding
    upper += padding
    grid_x, grid_y = np.meshgrid(
        np.linspace(lower[0], upper[0], grid_size),
        np.linspace(lower[1], upper[1], grid_size),
    )
    points = np.vstack([grid_x.ravel(), grid_y.ravel()])
    densities = []
    for group_value in (0, 1):
        selected = (treatment >= 0.5) == bool(group_value)
        if selected.sum() < 3:
            raise ValueError(f"Treatment group {group_value} has fewer than three samples.")
        density = gaussian_kde(
            coordinates[selected].T,
            weights=np.asarray(weights[selected], dtype=np.float64),
        )(points).reshape(grid_x.shape)
        density /= max(float(density.sum()), 1e-12)
        densities.append(density)
    overlap_density = np.minimum(densities[0], densities[1])
    overlap_coefficient = float(overlap_density.sum())
    return grid_x, grid_y, densities[0], densities[1], overlap_density, overlap_coefficient


def plot_density_overlap(
    ax: plt.Axes,
    coordinates: np.ndarray,
    treatment: np.ndarray,
    weights: np.ndarray,
    *,
    observed_overlap: float | None = None,
) -> float:
    grid_x, grid_y, density_open, density_closed, overlap, coefficient = treatment_density_overlap(
        coordinates, treatment, weights
    )
    ax.scatter(
        coordinates[:, 0],
        coordinates[:, 1],
        s=2.0,
        c="#B8BEC6",
        alpha=0.10,
        edgecolors="none",
        rasterized=True,
    )
    overlap_level = density_level_for_mass(overlap, 0.80)
    if np.isfinite(overlap_level) and float(overlap.max()) > overlap_level:
        ax.contourf(
            grid_x,
            grid_y,
            overlap,
            levels=[overlap_level, float(overlap.max()) + 1e-12],
            colors=["#55A868"],
            alpha=0.38,
        )
    for density, color in ((density_open, "#3572B0"), (density_closed, "#E69F00")):
        levels = sorted(
            {
                density_level_for_mass(density, 0.80),
                density_level_for_mass(density, 0.50),
            }
        )
        levels = [value for value in levels if np.isfinite(value)]
        if levels:
            ax.contour(grid_x, grid_y, density, levels=levels, colors=[color], linewidths=[1.0, 1.8])
    change = ""
    if observed_overlap is not None:
        change = f"  ({coefficient - observed_overlap:+.2f})"
    ax.text(
        0.03,
        0.96,
        f"Overlap = {coefficient:.2f}{change}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8,
        fontweight="bold",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 2.0},
    )
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color("#C7CDD4")
        spine.set_linewidth(0.7)
    return coefficient


def make_balance_figure(
    output_path: Path,
    outer_one_xy: np.ndarray,
    selected_xy: np.ndarray,
    actions: np.ndarray,
    selected_weights: np.ndarray,
    selected_outer: int,
    selected_observed_smd: float,
    selected_weighted_smd: float,
    selected_ess: float,
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
    fig = plt.figure(figsize=(10.8, 5.35), constrained_layout=True)
    grid = fig.add_gridspec(2, 4, width_ratios=[1.0, 1.0, 1.0, 0.78])
    axes = [[fig.add_subplot(grid[row, col]) for col in range(3)] for row in range(2)]
    metric_grid = grid[:, 3].subgridspec(2, 1, hspace=0.32)
    ax_overlap = fig.add_subplot(metric_grid[0, 0])
    ax_smd = fig.add_subplot(metric_grid[1, 0])

    unit_weights = np.ones(len(actions), dtype=np.float32)
    metrics = []
    row_names = ("School intervention", "Workplace intervention")
    column_specs = (
        ("Outer 1: uniform", outer_one_xy, unit_weights),
        (f"Outer {selected_outer}: observed", selected_xy, unit_weights),
        (f"Outer {selected_outer}: reweighted", selected_xy, selected_weights),
    )
    for treatment_idx, row_name in enumerate(row_names):
        observed_selected_overlap = None
        for column_idx, (column_name, coordinates, weights) in enumerate(column_specs):
            if column_idx == 2:
                observed_selected_overlap = metrics[-1]["overlap"]
            coefficient = plot_density_overlap(
                axes[treatment_idx][column_idx],
                coordinates,
                actions[:, treatment_idx],
                weights,
                observed_overlap=observed_selected_overlap if column_idx == 2 else None,
            )
            metrics.append(
                {
                    "treatment": row_name.split()[0].lower(),
                    "distribution": column_name,
                    "overlap": coefficient,
                }
            )
            if treatment_idx == 0:
                axes[treatment_idx][column_idx].set_title(
                    f"({chr(ord('a') + column_idx)}) {column_name}", loc="left", fontweight="bold"
                )
        axes[treatment_idx][0].set_ylabel(row_name, fontweight="bold", labelpad=7)

    overlap_df = pd.DataFrame(metrics)
    selected_rows = overlap_df[overlap_df["distribution"].str.contains(f"Outer {selected_outer}")]
    treatments = ["school", "workplace"]
    x = np.arange(len(treatments))
    observed_values = [
        float(selected_rows[(selected_rows.treatment == treatment) & selected_rows.distribution.str.contains("observed")].overlap.iloc[0])
        for treatment in treatments
    ]
    weighted_values = [
        float(selected_rows[(selected_rows.treatment == treatment) & selected_rows.distribution.str.contains("reweighted")].overlap.iloc[0])
        for treatment in treatments
    ]
    width = 0.34
    ax_overlap.bar(x - width / 2, observed_values, width, color="#AEB4BC", label="Observed")
    ax_overlap.bar(x + width / 2, weighted_values, width, color="#55A868", label="Reweighted")
    ax_overlap.set_xticks(x, ["School", "Workplace"])
    ax_overlap.set_ylim(0.0, 1.0)
    ax_overlap.set_ylabel("Density overlap")
    ax_overlap.set_title("(d) Direct balance gains", loc="left", fontweight="bold")
    ax_overlap.legend(frameon=False, loc="upper left")
    ax_overlap.grid(axis="y", color="#E4E7EB", lw=0.6)

    smd_values = [selected_observed_smd, selected_weighted_smd]
    bars = ax_smd.bar(
        [0, 1], smd_values, color=["#AEB4BC", "#CC79A7"], width=0.62
    )
    ax_smd.set_xticks([0, 1], ["Observed", "WeightNet\nreweighted"])
    ax_smd.tick_params(axis="x", labelsize=7)
    ax_smd.set_ylabel("Mean absolute SMD")
    ax_smd.set_ylim(0.0, max(smd_values) * 1.32)
    ax_smd.grid(axis="y", color="#E4E7EB", lw=0.6)
    for bar, value in zip(bars, smd_values):
        ax_smd.text(bar.get_x() + bar.get_width() / 2, value + 0.012, f"{value:.3f}", ha="center", va="bottom")
    reduction = 100.0 * (selected_observed_smd - selected_weighted_smd) / selected_observed_smd
    ax_smd.text(
        0.5,
        0.96,
        f"SMD reduction: {reduction:.1f}%\nESS fraction: {selected_ess:.3f}",
        transform=ax_smd.transAxes,
        ha="center",
        va="top",
        fontsize=8,
        bbox={"facecolor": "#F3F5F7", "edgecolor": "#D2D7DD", "pad": 4.0},
    )
    for axis in (ax_overlap, ax_smd):
        axis.spines[["top", "right"]].set_visible(False)

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


def save_sample_manifest(
    path: Path,
    transition_dataset: CTEstepDataset,
    selected_indices: Iterable[int],
    train_data: dict,
) -> None:
    rows = []
    for dataset_idx in selected_indices:
        row_idx, day_idx = transition_dataset.index[int(dataset_idx)]
        rows.append(
            {
                "dataset_index": int(dataset_idx),
                "row_index": int(row_idx),
                "day_index": int(day_idx),
                "county": f"{int(train_data['sim_county_id'][row_idx, day_idx, 0]):05d}",
                "sim_day": int(train_data["sim_day"][row_idx, day_idx, 0]),
                "school_action": float(train_data["current_treatments"][row_idx, day_idx, 0]),
                "workplace_action": float(train_data["current_treatments"][row_idx, day_idx, 1]),
            }
        )
    pd.DataFrame(rows).to_csv(path, index=False)


def make_figure(
    output_path: Path,
    outer_one_xy: np.ndarray,
    selected_xy: np.ndarray,
    regimes: np.ndarray,
    selected_weights: np.ndarray,
    diagnostics: pd.DataFrame,
    selected_outer: int,
) -> None:
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
    fig = plt.figure(figsize=(12.2, 3.25), constrained_layout=True)
    grid = fig.add_gridspec(1, 4, width_ratios=[1.0, 1.0, 1.0, 1.18])
    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[0, 1])
    ax_c = fig.add_subplot(grid[0, 2])
    diag_grid = grid[0, 3].subgridspec(3, 1, hspace=0.12)
    diag_axes = [fig.add_subplot(diag_grid[i, 0]) for i in range(3)]

    unit_weights = np.ones(len(regimes), dtype=np.float32)
    scatter_regimes(ax_a, outer_one_xy, regimes, unit_weights, weighted=False)
    scatter_regimes(ax_b, selected_xy, regimes, unit_weights, weighted=False)
    scatter_regimes(ax_c, selected_xy, regimes, selected_weights, weighted=True)
    ax_a.set_title("(a) Outer 1: uniform M-step", loc="left", fontweight="bold")
    ax_b.set_title(f"(b) Outer {selected_outer}: observed", loc="left", fontweight="bold")
    ax_c.set_title(f"(c) Outer {selected_outer}: reweighted", loc="left", fontweight="bold")

    legend = [
        Line2D([0], [0], marker="o", linestyle="", markersize=5, color=color, label=label)
        for label, color in REGIME_COLORS.items()
        if label in set(regimes)
    ]
    fig.legend(
        handles=legend,
        loc="outside lower center",
        ncol=max(1, len(legend)),
        frameon=False,
        title="School/workplace intervention state",
    )

    x = diagnostics["outer"].to_numpy()
    selected_color = "#D55E00"
    diag_axes[0].plot(x, diagnostics["logged_ess_fraction"], "o-", color="#0072B2", lw=1.3, ms=3)
    diag_axes[0].set_ylabel("ESS fraction")
    diag_axes[0].set_ylim(0.0, 1.05)

    diag_axes[1].plot(x, diagnostics["alignment_pre"], "o--", color="#7A7F87", lw=1.0, ms=2.7, label="pre")
    diag_axes[1].plot(x, diagnostics["alignment_post"], "o-", color="#009E73", lw=1.3, ms=2.7, label="post")
    diag_axes[1].set_ylabel("Sinkhorn")
    diag_axes[1].legend(frameon=False, ncol=2, loc="upper right", handlelength=1.5)

    diag_axes[2].plot(x, diagnostics["balance_unweighted_smd"], "o--", color="#7A7F87", lw=1.0, ms=2.7, label="observed")
    diag_axes[2].plot(x, diagnostics["balance_weighted_smd"], "o-", color="#CC79A7", lw=1.3, ms=2.7, label="weighted")
    diag_axes[2].set_ylabel("Mean |SMD|")
    diag_axes[2].set_xlabel("EM outer iteration")
    diag_axes[2].legend(frameon=False, ncol=2, loc="upper right", handlelength=1.5)

    for axis in diag_axes:
        axis.axvline(selected_outer, color=selected_color, lw=0.9, alpha=0.8)
        axis.grid(axis="y", color="#E4E7EB", lw=0.6)
        axis.spines[["top", "right"]].set_visible(False)
        axis.set_xlim(0.7, diagnostics["outer"].max() + 0.3)
        axis.set_xticks([1, 2, 4, 6, 8, 10, 12])
    diag_axes[0].tick_params(labelbottom=False)
    diag_axes[1].tick_params(labelbottom=False)
    diag_axes[0].set_title("(d) Reweighting diagnostics", loc="left", fontweight="bold")

    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".png"), dpi=320, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    run_root = args.run_root.resolve()
    output_dir = (args.output_dir or run_root / "figures" / f"weightnet_tsne_seed_{args.seed}").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.tsne_seed)

    cfg, collection, config_path = load_dataset_and_config(run_root, args.seed)
    train_data = collection.train_f.data
    transition_dataset = CTEstepDataset(train_data)
    selected_indices = select_balanced_indices(
        transition_dataset, args.samples_per_county, args.tsne_seed
    )
    subset = Subset(transition_dataset, selected_indices.tolist())
    loader = DataLoader(
        subset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_ct_estep_batch,
    )
    save_sample_manifest(
        output_dir / "sample_manifest.csv", transition_dataset, selected_indices, train_data
    )

    checkpoint_dir = run_root / "train" / f"seed_{args.seed}" / "em_ckpt"
    checkpoint_paths = sorted(checkpoint_dir.glob("ct_iql_em_outer*.pt"))
    if len(checkpoint_paths) != 12:
        raise ValueError(f"Expected 12 outer checkpoints, found {len(checkpoint_paths)}")
    selected_path = checkpoint_dir / "selected_best_by_val_target_rmse.pt"
    selected_obj = torch.load(selected_path, map_location="cpu", weights_only=False)
    selected_outer = int(selected_obj["outer_iter"])

    x_dim = _covariate_stream_dim(OmegaConf.to_container(cfg.dataset, resolve=True))
    model = CTEncoderWeightModel(cfg, x_dim).to(args.device)
    diagnostics = []
    embeddings: Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for checkpoint_path in checkpoint_paths:
        checkpoint_header = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        outer = int(checkpoint_header["outer_iter"])
        weight_max = checkpoint_header["iql"]["cfg"].get("weight_max")
        z, actions, weights, checkpoint = encode_checkpoint(
            model, loader, checkpoint_path, args.device, weight_max
        )
        unit_weights = np.ones(len(weights), dtype=np.float32)
        diagnostics.append(
            {
                "outer": outer,
                "alignment_pre": float(checkpoint["e_align_pre"]),
                "alignment_post": float(checkpoint["e_align_post"]),
                "alignment_gain": float(checkpoint["e_align_pre"] - checkpoint["e_align_post"]),
                "logged_ess_fraction": float(checkpoint["e_w_ess_frac"]),
                "sample_ess_fraction": effective_sample_size_fraction(weights),
                "sample_weight_std": float(np.std(weights)),
                "sample_weight_max": float(np.max(weights)),
                "balance_unweighted_smd": mean_absolute_smd(z, actions, unit_weights),
                "balance_weighted_smd": mean_absolute_smd(z, actions, weights),
                "balance_improvement": mean_absolute_smd(z, actions, unit_weights)
                - mean_absolute_smd(z, actions, weights),
            }
        )
        if outer in (1, selected_outer):
            embeddings[outer] = (z, actions, weights)

    diagnostics_df = pd.DataFrame(diagnostics).sort_values("outer")
    diagnostics_df.to_csv(output_dir / "outer_diagnostics.csv", index=False)
    outer_one_z, outer_one_actions, _outer_one_weights = embeddings[1]
    selected_z, selected_actions, selected_weights = embeddings[selected_outer]
    if not np.array_equal(outer_one_actions, selected_actions):
        raise AssertionError("Treatment labels changed across checkpoints for the fixed sample.")

    outer_one_xy = fit_tsne(outer_one_z, args.tsne_seed, args.perplexity)
    selected_xy = fit_tsne(selected_z, args.tsne_seed, args.perplexity)
    regimes = treatment_regimes(selected_actions)
    np.savez_compressed(
        output_dir / "tsne_data.npz",
        selected_indices=selected_indices,
        outer_one_xy=outer_one_xy,
        selected_xy=selected_xy,
        actions=selected_actions,
        regimes=regimes,
        selected_weights=selected_weights,
    )
    make_figure(
        output_dir / "epi_abm_weightnet_tsne_seed10",
        outer_one_xy,
        selected_xy,
        regimes,
        selected_weights,
        diagnostics_df,
        selected_outer,
    )

    selected_row = diagnostics_df.loc[diagnostics_df["outer"] == selected_outer].iloc[0]
    overlap_df = make_balance_figure(
        output_dir / "epi_abm_weightnet_balance_tsne_seed10",
        outer_one_xy,
        selected_xy,
        selected_actions,
        selected_weights,
        selected_outer,
        float(selected_row["balance_unweighted_smd"]),
        float(selected_row["balance_weighted_smd"]),
        float(selected_row["logged_ess_fraction"]),
    )
    overlap_df.to_csv(output_dir / "density_overlap_metrics.csv", index=False)
    summary = {
        "seed": args.seed,
        "selected_outer": selected_outer,
        "split": "train",
        "n_counties": int(train_data["current_treatments"].shape[0]),
        "n_samples": int(len(selected_indices)),
        "samples_per_county": args.samples_per_county,
        "config": str(config_path),
        "weight_max": float(selected_obj["iql"]["cfg"]["weight_max"]),
        "selected_outer_metrics": {
            key: float(selected_row[key])
            for key in (
                "alignment_pre",
                "alignment_post",
                "logged_ess_fraction",
                "sample_ess_fraction",
                "balance_unweighted_smd",
                "balance_weighted_smd",
                "balance_improvement",
            )
        },
        "regime_counts": {
            key: int((regimes == key).sum()) for key in sorted(set(regimes))
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
