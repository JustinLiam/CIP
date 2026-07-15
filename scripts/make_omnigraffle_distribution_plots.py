#!/usr/bin/env python3
"""Generate vector distribution plots for OmniGraffle insertion.

The PDF and SVG outputs are vector graphics. The PNG files are only previews.
"""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "plots" / "omnigraffle_distribution_vectors"


mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "axes.linewidth": 0.75,
    }
)


def save_panel(points, color, marker, filename, xlim, ylim, marker_size=95):
    fig, ax = plt.subplots(figsize=(3.0, 3.0))
    ax.scatter(
        points[:, 0],
        points[:, 1],
        marker=marker,
        s=marker_size,
        linewidths=0.85,
        color=color,
    )

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal", adjustable="box")
    ax.set_facecolor("white")

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#707070")
        spine.set_linewidth(0.75)

    fig.patch.set_facecolor("white")
    fig.tight_layout(pad=0.05)

    base = OUT_DIR / filename
    fig.savefig(str(base) + ".pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(str(base) + ".svg", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(str(base) + ".png", dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def correlated_points(seed=20260706):
    rng = np.random.RandomState(seed)
    cov = np.array([[1.0, 0.86], [0.86, 1.0]])
    main = rng.multivariate_normal([0.0, 0.0], cov, 230)
    tail = rng.multivariate_normal([1.1, 1.0], cov * 0.55, 36)
    outliers = np.array(
        [
            [-2.65, -2.25],
            [-2.35, -2.75],
            [-2.15, -2.45],
            [2.55, 2.8],
            [3.05, 3.35],
            [2.35, 2.55],
        ]
    )
    return np.vstack([main, tail, outliers])


def diffuse_points(seed=20260707):
    rng = np.random.RandomState(seed)
    center = rng.multivariate_normal(
        [0.0, 0.0], np.array([[1.0, -0.16], [-0.16, 1.1]]), 190
    )
    halo = rng.multivariate_normal(
        [0.0, 0.05], np.array([[2.05, 0.0], [0.0, 1.85]]), 70
    )
    outer = rng.uniform(low=[-2.8, -2.6], high=[2.8, 2.7], size=(18, 2))
    return np.vstack([center, halo, outer])


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    save_panel(
        correlated_points(),
        color="#1E90FF",
        marker="+",
        filename="distribution_correlated_blue_plus",
        xlim=(-3.25, 3.55),
        ylim=(-3.25, 3.55),
    )
    save_panel(
        diffuse_points(),
        color="#FF9900",
        marker="x",
        filename="distribution_diffuse_orange_x",
        xlim=(-3.2, 3.2),
        ylim=(-3.05, 3.05),
    )
    print("Wrote vector plots to {}".format(OUT_DIR))


if __name__ == "__main__":
    main()
