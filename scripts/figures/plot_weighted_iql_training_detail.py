#!/usr/bin/env python
"""Draw a detailed weighted-IQL training panel matching the current code path."""

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch


COLORS = {
    "ink": "#1F2933",
    "muted": "#5E6978",
    "line": "#B9C0CA",
    "data_fc": "#F7FAFC",
    "data_ec": "#77869A",
    "state_fc": "#EAF3FF",
    "state_ec": "#2F75B5",
    "weight_fc": "#EAF7EF",
    "weight_ec": "#31915B",
    "value_fc": "#F2ECFF",
    "value_ec": "#7657B8",
    "q_fc": "#FFF7E8",
    "q_ec": "#D89328",
    "actor_fc": "#FFEDEA",
    "actor_ec": "#D05B45",
}


def configure_matplotlib():
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "mathtext.fontset": "stix",
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 8,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def box(ax, xy, w, h, text="", fc="white", ec="#222", lw=1.2, radius=0.018,
        size=8, weight="regular", color=None, zorder=2):
    patch = FancyBboxPatch(
        xy,
        w,
        h,
        boxstyle="round,pad=0.006,rounding_size={}".format(radius),
        linewidth=lw,
        edgecolor=ec,
        facecolor=fc,
        zorder=zorder,
    )
    ax.add_patch(patch)
    if text:
        ax.text(
            xy[0] + w / 2,
            xy[1] + h / 2,
            text,
            ha="center",
            va="center",
            fontsize=size,
            fontweight=weight,
            color=color or COLORS["ink"],
            linespacing=1.15,
            zorder=zorder + 1,
        )
    return patch


def label(ax, x, y, text, size=8, weight="regular", color=None,
          ha="center", va="center", zorder=10):
    ax.text(
        x,
        y,
        text,
        fontsize=size,
        fontweight=weight,
        color=color or COLORS["ink"],
        ha=ha,
        va=va,
        linespacing=1.15,
        zorder=zorder,
    )


def arrow(ax, start, end, color=None, lw=1.2, style="-|>", ms=10,
          rad=0.0, ls="-", zorder=5):
    patch = FancyArrowPatch(
        start,
        end,
        arrowstyle=style,
        mutation_scale=ms,
        linewidth=lw,
        color=color or COLORS["ink"],
        linestyle=ls,
        connectionstyle="arc3,rad={}".format(rad),
        shrinkA=2,
        shrinkB=2,
        zorder=zorder,
    )
    ax.add_patch(patch)
    return patch


def dot(ax, x, y, text, fc, ec, size=8, r=0.026):
    circ = Circle((x, y), r, facecolor=fc, edgecolor=ec, linewidth=1.2, zorder=6)
    ax.add_patch(circ)
    label(ax, x, y, text, size=size, weight="bold", zorder=7)
    return circ


def draw_replay_inputs(ax):
    x, y, w, h = 0.045, 0.170, 0.205, 0.670
    box(ax, (x, y), w, h, fc=COLORS["data_fc"], ec=COLORS["data_ec"], lw=1.2)
    label(ax, x + 0.020, y + h - 0.045, "1", size=7, color="white", weight="bold")
    ax.add_patch(Circle((x + 0.020, y + h - 0.045), 0.015, facecolor=COLORS["data_ec"],
                        edgecolor=COLORS["data_ec"], zorder=3))
    label(ax, x + 0.048, y + h - 0.045, "Replay transition", size=9.2,
          weight="bold", ha="left")
    label(ax, x + 0.048, y + h - 0.080, "observed data used in the M-step",
          size=6.8, color=COLORS["muted"], ha="left")

    dot(ax, x + 0.055, y + 0.465, r"$s_t$", "#FFFFFF", COLORS["state_ec"], size=9)
    dot(ax, x + 0.148, y + 0.465, r"$A_t^{obs}$", "#FFFFFF", COLORS["data_ec"], size=8.4)
    dot(ax, x + 0.055, y + 0.340, r"$s_{t+1}$", "#FFFFFF", COLORS["state_ec"], size=8.3)
    dot(ax, x + 0.148, y + 0.340, r"$r_t,d_t$", "#FFFFFF", COLORS["data_ec"], size=8.0)

    box(ax, (x + 0.030, y + 0.150), 0.145, 0.070,
        r"WeightNet frozen" "\n" r"$\omega(Z_t,A_t^{obs})$",
        fc="#FFFFFF", ec=COLORS["weight_ec"], lw=1.0, size=7.0)
    dot(ax, x + 0.103, y + 0.065, r"$w_t$", "#FFFFFF", COLORS["weight_ec"], size=9, r=0.027)
    arrow(ax, (x + 0.103, y + 0.150), (x + 0.103, y + 0.093),
          color=COLORS["weight_ec"], lw=1.2)
    return {
        "s": (x + 0.055, y + 0.465),
        "a": (x + 0.148, y + 0.465),
        "s_next": (x + 0.055, y + 0.340),
        "rd": (x + 0.148, y + 0.340),
        "w": (x + 0.103, y + 0.065),
    }


def draw_training_row(ax, yy, color_fc, color_ec, title, calc_text, loss_text, note):
    step = box(ax, (0.310, yy), 0.105, 0.105, title, fc=color_fc, ec=color_ec,
               lw=1.2, size=8.2, weight="bold")
    calc = box(ax, (0.455, yy), 0.290, 0.105, calc_text, fc="#FFFFFF", ec=color_ec,
               lw=1.1, size=7.1)
    loss = box(ax, (0.785, yy), 0.145, 0.105, loss_text, fc="#FFFFFF", ec=color_ec,
               lw=1.1, size=7.0, weight="bold")
    arrow(ax, (0.415, yy + 0.052), (0.455, yy + 0.052), color=color_ec, lw=1.2)
    arrow(ax, (0.745, yy + 0.052), (0.785, yy + 0.052), color=color_ec, lw=1.2)
    label(ax, 0.455, yy - 0.030, note, size=6.2, color=COLORS["muted"], ha="left")
    return {
        "step_left": (0.310, yy + 0.052),
        "step_mid": (0.363, yy + 0.052),
        "calc_mid": (0.600, yy + 0.052),
        "loss_mid": (0.858, yy + 0.052),
        "loss_left": (0.785, yy + 0.052),
    }


def draw_panel(out_dir):
    configure_matplotlib()
    fig = plt.figure(figsize=(11.5, 5.2), dpi=300)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    label(ax, 0.50, 0.940, "Weighted IQL Training in the M-step", size=13,
          weight="bold")
    label(
        ax,
        0.50,
        0.905,
        "The current code uses Twin-Q critics, frozen WeightNet sample weights, and an AWR actor update.",
        size=7.8,
        color=COLORS["muted"],
    )

    inputs = draw_replay_inputs(ax)

    box(ax, (0.285, 0.170), 0.675, 0.670, fc="#FFFFFF", ec="#9CA3AF", lw=1.1)
    ax.add_patch(Circle((0.315, 0.795), 0.016, facecolor="#596579",
                        edgecolor="#596579", zorder=3))
    label(ax, 0.315, 0.795, "2", size=7, color="white", weight="bold")
    label(ax, 0.348, 0.795, "Sequential weighted optimization", size=9.2,
          weight="bold", ha="left")
    label(ax, 0.348, 0.760,
          r"V and actor use detached states; the encoder is updated only through the weighted Twin-Q loss.",
          size=6.8, color=COLORS["muted"], ha="left")

    value = draw_training_row(
        ax,
        0.615,
        COLORS["value_fc"],
        COLORS["value_ec"],
        "V-step\nupdate",
        r"$Q_1,Q_2=Q_\theta(s_t,A_t^{obs})$" "\n"
        r"$u_t=\min(Q_1,Q_2)-V_\psi(s_t)$",
        r"$L_V^w=\mathbb{E}_w$" "\n"
        r"$[|\tau-\mathbf{1}_{u_t<0}|u_t^2]$",
        r"updates $V_\psi$ only",
    )
    q_update = draw_training_row(
        ax,
        0.405,
        COLORS["q_fc"],
        COLORS["q_ec"],
        "Twin-Q\n+ Encoder",
        r"$y_t=r_t+\gamma(1-d_t)V_\psi(s_{t+1})$" "\n"
        r"$Q_j=Q_{\theta_j}(s_t,A_t^{obs}),\ j=1,2$",
        r"$L_Q^w=\frac{1}{2}\sum_j$" "\n"
        r"$\mathbb{E}_w[(Q_j-y_t)^2]$",
        r"updates $Q_{\theta_1},Q_{\theta_2}$ and the history encoder",
    )
    actor = draw_training_row(
        ax,
        0.215,
        COLORS["actor_fc"],
        COLORS["actor_ec"],
        "Actor\nAWR",
        r"$g_t=\mathrm{stopgrad}(\exp(\beta u_t))$" "\n"
        r"$\ell_{BC}(\pi_\phi(s_t),A_t^{obs})$",
        r"$L_\pi^w=\mathbb{E}_w$" "\n"
        r"$[g_t\,\ell_{BC}]$",
        r"updates $\pi_\phi$ only; default $\ell_{BC}$ is expectile behavior cloning",
    )

    arrow(ax, (0.250, 0.515), (0.310, 0.515), color=COLORS["line"], lw=1.0)
    label(ax, 0.280, 0.545, "M-step batch", size=6.2,
          color=COLORS["muted"])

    label(ax, 0.920, 0.735,
          r"Each loss uses the same capped and renormalized $w_t$ through $\mathbb{E}_w[\cdot]$.",
          size=5.8, color=COLORS["weight_ec"], ha="right")

    label(ax, 0.945, 0.102,
          r"Note: $q_{target}$ is soft-updated and saved, but the current TD target is $r_t+\gamma V(s_{t+1})$.",
          size=5.8, color=COLORS["muted"], ha="right")

    out_dir.mkdir(parents=True, exist_ok=True)
    base = out_dir / "weighted_iql_training_detail"
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.02)
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.02)
    fig.savefig(base.with_suffix(".png"), bbox_inches="tight", pad_inches=0.02, dpi=300)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=Path("plots/method_framework"))
    args = parser.parse_args()
    draw_panel(args.out_dir)


if __name__ == "__main__":
    main()
