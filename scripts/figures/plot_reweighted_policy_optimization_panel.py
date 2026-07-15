#!/usr/bin/env python
"""Draw the reweighted target-conditioned policy optimization panel.

The figure is intended as the third panel of the method framework. It
separates training-time observed treatments from inference-time planned
actions and shows reweighting as a loss-level operation, not as an
inference input.

Example:
    python scripts/figures/plot_reweighted_policy_optimization_panel.py \
        --out-dir plots/method_framework
"""

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle


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
    "loss_fc": "#FFF7E8",
    "loss_ec": "#D89328",
    "net_fc": "#F2ECFF",
    "net_ec": "#7657B8",
    "actor_fc": "#FFEDEA",
    "actor_ec": "#D05B45",
    "rollout_fc": "#EEF8F6",
    "rollout_ec": "#31877D",
}


def configure_matplotlib() -> None:
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


def box(ax, xy, w, h, text="", fc="white", ec="#222", lw=1.2, radius=0.02, size=8,
        weight="regular", color=None, zorder=2):
    patch = FancyBboxPatch(
        xy,
        w,
        h,
        boxstyle=f"round,pad=0.006,rounding_size={radius}",
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
            linespacing=1.12,
            zorder=zorder + 1,
        )
    return patch


def arrow(ax, start, end, color=None, lw=1.2, style="-|>", ms=10, rad=0.0,
          ls="-", zorder=5):
    patch = FancyArrowPatch(
        start,
        end,
        arrowstyle=style,
        mutation_scale=ms,
        linewidth=lw,
        color=color or COLORS["ink"],
        linestyle=ls,
        connectionstyle=f"arc3,rad={rad}",
        shrinkA=2,
        shrinkB=2,
        zorder=zorder,
    )
    ax.add_patch(patch)
    return patch


def label(ax, x, y, text, size=8, weight="regular", color=None, ha="center", va="center",
          rotation=0, zorder=10):
    ax.text(
        x,
        y,
        text,
        fontsize=size,
        fontweight=weight,
        color=color or COLORS["ink"],
        ha=ha,
        va=va,
        rotation=rotation,
        linespacing=1.12,
        zorder=zorder,
    )


def dot(ax, x, y, text, fc, ec, size=8, r=0.026):
    circ = Circle((x, y), r, facecolor=fc, edgecolor=ec, linewidth=1.2, zorder=6)
    ax.add_patch(circ)
    label(ax, x, y, text, size=size, weight="bold", color=COLORS["ink"], zorder=7)
    return circ


def chip(ax, x, y, text, fc, ec, w=0.078, h=0.035, size=7.5):
    return box(ax, (x - w / 2, y - h / 2), w, h, text, fc=fc, ec=ec, lw=1.0,
               radius=0.012, size=size, weight="bold")


def draw_training_sample(ax):
    x, y, w, h = 0.035, 0.555, 0.230, 0.335
    box(ax, (x, y), w, h, fc=COLORS["data_fc"], ec=COLORS["data_ec"], lw=1.2)
    ax.add_patch(Circle((x + 0.016, y + h - 0.026), 0.014, facecolor=COLORS["data_ec"],
                        edgecolor=COLORS["data_ec"], zorder=3))
    label(ax, x + 0.016, y + h - 0.026, "1", size=7, weight="bold", color="white")
    label(ax, x + 0.050, y + h - 0.026, "Offline transition", size=8.2,
          weight="bold", ha="left")
    label(ax, x + 0.050, y + h - 0.053, "observed trajectory, training only",
          size=6.4, color=COLORS["muted"], ha="left")

    y0 = y + 0.230
    dot(ax, x + 0.047, y0, r"$H_t$", "#FFFFFF", COLORS["data_ec"], size=8.5)
    dot(ax, x + 0.120, y0, r"$A_t^{obs}$", "#FFFFFF", COLORS["data_ec"], size=8.2)
    dot(ax, x + 0.193, y0, r"$H_{t+1}$", "#FFFFFF", COLORS["data_ec"], size=8.0)
    arrow(ax, (x + 0.073, y0), (x + 0.094, y0), color=COLORS["data_ec"], lw=1.3)
    arrow(ax, (x + 0.146, y0), (x + 0.167, y0), color=COLORS["data_ec"], lw=1.3)

    box(ax, (x + 0.030, y + 0.116), 0.082, 0.060, r"target" "\n" r"$Y^\star$",
        fc="#FFFFFF", ec=COLORS["data_ec"], lw=1.0, size=7.3)
    box(ax, (x + 0.126, y + 0.116), 0.082, 0.060, r"horizon" "\n" r"$\Delta_t$",
        fc="#FFFFFF", ec=COLORS["data_ec"], lw=1.0, size=7.3)
    box(ax, (x + 0.030, y + 0.036), 0.178, 0.050,
        r"reward from target progress:  $r_t$", fc="#FFFFFF", ec=COLORS["data_ec"],
        lw=1.0, size=6.9)
    return {
        "H_t": (x + 0.047, y0),
        "A_obs": (x + 0.120, y0),
        "H_next": (x + 0.193, y0),
        "target": (x + 0.071, y + 0.146),
        "delta": (x + 0.167, y + 0.146),
        "reward": (x + 0.119, y + 0.061),
    }


def draw_state_builder(ax, anchors):
    x, y, w, h = 0.315, 0.555, 0.210, 0.335
    box(ax, (x, y), w, h, fc=COLORS["state_fc"], ec=COLORS["state_ec"], lw=1.2)
    ax.add_patch(Circle((x + 0.016, y + h - 0.026), 0.014, facecolor=COLORS["state_ec"],
                        edgecolor=COLORS["state_ec"], zorder=3))
    label(ax, x + 0.016, y + h - 0.026, "2", size=7, weight="bold", color="white")
    label(ax, x + 0.050, y + h - 0.026, "Target-conditioned state", size=8.2,
          weight="bold", ha="left")
    label(ax, x + 0.050, y + h - 0.053, "history + target + horizon",
          size=6.4, color=COLORS["muted"], ha="left")

    box(ax, (x + 0.030, y + 0.214), 0.072, 0.052, "Encoder", fc="#FFFFFF",
        ec=COLORS["state_ec"], lw=1.0, size=7.4, weight="bold")
    dot(ax, x + 0.154, y + 0.240, r"$Z_t$", "#FFFFFF", COLORS["state_ec"], size=8.5, r=0.024)
    arrow(ax, (x + 0.103, y + 0.240), (x + 0.128, y + 0.240), color=COLORS["state_ec"], lw=1.3)

    box(ax, (x + 0.030, y + 0.116), 0.150, 0.064,
        r"$s_t=[Z_t,Y^\star,\Delta_t,A_{t-1}^{obs}]$",
        fc="#FFFFFF", ec=COLORS["state_ec"], lw=1.0, size=7.0)
    box(ax, (x + 0.030, y + 0.035), 0.150, 0.052,
        r"$s_{t+1}=[Z_{t+1},Y^\star,\Delta_{t+1},A_t^{obs}]$",
        fc="#FFFFFF", ec=COLORS["state_ec"], lw=1.0, size=6.8)

    arrow(ax, anchors["H_t"], (x + 0.030, y + 0.240), color=COLORS["state_ec"], lw=1.2, rad=-0.05)
    arrow(ax, (x + 0.154, y + 0.216), (x + 0.105, y + 0.180), color=COLORS["state_ec"], lw=1.0)
    arrow(ax, anchors["target"], (x + 0.065, y + 0.180), color=COLORS["line"], lw=1.0, rad=0.04)
    arrow(ax, anchors["delta"], (x + 0.118, y + 0.180), color=COLORS["line"], lw=1.0, rad=-0.04)
    return {
        "Z": (x + 0.154, y + 0.240),
        "s": (x + 0.180, y + 0.148),
        "snext": (x + 0.180, y + 0.061),
    }


def draw_weight_badge(ax, start_anchor):
    x, y, w, h = 0.318, 0.435, 0.205, 0.100
    box(ax, (x, y), w, h, fc=COLORS["weight_fc"], ec=COLORS["weight_ec"], lw=1.2)
    ax.add_patch(Circle((x + 0.016, y + h - 0.024), 0.014, facecolor=COLORS["weight_ec"],
                        edgecolor=COLORS["weight_ec"], zorder=3))
    label(ax, x + 0.016, y + h - 0.024, "3", size=7, weight="bold", color="white")
    label(ax, x + 0.050, y + h - 0.024, "Deconfounding weight", size=8.1,
          weight="bold", ha="left")
    label(ax, x + 0.050, y + h - 0.050, r"from  $\omega(Z_t,A_t^{obs})$",
          size=6.5, color=COLORS["muted"], ha="left")
    dot(ax, x + 0.062, y + 0.026, r"$w_t$", "#FFFFFF", COLORS["weight_ec"], size=9, r=0.024)
    box(ax, (x + 0.112, y + 0.008), 0.070, 0.038, "cap +\nrenorm", fc="#FFFFFF",
        ec=COLORS["weight_ec"], lw=0.9, size=5.7)
    arrow(ax, start_anchor, (x + 0.041, y + 0.026), color=COLORS["weight_ec"], lw=1.3,
          rad=-0.12)
    return (x + 0.062, y + 0.026)


def draw_iql_losses(ax, state_anchors, weight_anchor):
    x, y, w, h = 0.575, 0.480, 0.360, 0.420
    box(ax, (x, y), w, h, fc="#FFFFFF", ec="#9CA3AF", lw=1.1)
    ax.add_patch(Circle((x + 0.020, y + h - 0.030), 0.014, facecolor="#596579",
                        edgecolor="#596579", zorder=3))
    label(ax, x + 0.020, y + h - 0.030, "4", size=7, weight="bold", color="white")
    label(ax, x + 0.055, y + h - 0.030, "Weighted IQL training objectives", size=8.5,
          weight="bold", ha="left")
    label(ax, x + 0.055, y + h - 0.058,
          r"code path: V-step $\rightarrow$ Twin-Q/encoder-step $\rightarrow$ AWR actor-step",
          size=6.4, color=COLORS["muted"], ha="left")

    def train_row(yy, step_title, formula, loss_text, desc, fc, ec):
        box(ax, (x + 0.030, yy), 0.068, 0.066, step_title, fc=fc, ec=ec,
            lw=1.0, size=6.8, weight="bold")
        box(ax, (x + 0.118, yy), 0.164, 0.066, formula, fc="#FFFFFF", ec=ec,
            lw=1.0, size=5.9)
        box(ax, (x + 0.302, yy), 0.048, 0.066, loss_text, fc="#FFFFFF", ec=ec,
            lw=1.0, size=6.7, weight="bold")
        label(ax, x + 0.118, yy - 0.014, desc, size=5.4, color=COLORS["muted"], ha="left")
        arrow(ax, (x + 0.098, yy + 0.033), (x + 0.118, yy + 0.033), color=ec, lw=1.0)
        arrow(ax, (x + 0.282, yy + 0.033), (x + 0.302, yy + 0.033), color=ec, lw=1.0)
        return {
            "step": (x + 0.030, yy + 0.033),
            "formula": (x + 0.118, yy + 0.033),
            "loss": (x + 0.326, yy + 0.033),
        }

    value = train_row(
        y + 0.278,
        "V\nupdate",
        r"$u_t=\min(Q_1,Q_2)-V_\psi(s_t)$" "\n"
        r"weighted expectile loss",
        "V\nloss",
        r"states detached; critic queried with observed action",
        COLORS["net_fc"],
        COLORS["net_ec"],
    )
    q_update = train_row(
        y + 0.168,
        "Twin-Q\n+ Enc",
        r"target: $r_t+\gamma(1-d_t)V_\psi(s_{t+1})$" "\n"
        r"weighted TD loss for $Q_1,Q_2$",
        "Q\nloss",
        r"updates both Q critics and the history encoder",
        COLORS["loss_fc"],
        COLORS["loss_ec"],
    )
    actor = train_row(
        y + 0.058,
        "Actor\nAWR",
        r"$\exp(\beta u_t)\cdot\ell_{BC}(\pi_\phi(s_t),A_t^{obs})$" "\n"
        r"weighted AWR actor loss",
        r"$\pi$" "\n" "loss",
        r"default $\ell_{BC}$ is expectile behavior cloning",
        COLORS["actor_fc"],
        COLORS["actor_ec"],
    )

    label(ax, x + 0.028, y + 0.310, r"$s_t,A_t^{obs}$", size=5.6,
          color=COLORS["state_ec"], ha="right")
    label(ax, x + 0.028, y + 0.200, r"$s_t,A_t^{obs},r_t,d_t$", size=5.4,
          color=COLORS["state_ec"], ha="right")
    label(ax, x + 0.028, y + 0.090, r"$s_t,A_t^{obs}$", size=5.6,
          color=COLORS["state_ec"], ha="right")

    arrow(ax, state_anchors["s"], value["step"], color=COLORS["state_ec"], lw=1.1, rad=-0.06)
    arrow(ax, state_anchors["s"], q_update["step"], color=COLORS["state_ec"], lw=1.1)
    arrow(ax, state_anchors["s"], actor["step"], color=COLORS["state_ec"], lw=1.1, rad=0.06)
    arrow(ax, state_anchors["snext"], (x + 0.206, y + 0.201), color=COLORS["state_ec"], lw=0.9,
          ls="--", rad=0.10)
    label(ax, x + 0.205, y + 0.245, r"$s_{t+1}$ for TD target", size=5.3,
          color=COLORS["state_ec"])

    # The same deconfounding sample weight multiplies every optimization loss.
    for pt in (value["loss"], q_update["loss"], actor["loss"]):
        arrow(ax, weight_anchor, pt, color=COLORS["weight_ec"], lw=1.2, rad=0.03,
              ls="--", zorder=1)

    return {
        "actor_net": actor["formula"],
        "actor_loss": actor["loss"],
    }


def draw_inference_rollout(ax, actor_anchor):
    x, y, w, h = 0.575, 0.085, 0.360, 0.305
    box(ax, (x, y), w, h, fc=COLORS["rollout_fc"], ec=COLORS["rollout_ec"], lw=1.2)
    ax.add_patch(Circle((x + 0.020, y + h - 0.030), 0.014, facecolor=COLORS["rollout_ec"],
                        edgecolor=COLORS["rollout_ec"], zorder=3))
    label(ax, x + 0.020, y + h - 0.030, "5", size=7, weight="bold", color="white")
    label(ax, x + 0.055, y + h - 0.030, "Inference: closed-loop target planning", size=8.5,
          weight="bold", ha="left")
    label(ax, x + 0.055, y + h - 0.058,
          "WeightNet is not queried; the trained actor chooses actions.",
          size=6.4, color=COLORS["muted"], ha="left")

    box(ax, (x + 0.030, y + 0.166), 0.085, 0.050, r"$H_t$", fc="#FFFFFF",
        ec=COLORS["rollout_ec"], lw=1.0, size=8.2)
    box(ax, (x + 0.142, y + 0.166), 0.088, 0.050, r"Encoder" "\n" r"$\to Z_t$",
        fc="#FFFFFF", ec=COLORS["rollout_ec"], lw=1.0, size=7.1)
    box(ax, (x + 0.257, y + 0.166), 0.070, 0.050, r"$s_t^{plan}$", fc="#FFFFFF",
        ec=COLORS["rollout_ec"], lw=1.0, size=8.0)
    arrow(ax, (x + 0.115, y + 0.191), (x + 0.142, y + 0.191), color=COLORS["rollout_ec"], lw=1.1)
    arrow(ax, (x + 0.230, y + 0.191), (x + 0.257, y + 0.191), color=COLORS["rollout_ec"], lw=1.1)

    box(ax, (x + 0.070, y + 0.077), 0.098, 0.052, r"$Y^\star,\Delta_t,$" "\n" r"$a_{t-1}$",
        fc="#FFFFFF", ec=COLORS["rollout_ec"], lw=1.0, size=7.1)
    arrow(ax, (x + 0.166, y + 0.103), (x + 0.257, y + 0.166), color=COLORS["rollout_ec"], lw=1.0,
          rad=-0.15)

    box(ax, (x + 0.130, y + 0.010), 0.085, 0.046, r"Actor $\pi_\phi$", fc=COLORS["actor_fc"],
        ec=COLORS["actor_ec"], lw=1.0, size=7.5, weight="bold")
    dot(ax, x + 0.287, y + 0.033, r"$a_t^\pi$", "#FFFFFF", COLORS["actor_ec"], size=8.5, r=0.026)
    arrow(ax, (x + 0.292, y + 0.166), (x + 0.215, y + 0.056), color=COLORS["actor_ec"], lw=1.0,
          rad=-0.05)
    arrow(ax, (x + 0.215, y + 0.033), (x + 0.261, y + 0.033), color=COLORS["actor_ec"], lw=1.2)
    arrow(ax, (x + 0.312, y + 0.033), (x + 0.344, y + 0.190), color=COLORS["rollout_ec"], lw=1.0,
          rad=-0.25)
    label(ax, x + 0.345, y + 0.214, "roll out\none step", size=5.9, color=COLORS["rollout_ec"])
    arrow(ax, actor_anchor, (x + 0.173, y + 0.056), color=COLORS["actor_ec"], lw=0.9,
          ls="--", rad=-0.20)
    label(ax, x + 0.105, y + 0.010, "trained parameters", size=5.8, color=COLORS["actor_ec"], ha="right")


def draw_legend(ax):
    x, y = 0.045, 0.105
    label(ax, x, y + 0.040, "Notation", size=7.4, weight="bold", ha="left")
    label(ax, x, y + 0.018, r"$A_t^{obs}$: logged treatment used for training", size=6.4,
          color=COLORS["muted"], ha="left")
    label(ax, x, y - 0.005, r"$a_t^\pi$: planned treatment output by policy", size=6.4,
          color=COLORS["muted"], ha="left")
    label(ax, x, y - 0.028, r"$w_t$: deconfounding weight from the balance module", size=6.4,
          color=COLORS["muted"], ha="left")


def draw_figure(out_dir: Path) -> None:
    configure_matplotlib()
    fig = plt.figure(figsize=(11.2, 5.0), dpi=300)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    label(ax, 0.5, 0.955, "Reweighted Target-Conditioned Policy Optimization", size=12,
          weight="bold")
    label(
        ax,
        0.5,
        0.925,
        "Training uses observed treatments and deconfounding weights; inference uses the learned actor to plan target-reaching doses.",
        size=7.6,
        color=COLORS["muted"],
    )

    sample = draw_training_sample(ax)
    state = draw_state_builder(ax, sample)
    weight = draw_weight_badge(ax, state["Z"])
    losses = draw_iql_losses(ax, state, weight)
    draw_inference_rollout(ax, losses["actor_net"])
    draw_legend(ax)

    label(ax, 0.040, 0.455, "training phase", size=7.4, weight="bold",
          color=COLORS["muted"], ha="left")
    ax.plot([0.035, 0.940], [0.430, 0.430], color="#D1D5DB", lw=0.9, ls=(0, (3, 3)))
    label(ax, 0.040, 0.407, "deployment / evaluation phase", size=7.4, weight="bold",
          color=COLORS["muted"], ha="left")

    out_dir.mkdir(parents=True, exist_ok=True)
    base = out_dir / "reweighted_target_conditioned_policy_optimization"
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.02)
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.02)
    fig.savefig(base.with_suffix(".png"), bbox_inches="tight", pad_inches=0.02, dpi=300)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=Path("plots/method_framework"))
    args = parser.parse_args()
    draw_figure(args.out_dir)


if __name__ == "__main__":
    main()
