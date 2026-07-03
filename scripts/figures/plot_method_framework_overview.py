#!/usr/bin/env python
"""
Draw a two-column method-framework overview for the CRIPO / CT-IQL method.

The schematic is designed as a KDD/ACM double-column figure. It is not copied
from any reference figure; it uses the same visual grammar of multi-stream
causal sequence models and target-driven offline policy optimization.

Example:
    python scripts/figures/plot_method_framework_overview.py \
        --out-dir plots/method_framework_overview
"""

import argparse
from pathlib import Path
import textwrap

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Circle, Ellipse, Rectangle
import numpy as np


COLORS = {
    "ink": "#202124",
    "muted": "#5F6368",
    "hairline": "#C9CDD4",
    "data_fc": "#EAF4F0",
    "data_ec": "#4F8B78",
    "target_fc": "#F2F6E9",
    "target_ec": "#7A9447",
    "encoder_fc": "#EAF1FB",
    "encoder_ec": "#477DB3",
    "iql_fc": "#FBEAF2",
    "iql_ec": "#C04D84",
    "rollout_fc": "#FFF3D6",
    "rollout_ec": "#C68A2A",
    "support_fc": "#F5F7FA",
    "support_ec": "#6B7280",
    "stream_a": "#7B9ACC",
    "stream_y": "#8CCB9B",
    "stream_x": "#D9A85C",
    "ours": "#D81B60",
    "critic": "#574B90",
    "median": "#2F6F73",
    "gray": "#8A8F98",
}


def configure_matplotlib() -> None:
    """Configure publication-style matplotlib defaults."""
    font_paths = [
        "/usr/share/fonts/opentype/linux-libertine/LinLibertine_R.otf",
        "/usr/share/fonts/opentype/linux-libertine/LinLibertine_RB.otf",
        "/Library/Fonts/Linux Libertine O.ttf",
    ]
    for path in font_paths:
        p = Path(path)
        if p.exists():
            font_manager.fontManager.addfont(str(p))

    available = {f.name for f in font_manager.fontManager.ttflist}
    if "Linux Libertine O" in available:
        family = "serif"
        serif = ["Linux Libertine O", "Times New Roman", "DejaVu Serif"]
    elif "Times New Roman" in available:
        family = "serif"
        serif = ["Times New Roman", "DejaVu Serif"]
    else:
        family = "sans-serif"
        serif = ["Arial", "Helvetica", "DejaVu Sans"]

    mpl.rcParams.update(
        {
            "font.family": family,
            "font.serif": serif,
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "mathtext.fontset": "stix",
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 7.0,
            "axes.linewidth": 0.7,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "legend.frameon": False,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def rounded_box(
    ax,
    xy,
    w,
    h,
    fc,
    ec,
    lw=1.1,
    radius=0.018,
    alpha=1.0,
    zorder=1,
):
    patch = FancyBboxPatch(
        xy,
        w,
        h,
        boxstyle=f"round,pad=0.006,rounding_size={radius}",
        linewidth=lw,
        edgecolor=ec,
        facecolor=fc,
        alpha=alpha,
        zorder=zorder,
    )
    ax.add_patch(patch)
    return patch


def arrow(ax, xy0, xy1, color=None, lw=1.0, mutation_scale=8, style="-|>", ls="-", rad=0.0, zorder=4):
    patch = FancyArrowPatch(
        xy0,
        xy1,
        arrowstyle=style,
        mutation_scale=mutation_scale,
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


def label(ax, x, y, text, size=7, weight="regular", color=None, ha="center", va="center", **kwargs):
    return ax.text(
        x,
        y,
        text,
        fontsize=size,
        fontweight=weight,
        color=color or COLORS["ink"],
        ha=ha,
        va=va,
        linespacing=1.08,
        **kwargs,
    )


def wrapped_label(ax, x, y, text, width, size=6.3, color=None, ha="center", va="center", **kwargs):
    wrapped = "\n".join(textwrap.wrap(text, width=width, break_long_words=False))
    return label(ax, x, y, wrapped, size=size, color=color, ha=ha, va=va, **kwargs)


def draw_header(ax):
    label(
        ax,
        0.5,
        0.975,
        "CRIPO method overview: target-conditioned CT-IQL for continuous treatment control",
        size=9.3,
        weight="bold",
    )
    label(
        ax,
        0.5,
        0.944,
        "Local-global causal history representation -> target-conditioned offline RL -> closed-loop dose planning",
        size=6.5,
        color=COLORS["muted"],
    )


def draw_stage_number(ax, x, y, n, color):
    circ = Circle((x, y), 0.013, facecolor="white", edgecolor=color, lw=1.0, zorder=6)
    ax.add_patch(circ)
    label(ax, x, y - 0.0005, str(n), size=6.2, weight="bold", color=color, zorder=7)


def token_row(ax, x0, y0, w, h, n=7, color="#7B9ACC", label_text=None):
    gap = w * 0.018
    token_w = (w - gap * (n - 1)) / n
    for i in range(n):
        alpha = 0.33 + 0.055 * i
        rect = FancyBboxPatch(
            (x0 + i * (token_w + gap), y0),
            token_w,
            h,
            boxstyle="round,pad=0.002,rounding_size=0.006",
            facecolor=color,
            edgecolor="white",
            linewidth=0.4,
            alpha=alpha,
            zorder=3,
        )
        ax.add_patch(rect)
    if label_text:
        label(ax, x0 - 0.010, y0 + h / 2, label_text, size=5.9, color=COLORS["muted"], ha="right")


def draw_data_module(ax):
    x, y, w, h = 0.035, 0.575, 0.235, 0.325
    rounded_box(ax, (x, y), w, h, COLORS["data_fc"], COLORS["data_ec"])
    draw_stage_number(ax, x + 0.017, y + h - 0.026, 1, COLORS["data_ec"])
    label(ax, x + 0.058, y + h - 0.026, "Observational histories", size=7.1, weight="bold", ha="left")
    label(ax, x + 0.058, y + h - 0.052, r"$H_t=(X_{\leq t}, A_{<t}, Y_{\leq t}, V)$", size=6.1, color=COLORS["muted"], ha="left")

    tx = x + 0.050
    token_row(ax, tx, y + 0.195, w - 0.078, 0.027, n=7, color=COLORS["stream_x"], label_text=r"$X,V$")
    token_row(ax, tx, y + 0.150, w - 0.078, 0.027, n=7, color=COLORS["stream_a"], label_text=r"$A$")
    token_row(ax, tx, y + 0.105, w - 0.078, 0.027, n=7, color=COLORS["stream_y"], label_text=r"$Y$")
    for i, t in enumerate([r"$t-5$", r"$t-4$", r"$t-3$", r"$t-2$", r"$t-1$", r"$t$", ""]):
        label(ax, tx + (w - 0.078) * (i + 0.5) / 7, y + 0.073, t, size=5.0, color=COLORS["muted"])
    rounded_box(ax, (x + 0.025, y + 0.012), w - 0.050, 0.052, "#FFFFFF", COLORS["data_ec"], lw=0.8, radius=0.010)
    label(ax, x + w / 2, y + 0.047, "continuous doses in simulator space", size=5.4, color=COLORS["muted"])
    label(ax, x + w / 2, y + 0.028, r"$a_t=(a^{chemo},a^{radio})\in[0,1]^2$", size=5.6)
    return x, y, w, h


def draw_target_module(ax):
    x, y, w, h = 0.035, 0.145, 0.235, 0.315
    rounded_box(ax, (x, y), w, h, COLORS["target_fc"], COLORS["target_ec"])
    draw_stage_number(ax, x + 0.017, y + h - 0.026, 2, COLORS["target_ec"])
    label(ax, x + 0.058, y + h - 0.026, "Horizon-aligned target replay", size=7.1, weight="bold", ha="left")
    label(ax, x + 0.058, y + h - 0.052, r"sample future target $Y^\star=Y_{t+\tau}$", size=6.1, color=COLORS["muted"], ha="left")

    # Mini timeline with target endpoint.
    x0, y0 = x + 0.037, y + 0.185
    ax.plot([x0, x0 + 0.155], [y0, y0], color=COLORS["target_ec"], lw=1.0, zorder=3)
    for i in range(5):
        cx = x0 + 0.155 * i / 4
        ax.add_patch(Circle((cx, y0), 0.006, facecolor="white", edgecolor=COLORS["target_ec"], lw=0.9, zorder=4))
    ax.add_patch(Circle((x0 + 0.155, y0), 0.010, facecolor="#D9E8B4", edgecolor=COLORS["target_ec"], lw=1.0, zorder=5))
    label(ax, x0, y0 - 0.030, r"$t$", size=5.6, color=COLORS["muted"])
    label(ax, x0 + 0.155, y0 - 0.030, r"$t+\tau$", size=5.6, color=COLORS["muted"])
    label(ax, x0 + 0.155, y0 + 0.030, r"$Y^\star$", size=6.2, weight="bold", color=COLORS["target_ec"])

    tuple_box = (x + 0.022, y + 0.035, w - 0.044, 0.100)
    rounded_box(ax, tuple_box[:2], tuple_box[2], tuple_box[3], "#FFFFFF", COLORS["target_ec"], lw=0.8, radius=0.010)
    label(ax, x + w / 2, y + 0.104, "raw IQL tuple", size=6.1, weight="bold", color=COLORS["target_ec"])
    label(ax, x + w / 2, y + 0.075, r"$(H_t,a_t,r_t,H_{t+1},Y^\star,\Delta_t,a_{t-1})$", size=5.8)
    label(ax, x + w / 2, y + 0.048, r"$r_t$: progress toward target outcome", size=5.5, color=COLORS["muted"])
    return x, y, w, h


def draw_encoder_module(ax):
    x, y, w, h = 0.335, 0.575, 0.285, 0.325
    rounded_box(ax, (x, y), w, h, COLORS["encoder_fc"], COLORS["encoder_ec"])
    draw_stage_number(ax, x + 0.017, y + h - 0.026, 3, COLORS["encoder_ec"])
    label(ax, x + 0.058, y + h - 0.026, "Local-global causal history encoder", size=7.1, weight="bold", ha="left")
    label(ax, x + 0.058, y + h - 0.052, "three separated CT streams with shared static context", size=6.0, color=COLORS["muted"], ha="left")

    sx = x + 0.025
    stream_w = 0.106
    token_row(ax, sx + 0.012, y + 0.205, stream_w, 0.022, n=5, color=COLORS["stream_a"], label_text=r"$A$")
    token_row(ax, sx + 0.012, y + 0.165, stream_w, 0.022, n=5, color=COLORS["stream_y"], label_text=r"$Y$")
    token_row(ax, sx + 0.012, y + 0.125, stream_w, 0.022, n=5, color=COLORS["stream_x"], label_text=r"$X,V$")

    # Two stacked processing blocks inspired by local ConvFormer and CT.
    bx = x + 0.158
    bw = 0.101
    rounded_box(ax, (bx, y + 0.180), bw, 0.055, "#FFFFFF", COLORS["encoder_ec"], lw=0.8, radius=0.010, zorder=3)
    label(ax, bx + bw / 2, y + 0.207, "local causal\nconv mixer", size=5.8)
    rounded_box(ax, (bx, y + 0.103), bw, 0.060, "#FFFFFF", COLORS["encoder_ec"], lw=0.8, radius=0.010, zorder=3)
    label(ax, bx + bw / 2, y + 0.133, "multi-input\nself/cross-attn", size=5.8)
    arrow(ax, (sx + 0.130, y + 0.176), (bx - 0.006, y + 0.207), COLORS["encoder_ec"], lw=0.8, mutation_scale=6)
    arrow(ax, (sx + 0.130, y + 0.176), (bx - 0.006, y + 0.133), COLORS["encoder_ec"], lw=0.8, mutation_scale=6)
    arrow(ax, (bx + bw / 2, y + 0.180), (bx + bw / 2, y + 0.163), COLORS["encoder_ec"], lw=0.8, mutation_scale=6)

    rounded_box(ax, (x + 0.084, y + 0.026), w - 0.168, 0.048, "#FFFFFF", COLORS["encoder_ec"], lw=0.8, radius=0.010)
    label(ax, x + w / 2, y + 0.050, r"project and gather last valid state: $Z_t$", size=6.0)
    arrow(ax, (bx + bw / 2, y + 0.103), (x + w / 2, y + 0.077), COLORS["encoder_ec"], lw=0.8, mutation_scale=6)

    # Cross-stream links.
    for yy0 in [y + 0.216, y + 0.176, y + 0.136]:
        arrow(ax, (sx + 0.055, yy0), (sx + 0.105, yy0 - 0.040), COLORS["hairline"], lw=0.55, mutation_scale=4, style="-")
    return x, y, w, h


def draw_iql_module(ax):
    x, y, w, h = 0.675, 0.575, 0.290, 0.325
    rounded_box(ax, (x, y), w, h, COLORS["iql_fc"], COLORS["iql_ec"])
    draw_stage_number(ax, x + 0.017, y + h - 0.026, 4, COLORS["iql_ec"])
    label(ax, x + 0.058, y + h - 0.026, "Target-conditioned IQL/AWR", size=7.1, weight="bold", ha="left")
    label(ax, x + 0.058, y + h - 0.052, r"$s_t=[Z_t,Y^\star,\Delta_t,a_{t-1}]$", size=6.2, color=COLORS["muted"], ha="left")

    # State vector.
    vx, vy = x + 0.032, y + 0.205
    comps = [
        (r"$Z_t$", "#DBEAFE"),
        (r"$Y^\star$", "#FCE7F3"),
        (r"$\Delta_t$", "#FEF3C7"),
        (r"$a_{t-1}$", "#E5E7EB"),
    ]
    cx = vx
    for txt, fc in comps:
        rounded_box(ax, (cx, vy), 0.047, 0.035, fc, COLORS["hairline"], lw=0.6, radius=0.008)
        label(ax, cx + 0.0235, vy + 0.0175, txt, size=6.0)
        cx += 0.051
    # V/Q/pi nodes.
    node_y = y + 0.105
    node_xs = [x + 0.061, x + 0.145, x + 0.229]
    node_labels = [r"$V_\tau(s)$", r"$Q_1,Q_2(s,a)$", r"$\pi_\theta(a|s)$"]
    node_sub = ["expectile", "twin critic", "AWR + expectile BC"]
    for nx, nl, ns in zip(node_xs, node_labels, node_sub):
        rounded_box(ax, (nx - 0.038, node_y), 0.076, 0.058, "#FFFFFF", COLORS["iql_ec"], lw=0.85, radius=0.012)
        label(ax, nx, node_y + 0.036, nl, size=6.0, weight="bold")
        label(ax, nx, node_y + 0.015, ns, size=4.8, color=COLORS["muted"])
        arrow(ax, (vx + 0.102, vy), (nx, node_y + 0.060), COLORS["iql_ec"], lw=0.65, mutation_scale=5)

    arrow(ax, (node_xs[1] - 0.038, node_y + 0.029), (node_xs[0] + 0.038, node_y + 0.029), COLORS["iql_ec"], lw=0.65, mutation_scale=5)
    arrow(ax, (node_xs[1] + 0.038, node_y + 0.029), (node_xs[2] - 0.038, node_y + 0.029), COLORS["iql_ec"], lw=0.65, mutation_scale=5)
    label(ax, x + w / 2, y + 0.043, r"advantage weights: $\exp(\beta(Q-V))$", size=5.8, color=COLORS["iql_ec"])
    return x, y, w, h


def draw_em_module(ax):
    x, y, w, h = 0.335, 0.145, 0.285, 0.315
    rounded_box(ax, (x, y), w, h, "#FFFFFF", COLORS["hairline"], lw=1.0, radius=0.018)
    label(ax, x + 0.020, y + h - 0.028, "EM-style deconfounded training loop", size=7.0, weight="bold", ha="left")
    label(ax, x + 0.020, y + h - 0.054, "weights are learned on the replayed observational transitions", size=5.8, color=COLORS["muted"], ha="left")

    # E/M loop nodes.
    rounded_box(ax, (x + 0.026, y + 0.168), 0.095, 0.070, "#EFF6FF", COLORS["encoder_ec"], lw=0.8, radius=0.012)
    label(ax, x + 0.0735, y + 0.207, "E-step", size=6.2, weight="bold", color=COLORS["encoder_ec"])
    label(ax, x + 0.0735, y + 0.184, "WeightNet\nlearns $w_t$", size=5.5)

    rounded_box(ax, (x + 0.163, y + 0.168), 0.095, 0.070, "#FDF2F8", COLORS["iql_ec"], lw=0.8, radius=0.012)
    label(ax, x + 0.2105, y + 0.207, "M-step", size=6.2, weight="bold", color=COLORS["iql_ec"])
    label(ax, x + 0.2105, y + 0.184, r"weighted" + "\n" + r"$V\rightarrow Q\rightarrow \pi$", size=5.5)
    arrow(ax, (x + 0.122, y + 0.203), (x + 0.163, y + 0.203), COLORS["muted"], lw=0.9, mutation_scale=7)
    arrow(ax, (x + 0.210, y + 0.168), (x + 0.073, y + 0.168), COLORS["muted"], lw=0.8, mutation_scale=7, rad=-0.28)

    rounded_box(ax, (x + 0.026, y + 0.045), 0.232, 0.075, "#FAFAFA", COLORS["hairline"], lw=0.75, radius=0.012)
    label(ax, x + 0.142, y + 0.097, "representation update", size=5.9, weight="bold", color=COLORS["muted"])
    wrapped_label(
        ax,
        x + 0.142,
        y + 0.066,
        "encoder gradients flow through the weighted critic loss; policy update stays close to supported actions",
        width=55,
        size=5.4,
        color=COLORS["muted"],
    )
    return x, y, w, h


def draw_support_inset(ax, x0, y0, w, h):
    rounded_box(ax, (x0, y0), w, h, COLORS["support_fc"], COLORS["support_ec"], lw=0.8, radius=0.010)
    # Mini axes.
    px0, py0 = x0 + 0.020, y0 + 0.017
    pw, ph = w - 0.040, h - 0.052
    ax.plot([px0, px0, px0 + pw], [py0 + ph, py0, py0], color=COLORS["hairline"], lw=0.55, zorder=3)
    ax.add_patch(Ellipse((px0 + 0.060, py0 + 0.026), 0.092, 0.038, angle=18, facecolor="#D1D5DB", edgecolor="none", alpha=0.65, zorder=3))
    rng = np.random.RandomState(2)
    pts = rng.normal(size=(28, 2))
    pts[:, 0] = px0 + 0.060 + pts[:, 0] * 0.020
    pts[:, 1] = py0 + 0.026 + pts[:, 1] * 0.008
    ax.scatter(pts[:, 0], pts[:, 1], s=3.8, color=COLORS["gray"], alpha=0.75, linewidths=0, zorder=4)
    ax.scatter([px0 + 0.072], [py0 + 0.031], s=28, marker="*", color=COLORS["ours"], edgecolor="white", linewidth=0.35, zorder=6)
    ax.scatter([px0 + 0.132], [py0 + 0.034], s=22, marker="x", color=COLORS["critic"], linewidth=1.0, zorder=6)
    ax.scatter([px0 + 0.058], [py0 + 0.024], s=16, marker="o", facecolor="white", edgecolor=COLORS["median"], linewidth=0.8, zorder=6)
    label(
        ax,
        x0 + w / 2,
        y0 + h - 0.014,
        "empirical local action support",
        size=5.2,
        weight="bold",
        zorder=8,
        bbox=dict(facecolor=COLORS["support_fc"], edgecolor="none", boxstyle="round,pad=0.10", alpha=0.95),
    )
    label(ax, px0 + pw - 0.007, py0 - 0.007, "chemo", size=4.4, color=COLORS["muted"], ha="right", va="top")
    label(ax, px0 - 0.003, py0 + ph + 0.001, "radio", size=4.4, color=COLORS["muted"], ha="right", va="bottom", rotation=90)


def draw_rollout_module(ax):
    x, y, w, h = 0.675, 0.145, 0.290, 0.315
    rounded_box(ax, (x, y), w, h, COLORS["rollout_fc"], COLORS["rollout_ec"])
    draw_stage_number(ax, x + 0.017, y + h - 0.026, 5, COLORS["rollout_ec"])
    label(ax, x + 0.058, y + h - 0.026, "Closed-loop intervention rollout", size=7.1, weight="bold", ha="left")
    label(ax, x + 0.058, y + h - 0.052, r"replan at each step $k=0,\ldots,\tau-1$", size=6.0, color=COLORS["muted"], ha="left")

    # Rollout loop.
    rounded_box(ax, (x + 0.030, y + 0.190), 0.080, 0.052, "#FFFFFF", COLORS["rollout_ec"], lw=0.8, radius=0.011)
    label(ax, x + 0.070, y + 0.216, r"$\pi_\theta(s)$", size=6.3, weight="bold", color=COLORS["iql_ec"])
    label(ax, x + 0.070, y + 0.198, "policy", size=5.0, color=COLORS["muted"])

    rounded_box(ax, (x + 0.142, y + 0.190), 0.105, 0.052, "#FFFFFF", COLORS["rollout_ec"], lw=0.8, radius=0.011)
    label(ax, x + 0.1945, y + 0.218, r"$a_t=(chemo,radio)$", size=5.9, weight="bold")
    label(ax, x + 0.1945, y + 0.198, r"simulator space $[0,1]$", size=5.0, color=COLORS["muted"])

    rounded_box(ax, (x + 0.085, y + 0.111), 0.110, 0.050, "#FFFFFF", COLORS["rollout_ec"], lw=0.8, radius=0.011)
    label(ax, x + 0.140, y + 0.139, r"$Y_{t+1}$ and updated $H_{t+1}$", size=5.7, weight="bold")
    label(ax, x + 0.140, y + 0.121, "world model / simulator", size=5.0, color=COLORS["muted"])
    arrow(ax, (x + 0.110, y + 0.216), (x + 0.142, y + 0.216), COLORS["rollout_ec"], lw=0.8, mutation_scale=6)
    arrow(ax, (x + 0.194, y + 0.190), (x + 0.163, y + 0.162), COLORS["rollout_ec"], lw=0.8, mutation_scale=6)
    arrow(ax, (x + 0.086, y + 0.136), (x + 0.055, y + 0.190), COLORS["rollout_ec"], lw=0.75, mutation_scale=6, rad=-0.28)

    draw_support_inset(ax, x + 0.030, y + 0.021, 0.217, 0.077)
    return x, y, w, h


def connect_modules(ax, boxes):
    data = boxes["data"]
    target = boxes["target"]
    encoder = boxes["encoder"]
    iql = boxes["iql"]
    em = boxes["em"]
    rollout = boxes["rollout"]
    # Data to encoder and target replay.
    arrow(ax, (data[0] + data[2], data[1] + 0.215), (encoder[0], encoder[1] + 0.215), COLORS["muted"], lw=1.05, mutation_scale=9)
    label(ax, 0.302, 0.798, r"$H_t$", size=5.7, color=COLORS["muted"])
    arrow(ax, (data[0] + data[2] / 2, data[1]), (target[0] + target[2] / 2, target[1] + target[3]), COLORS["target_ec"], lw=0.9, mutation_scale=8, rad=0.12)
    # Encoder and replay to IQL.
    arrow(ax, (encoder[0] + encoder[2], encoder[1] + 0.172), (iql[0], iql[1] + 0.172), COLORS["muted"], lw=1.05, mutation_scale=9)
    label(ax, 0.648, 0.754, r"$Z_t$", size=5.7, color=COLORS["muted"])
    arrow(ax, (target[0] + target[2], target[1] + 0.145), (em[0], em[1] + 0.145), COLORS["target_ec"], lw=0.9, mutation_scale=8)
    # EM loop supports IQL.
    arrow(ax, (em[0] + em[2], em[1] + 0.205), (iql[0] + 0.057, iql[1] + 0.070), COLORS["muted"], lw=0.85, mutation_scale=7, rad=-0.16)
    label(ax, 0.645, 0.392, r"$w_t$", size=5.5, color=COLORS["muted"])
    # IQL to rollout.
    arrow(ax, (iql[0] + iql[2] / 2, iql[1]), (rollout[0] + rollout[2] / 2, rollout[1] + rollout[3]), COLORS["rollout_ec"], lw=1.0, mutation_scale=9)
    label(ax, 0.820, 0.515, r"$\pi_\theta$", size=5.8, color=COLORS["rollout_ec"])


def draw_legend_strip(ax):
    x0, y0 = 0.035, 0.055
    items = [
        ("treatment stream", COLORS["stream_a"]),
        ("outcome stream", COLORS["stream_y"]),
        ("covariate/static stream", COLORS["stream_x"]),
        ("CRIPO action", COLORS["ours"]),
        ("critic-greedy diagnostic", COLORS["critic"]),
    ]
    x = x0
    for text, color in items:
        ax.add_patch(Circle((x, y0), 0.006, facecolor=color, edgecolor="none", zorder=4))
        label(ax, x + 0.012, y0, text, size=5.5, color=COLORS["muted"], ha="left")
        x += 0.168 if text != "covariate/static stream" else 0.190


def write_caption(out_dir: Path) -> None:
    caption = (
        "Figure X | CRIPO method overview. Observational trajectories are converted into "
        "horizon-aligned target replay tuples. A local-global causal history encoder first "
        "mixes local temporal neighborhoods within treatment, outcome, and covariate streams, "
        "then applies CT-style multi-input self/cross-attention to obtain the latent state Z_t. "
        "The target-conditioned IQL state concatenates Z_t, the target outcome, remaining "
        "horizon, and previous action. EM-style training learns deconfounding weights and "
        "updates the value, twin critic, and AWR/expectile policy. At deployment, the policy "
        "selects continuous chemo/radio actions in simulator space and rolls forward in closed "
        "loop; the inset illustrates the empirical local action support diagnostic used to "
        "compare planned actions with nearby behavior actions."
    )
    (out_dir / "caption.txt").write_text(caption + "\n", encoding="utf-8")


def build_figure():
    configure_matplotlib()
    fig = plt.figure(figsize=(7.25, 4.45), dpi=600)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    draw_header(ax)
    boxes = {
        "data": draw_data_module(ax),
        "target": draw_target_module(ax),
        "encoder": draw_encoder_module(ax),
        "iql": draw_iql_module(ax),
        "em": draw_em_module(ax),
        "rollout": draw_rollout_module(ax),
    }
    connect_modules(ax, boxes)
    draw_legend_strip(ax)
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default="plots/method_framework_overview", help="Output directory.")
    parser.add_argument("--basename", default="cripo_method_framework_overview", help="Output file basename.")
    parser.add_argument("--dpi", type=int, default=600, help="PNG export DPI.")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fig = build_figure()
    stem = out_dir / args.basename
    fig.savefig(str(stem) + ".svg", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(str(stem) + ".pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(str(stem) + ".png", dpi=args.dpi, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    write_caption(out_dir)
    print(f"Saved {stem}.svg/.pdf/.png")
    print(f"Saved {out_dir / 'caption.txt'}")


if __name__ == "__main__":
    main()
