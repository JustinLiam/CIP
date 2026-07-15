#!/usr/bin/env python
"""
Create an editable SVG illustration of a smooth data manifold.

The primary target is OmniGraffle editing, so the SVG avoids bitmap fills and
SVG blur filters. Soft shadows are simulated with stacked translucent vector
paths/ellipses. The script also exports PNG/PDF previews with matplotlib.

Example:
    python scripts/figures/plot_data_manifold_omnigraffle.py \
        --out-dir plots/omnigraffle_data_manifold
"""

import argparse
import math
from pathlib import Path as FilePath

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Ellipse, PathPatch
from matplotlib.path import Path as MplPath


W, H = 900, 650

MANIFOLD_PATH_D = (
    "M 60 505 "
    "C 125 418 231 391 290 334 "
    "C 365 263 345 209 270 168 "
    "C 216 139 196 90 229 47 "
    "C 250 22 294 22 346 22 "
    "C 430 21 520 20 650 20 "
    "C 590 51 565 96 591 148 "
    "C 637 232 667 275 628 352 "
    "C 589 430 515 494 450 586 "
    "C 352 558 219 540 60 505 Z"
)

HIGHLIGHT_PATH_D = (
    "M 248 53 "
    "C 226 88 249 127 295 153 "
    "C 360 191 382 244 336 310 "
    "C 288 379 204 411 127 470 "
    "C 244 456 337 424 420 356 "
    "C 485 303 522 237 494 166 "
    "C 472 111 502 64 590 29 "
    "C 459 32 326 32 248 53 Z"
)

RIGHT_SHADE_PATH_D = (
    "M 610 42 "
    "C 567 82 566 121 594 166 "
    "C 633 228 651 278 615 346 "
    "C 581 410 520 474 454 565 "
    "C 499 535 588 459 643 362 "
    "C 694 272 646 205 608 142 "
    "C 585 103 594 70 610 42 Z"
)

POINTS = [
    (515, 93, "(s1, a1)"),
    (598, 178, "(s2, a2)"),
    (556, 290, "(s3, a3)"),
    (458, 438, "(sn, an)"),
]

SMALL_POINTS = [(512, 313), (532, 352), (548, 397)]


def configure_matplotlib():
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )


def manifold_path():
    verts = [
        (60, 505),
        (125, 418), (231, 391), (290, 334),
        (365, 263), (345, 209), (270, 168),
        (216, 139), (196, 90), (229, 47),
        (250, 22), (294, 22), (346, 22),
        (430, 21), (520, 20), (650, 20),
        (590, 51), (565, 96), (591, 148),
        (637, 232), (667, 275), (628, 352),
        (589, 430), (515, 494), (450, 586),
        (352, 558), (219, 540), (60, 505),
        (60, 505),
    ]
    codes = [
        MplPath.MOVETO,
        MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4,
        MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4,
        MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4,
        MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4,
        MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4,
        MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4,
        MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4,
        MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4,
        MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4,
        MplPath.CLOSEPOLY,
    ]
    return MplPath(verts, codes)


def shifted_path(path, dx, dy):
    verts = path.vertices.copy()
    verts[:, 0] += dx
    verts[:, 1] += dy
    return MplPath(verts, path.codes)


def draw_matplotlib_preview(out_dir, stem, dpi):
    configure_matplotlib()
    fig = plt.figure(figsize=(6.0, 4.33), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.axis("off")

    path = manifold_path()
    for dx, dy, alpha in [(18, 18, 0.055), (11, 12, 0.070), (5, 6, 0.050)]:
        ax.add_patch(PathPatch(shifted_path(path, dx, dy), facecolor="#245EA7", edgecolor="none", alpha=alpha))

    ax.add_patch(PathPatch(path, facecolor="#E8F2FF", edgecolor="#2567AF", linewidth=1.35))
    ax.add_patch(PathPatch(path, facecolor="#FFFFFF", edgecolor="none", alpha=0.12))

    highlight = path_from_svg_like(HIGHLIGHT_PATH_D)
    right_shade = path_from_svg_like(RIGHT_SHADE_PATH_D)
    ax.add_patch(PathPatch(highlight, facecolor="#FFFFFF", edgecolor="none", alpha=0.25))
    ax.add_patch(PathPatch(right_shade, facecolor="#6EA4DE", edgecolor="none", alpha=0.18))

    ax.text(140, 482, r"$\mathcal{M}_{D}$", fontsize=21, color="#0B2D8D", ha="center", va="center")
    for x, y, text in POINTS:
        draw_point_preview(ax, x, y)
        ax.text(x + 46, y - 10, text.replace("1", r"$_1$").replace("2", r"$_2$").replace("3", r"$_3$").replace("n", r"$_n$"),
                fontsize=14, color="#0B2D8D", ha="left", va="center")
    for x, y in SMALL_POINTS:
        draw_point_preview(ax, x, y, radius=12, label=False)
    ax.text(507, 371, r"$\vdots$", fontsize=21, color="#0B2D8D", ha="center", va="center")

    fig.savefig(out_dir / (stem + "_preview.png"), dpi=dpi, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(out_dir / (stem + "_preview.pdf"), bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def draw_point_preview(ax, x, y, radius=17, label=True):
    ax.add_patch(Ellipse((x + 18, y + 20), radius * 2.1, radius * 1.0, facecolor="#1A376A", edgecolor="none", alpha=0.12))
    ax.add_patch(Ellipse((x + 10, y + 13), radius * 1.6, radius * 0.78, facecolor="#1A376A", edgecolor="none", alpha=0.12))
    ax.add_patch(Circle((x, y), radius, facecolor="#1E6BE0", edgecolor="#062B91", linewidth=1.3))
    ax.add_patch(Circle((x - radius * 0.35, y - radius * 0.35), radius * 0.45, facecolor="#6CA8FF", edgecolor="none", alpha=0.75))


def path_from_svg_like(d):
    tokens = d.replace(",", " ").split()
    verts = []
    codes = []
    i = 0
    while i < len(tokens):
        token = tokens[i]
        if token == "M":
            verts.append((float(tokens[i + 1]), float(tokens[i + 2])))
            codes.append(MplPath.MOVETO)
            i += 3
        elif token == "C":
            for j in range(3):
                verts.append((float(tokens[i + 1 + 2 * j]), float(tokens[i + 2 + 2 * j])))
                codes.append(MplPath.CURVE4)
            i += 7
        elif token == "Z":
            verts.append(verts[-1])
            codes.append(MplPath.CLOSEPOLY)
            i += 1
        else:
            raise ValueError("Unexpected SVG path token: %r" % token)
    return MplPath(verts, codes)


def svg_text(x, y, text, size=32, weight="normal", fill="#0B2D8D", anchor="start"):
    return (
        '<text x="{x}" y="{y}" font-family="Times New Roman, STIXGeneral, serif" '
        'font-size="{size}" font-weight="{weight}" fill="{fill}" '
        'text-anchor="{anchor}" dominant-baseline="middle">{text}</text>'
    ).format(x=x, y=y, text=text, size=size, weight=weight, fill=fill, anchor=anchor)


def sample_label_svg(x, y, subscript, size=28):
    sub_size = int(round(size * 0.68))
    text = (
        "(s<tspan baseline-shift=\"sub\" font-size=\"{sub_size}\">{sub}</tspan>, "
        "a<tspan baseline-shift=\"sub\" font-size=\"{sub_size}\">{sub}</tspan>)"
    ).format(sub_size=sub_size, sub=subscript)
    return svg_text(x, y, text, size=size)


def point_svg(x, y, r=17):
    return """
  <g class="editable-data-point" id="point-{x}-{y}">
    <ellipse cx="{sx3}" cy="{sy3}" rx="{rx3}" ry="{ry3}" fill="#17427C" opacity="0.055"/>
    <ellipse cx="{sx2}" cy="{sy2}" rx="{rx2}" ry="{ry2}" fill="#17427C" opacity="0.075"/>
    <ellipse cx="{sx1}" cy="{sy1}" rx="{rx1}" ry="{ry1}" fill="#17427C" opacity="0.105"/>
    <circle cx="{x}" cy="{y}" r="{r}" fill="url(#dotGradient)" stroke="#062B91" stroke-width="2.2"/>
    <circle cx="{hx}" cy="{hy}" r="{hr}" fill="#8EBBFF" opacity="0.72"/>
  </g>""".format(
        x=int(x),
        y=int(y),
        r=r,
        sx1=x + r * 0.60,
        sy1=y + r * 0.75,
        rx1=r * 1.05,
        ry1=r * 0.48,
        sx2=x + r * 0.95,
        sy2=y + r * 1.12,
        rx2=r * 1.35,
        ry2=r * 0.62,
        sx3=x + r * 1.35,
        sy3=y + r * 1.50,
        rx3=r * 1.70,
        ry3=r * 0.78,
        hx=x - r * 0.33,
        hy=y - r * 0.36,
        hr=r * 0.38,
    )


def point_svg_flat(x, y, r=17):
    return """
  <g class="editable-data-point-flat" id="point-flat-{x}-{y}">
    <ellipse cx="{sx3}" cy="{sy3}" rx="{rx3}" ry="{ry3}" fill="#17427C" opacity="0.05"/>
    <ellipse cx="{sx2}" cy="{sy2}" rx="{rx2}" ry="{ry2}" fill="#17427C" opacity="0.08"/>
    <ellipse cx="{sx1}" cy="{sy1}" rx="{rx1}" ry="{ry1}" fill="#17427C" opacity="0.12"/>
    <circle cx="{x}" cy="{y}" r="{r}" fill="#1F6FDD" stroke="#062B91" stroke-width="2.2"/>
    <circle cx="{hx}" cy="{hy}" r="{hr}" fill="#7FB5FF" opacity="0.82"/>
  </g>""".format(
        x=int(x),
        y=int(y),
        r=r,
        sx1=x + r * 0.60,
        sy1=y + r * 0.75,
        rx1=r * 1.05,
        ry1=r * 0.48,
        sx2=x + r * 0.95,
        sy2=y + r * 1.12,
        rx2=r * 1.35,
        ry2=r * 0.62,
        sx3=x + r * 1.35,
        sy3=y + r * 1.50,
        rx3=r * 1.70,
        ry3=r * 0.78,
        hx=x - r * 0.33,
        hy=y - r * 0.36,
        hr=r * 0.38,
    )


def write_svg(out_dir, stem, gradient=True):
    if gradient:
        defs = """
  <defs>
    <linearGradient id="manifoldFill" x1="140" y1="42" x2="560" y2="565" gradientUnits="userSpaceOnUse">
      <stop offset="0%" stop-color="#F5FAFF"/>
      <stop offset="44%" stop-color="#E6F1FF"/>
      <stop offset="100%" stop-color="#CFE3FF"/>
    </linearGradient>
    <radialGradient id="dotGradient" cx="34%" cy="28%" r="72%">
      <stop offset="0%" stop-color="#86B8FF"/>
      <stop offset="44%" stop-color="#1E6EE5"/>
      <stop offset="100%" stop-color="#0C3BAA"/>
    </radialGradient>
  </defs>"""
        fill = "url(#manifoldFill)"
        dot_fn = point_svg
        name = stem + "_editable.svg"
    else:
        defs = ""
        fill = "#E8F2FF"
        dot_fn = point_svg_flat
        name = stem + "_flat_editable.svg"

    points = []
    for idx, (x, y, text) in enumerate(POINTS):
        points.append(dot_fn(x, y, 17))
        sub = str(idx + 1) if idx < 3 else "n"
        points.append(sample_label_svg(x + 46, y - 6, sub, size=28))
    for x, y in SMALL_POINTS:
        points.append(dot_fn(x, y, 12))
    points.append(svg_text(498, 370, "⋮", size=34, anchor="middle"))

    svg = """<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="900" height="650" viewBox="0 0 900 650">
  <title>Editable blue data manifold illustration</title>
  <desc>All visible elements are vector paths, ellipses, circles, or text for OmniGraffle editing.</desc>
{defs}
  <rect id="background-white" x="0" y="0" width="900" height="650" fill="#FFFFFF"/>

  <g id="manifold-soft-shadow" opacity="1">
    <path d="{manifold}" transform="translate(22 22)" fill="#245EA7" opacity="0.035"/>
    <path d="{manifold}" transform="translate(16 17)" fill="#245EA7" opacity="0.045"/>
    <path d="{manifold}" transform="translate(10 10)" fill="#245EA7" opacity="0.050"/>
    <path d="{manifold}" transform="translate(5 5)" fill="#245EA7" opacity="0.032"/>
  </g>

  <g id="editable-manifold">
    <path id="manifold-main-fill" d="{manifold}" fill="{fill}" stroke="#2468AF" stroke-width="3.0" stroke-linejoin="round"/>
    <path id="manifold-inner-light" d="{highlight}" fill="#FFFFFF" opacity="0.30"/>
    <path id="manifold-right-side-shade" d="{right_shade}" fill="#5E9FE4" opacity="0.16"/>
    <path id="manifold-front-edge-highlight" d="{manifold}" fill="none" stroke="#7FB2EA" stroke-width="1.15" opacity="0.65"/>
  </g>

  <g id="editable-label-manifold">
    {m_label}
  </g>

  <g id="editable-samples-and-labels">
{points}
  </g>
</svg>
""".format(
        defs=defs,
        manifold=MANIFOLD_PATH_D,
        highlight=HIGHLIGHT_PATH_D,
        right_shade=RIGHT_SHADE_PATH_D,
        fill=fill,
        m_label=svg_text(146, 485, "ℳ<tspan baseline-shift=\"sub\" font-size=\"19\">D</tspan>", size=36, anchor="middle"),
        points="\n".join(points),
    )
    (out_dir / name).write_text(svg, encoding="utf-8")


def write_caption(out_dir):
    caption = (
        "Editable data manifold illustration. The manifold surface, soft shadow, "
        "highlight, sample points, point shadows, and labels are separate SVG "
        "objects so they can be selected and modified in OmniGraffle. The figure "
        "intentionally omits internal straight lines and connector lines."
    )
    (out_dir / "caption.txt").write_text(caption + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default="plots/omnigraffle_data_manifold")
    parser.add_argument("--stem", default="data_manifold")
    parser.add_argument("--dpi", type=int, default=600)
    args = parser.parse_args()

    out_dir = FilePath(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    write_svg(out_dir, args.stem, gradient=True)
    write_svg(out_dir, args.stem, gradient=False)
    draw_matplotlib_preview(out_dir, args.stem, args.dpi)
    write_caption(out_dir)
    print("Saved editable SVG files and PNG/PDF previews to %s" % out_dir)


if __name__ == "__main__":
    main()
