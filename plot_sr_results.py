"""Figures for SR_COMPARISON.md from results/sr_comparison.json and results/mapping_experiment.json."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
# Fixed colour per method (validated categorical slots 1-4); text stays in ink colours.
COLORS = dict(bicubic="#2a78d6", rcan="#eb6834", sen2sr="#1baf7a", native="#eda100")
NAMES = dict(native="Original 10 m", bicubic="Bicubic", rcan="RCAN (project)", sen2sr="SEN2SR (ESA)")
INK, MUTED, SURFACE, GRID = "#0b0b0b", "#52514e", "#fcfcfb", "#e4e3df"


def style(ax, title, ylabel):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", fontsize=11, color=INK, pad=10)
    ax.set_ylabel(ylabel, color=MUTED, fontsize=9)
    ax.tick_params(colors=MUTED, labelsize=9, length=0)
    ax.grid(axis="y", color=GRID, linewidth=.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)


def bars(ax, methods, values, errors=None, fmt="{:+.2f}"):
    x = np.arange(len(methods))
    ax.bar(x, values, width=.6, color=[COLORS[m] for m in methods], edgecolor=SURFACE, linewidth=2,
           yerr=errors, error_kw=dict(ecolor=MUTED, elinewidth=1, capsize=3))
    ax.set_xticks(x, [NAMES[m] for m in methods])
    span = max(abs(v) for v in values) or 1
    for xi, v, e in zip(x, values, errors if errors is not None else [0]*len(values)):
        ax.text(xi, v+(e+span*.04)*(1 if v >= 0 else -1), fmt.format(v), ha="center",
                va="bottom" if v >= 0 else "top", fontsize=9, color=INK)


def summary_figure():
    sr = json.loads((ROOT/"results"/"sr_comparison.json").read_text(encoding="utf-8"))["summary"]
    mapping_path = ROOT/"results"/"mapping_experiment.json"
    mapping = json.loads(mapping_path.read_text(encoding="utf-8"))["summary"] if mapping_path.exists() else None
    fig, axes = plt.subplots(1, 2 if mapping else 1, figsize=(11 if mapping else 5.5, 4), facecolor=SURFACE, squeeze=False)
    ax = axes[0, 0]
    methods = ["rcan", "sen2sr"]
    bars(ax, methods, [sr["reduced_scale"][m]["psnr_gain_vs_bicubic_db"] for m in methods], fmt="{:+.2f} dB")
    ax.axhline(0, color=MUTED, linewidth=1)
    style(ax, "Image quality on real Sentinel-2 (42 scenes)", "PSNR change vs bicubic (dB)")
    ax.text(.0, -.2, "Reduced-scale test: 20 m input rebuilt to 10 m, scored against the real 10 m image.",
            transform=ax.transAxes, fontsize=8, color=MUTED)
    ax.margins(y=.3)
    if mapping:
        ax = axes[0, 1]
        methods = [m for m in ("native", "bicubic", "rcan", "sen2sr") if m in mapping]
        bars(ax, methods, [100*mapping[m]["mean_iou"]["mean"] for m in methods],
             [100*mapping[m]["mean_iou"]["std"] for m in methods], fmt="{:.1f}%")
        low = min(100*mapping[m]["mean_iou"]["mean"] for m in methods)
        ax.set_ylim(max(0, low-12), None)
        style(ax, "Land-cover mapping on held-out sites (8 sites)", "Mean IoU vs WorldCover (%)")
        ax.text(.0, -.2, "Same U-Net, labels and seeds for every input; bars = mean of 3 seeds, lines = spread.",
                transform=ax.transAxes, fontsize=8, color=MUTED)
    fig.tight_layout()
    out = ROOT/"docs"/"sr_comparison.png"
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    print("Saved", out)


def visual_figure():
    from compare_sr_models import Upsamplers
    from geo_io import load_any_user_image
    _, image, _, _ = load_any_user_image(ROOT/"demo_images"/"sentinel2_urban_delhi_2023-12-05.tif")
    crop = image[96:160, 96:160]
    up = Upsamplers()
    panels = [("native", up("nearest", crop)), ("bicubic", up("bicubic", crop)),
              ("rcan", up("rcan", crop)), ("sen2sr", up("sen2sr", crop))]
    rgb = [p[..., [2, 1, 0]].astype(np.float32) for _, p in panels]
    lo, hi = np.percentile(rgb[0], 2), np.percentile(rgb[0], 98)
    fig, axes = plt.subplots(1, 4, figsize=(12, 3.4), facecolor=SURFACE)
    for ax, (name, _), image_rgb in zip(axes, panels, rgb):
        ax.imshow(np.clip((image_rgb-lo)/(hi-lo), 0, 1), interpolation="nearest")
        ax.set_title(NAMES[name], fontsize=10, color=INK)
        ax.axis("off")
    fig.suptitle("Central New Delhi, 640 m x 640 m, shown on a 5 m grid (same colour stretch in every panel)",
                 fontsize=9, color=MUTED, y=.03)
    fig.tight_layout(rect=(0, .07, 1, 1))
    out = ROOT/"docs"/"sr_visual_comparison.png"
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    print("Saved", out)


if __name__ == "__main__":
    summary_figure()
    visual_figure()
