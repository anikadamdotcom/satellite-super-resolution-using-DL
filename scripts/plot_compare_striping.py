"""Compare-view check: change-vs-bicubic maps and a striping score for several RCAN checkpoints.

Striping score = share of the change-vs-bicubic signal's spectral energy at the 2-pixel
period (rows or columns), i.e. the regular stripes seen with the synthetic-trained RCAN.

Usage: python scripts/plot_compare_striping.py name=path.pt [...] --output docs/compare_striping.png
"""
import _bootstrap  # noqa: F401  (puts src/ on sys.path)
import argparse
import json
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image, ImageDraw

from satsr.models.rcan import L1BSRSR
from _bootstrap import ROOT

SCENES = sorted((ROOT/"demo_images").glob("sentinel2_*.tif"))


def striping(diff):
    """Fraction of spectral energy in the Nyquist row/column (2 px period) of a 2-D signal."""
    d = diff - diff.mean()
    power = np.abs(np.fft.fft2(d))**2
    h, w = power.shape
    nyquist = power[h//2, :].sum() + power[:, w//2].sum() - power[h//2, w//2]
    return float(nyquist / max(power.sum() - power[0, 0], 1e-12))


def to_rgb(bands):
    rgb = bands[..., [2, 1, 0]].astype(np.float32)
    lo, hi = np.percentile(rgb, (2, 98))
    return (np.clip((rgb - lo)/(hi - lo + 1e-6), 0, 1)*255).astype(np.uint8)


def to_heat(diff, top=300.0):
    v = np.clip(diff/top, 0, 1)
    return (np.stack([1 - .75*v, 1 - .55*v, 1 - .15*v], -1)*255).astype(np.uint8)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("models", nargs="+", help="name=checkpoint.pt")
    parser.add_argument("--output", default="docs/compare_striping.png")
    parser.add_argument("--json", default="results/compare_striping.json")
    parser.add_argument("--crop", type=int, default=160)
    args = parser.parse_args()
    engines = {n: L1BSRSR(p, "cpu") for n, p in (m.split("=", 1) for m in args.models)}
    scores, rows = {n: [] for n in engines}, []
    for path in SCENES:
        with rasterio.open(path) as src:
            image = np.moveaxis(src.read(), 0, -1)[:args.crop, :args.crop, :4]
        bicubic = L1BSRSR.bicubic_upscale(image).astype(np.float32)
        tiles = [to_rgb(bicubic)]
        for name, engine in engines.items():
            out = engine.super_resolve(image).astype(np.float32)
            diff = np.abs(out - bicubic).mean(-1)
            scores[name].append(dict(scene=path.stem, striping=round(striping((out - bicubic).mean(-1)), 4),
                                     mean_change=round(float(diff.mean()), 1)))
            tiles.append(to_heat(diff))
        rows.append(tiles)
    size, gap, head = rows[0][0].shape[0], 6, 28
    names = ["Bicubic (RGB)"] + [f"{n}: change" for n in engines]
    canvas = Image.new("RGB", (len(names)*(size + gap) - gap, head + len(rows)*(size + gap)), "white")
    draw = ImageDraw.Draw(canvas)
    for c, name in enumerate(names):
        draw.text((c*(size + gap) + 4, 8), name, fill=(20, 20, 20))
        for r, tiles in enumerate(rows):
            canvas.paste(Image.fromarray(tiles[c]), (c*(size + gap), head + r*(size + gap)))
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    canvas.quantize(256).save(args.output, optimize=True)   # palette PNG keeps the repo small
    summary = {n: dict(mean_striping=round(float(np.mean([s["striping"] for s in v])), 4),
                       mean_change=round(float(np.mean([s["mean_change"] for s in v])), 1), scenes=v)
               for n, v in scores.items()}
    Path(args.json).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    for n, v in summary.items():
        print(f"{n:16s} striping={v['mean_striping']:.4f} mean change={v['mean_change']:.1f}")


if __name__ == "__main__":
    main()
