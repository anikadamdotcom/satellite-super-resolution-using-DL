"""Convert extracted SEN2VENµS sites into memory-mappable training arrays.

Input (from download_sen2venus.py): datasets/sen2venus/<SITE>/<SITE>/
  <SITE>_<TILE>_<DATE>_10m_b2b3b4b8.pt  int16 (N, 4, 128, 128)  Sentinel-2 10 m
  <SITE>_<TILE>_<DATE>_05m_b2b3b4b8.pt  int16 (N, 4, 256, 256)  VENµS 5 m reference
Both are reflectance x 10000 in band order B2, B3, B4, B8 (the project convention).

Output: datasets/sen2venus_npy/<split>/<SITE>_<TILE>_<DATE>_{lr,hr}.npy as uint16 NHWC
plus <split>/patches.csv (site, tile, date, file, index, negative fraction, estimated shift).

Small negative reflectances (dark water, atmospheric-correction noise) are clipped to 0.
Patches are dropped when more than --max-negative of their pixels are negative, when any
value looks like nodata/saturation, or when a band is constant (fill).
"""
import _bootstrap  # noqa: F401  (puts src/ on sys.path)
import argparse
import csv
import re
import shutil
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from download_sen2venus import SPLITS

PATTERN = re.compile(r"^(?P<site>.+)_(?P<tile>\d{2}[A-Z]{3})_(?P<date>\d{4}-\d{2}-\d{2})_10m_b2b3b4b8\.pt$")


def estimate_shift(lr, hr, radius=3, margin=16):
    """Integer HR-pixel shift (dy, dx) that best aligns HR to bicubic-upsampled LR."""
    up = F.interpolate(lr, scale_factor=2, mode="bicubic", align_corners=False)
    errors = torch.stack([
        (torch.roll(hr, (dy, dx), (2, 3)) - up)[..., margin:-margin, margin:-margin].abs().mean((1, 2, 3))
        for dy in range(-radius, radius + 1) for dx in range(-radius, radius + 1)], 1)
    index = errors.argmin(1)
    side = 2 * radius + 1
    return (index // side - radius).tolist(), (index % side - radius).tolist()


def patch_filter(lr, hr, max_negative):
    """Boolean keep mask and negative-pixel fraction per patch (int16 tensors)."""
    neg = torch.maximum((lr < 0).float().mean((1, 2, 3)), (hr < 0).float().mean((1, 2, 3)))
    nodata = (lr.amin((1, 2, 3)) <= -1000) | (hr.amin((1, 2, 3)) <= -1000)
    saturated = (lr.amax((1, 2, 3)) >= 15000) | (hr.amax((1, 2, 3)) >= 15000)
    constant = (lr.float().flatten(2).std(2).amin(1) == 0) | (hr.float().flatten(2).std(2).amin(1) == 0)
    keep = (neg <= max_negative) & ~nodata & ~saturated & ~constant
    return keep, neg


def to_uint16_nhwc(tensor):
    return tensor.clamp(min=0).permute(0, 2, 3, 1).numpy().astype(np.uint16)


def prepare(source, output, splits, max_negative=0.1):
    output = Path(output)
    tmp = output.with_name(output.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    summary = {}
    for split, sites in splits.items():
        folder = tmp / split
        folder.mkdir(parents=True)
        rows = []
        for site in sites:
            files = sorted((Path(source) / site).rglob(f"{site}_*_10m_b2b3b4b8.pt"))
            if not files:
                raise FileNotFoundError(f"No extracted SEN2VENµS files for {site} under {source}")
            kept = total = 0
            for lr_path in files:
                match = PATTERN.match(lr_path.name)
                hr_path = lr_path.with_name(lr_path.name.replace("_10m_b2b3b4b8", "_05m_b2b3b4b8"))
                if not match or not hr_path.exists():
                    raise ValueError(f"Unexpected or unpaired file: {lr_path}")
                lr = torch.load(lr_path, map_location="cpu", weights_only=True)
                hr = torch.load(hr_path, map_location="cpu", weights_only=True)
                if (lr.dtype != torch.int16 or hr.dtype != torch.int16 or lr.ndim != 4 or lr.shape[1] != 4
                        or hr.shape[0] != lr.shape[0] or hr.shape[1] != 4 or hr.shape[2:] != (lr.shape[2] * 2, lr.shape[3] * 2)):
                    raise ValueError(f"{lr_path.name}: unexpected tensors {lr.shape} {lr.dtype} / {hr.shape} {hr.dtype}")
                keep, neg = patch_filter(lr, hr, max_negative)
                total += lr.shape[0]
                if not keep.any():
                    continue
                index = keep.nonzero().flatten()
                lr, hr, neg = lr[index], hr[index], neg[index]
                dy, dx = estimate_shift(lr.float().clamp(min=0) / 1e4, hr.float().clamp(min=0) / 1e4)
                stem = lr_path.name.replace("_10m_b2b3b4b8.pt", "")
                np.save(folder / f"{stem}_lr.npy", to_uint16_nhwc(lr))
                np.save(folder / f"{stem}_hr.npy", to_uint16_nhwc(hr))
                for row, original in enumerate(index.tolist()):
                    rows.append(dict(site=site, tile=match["tile"], date=match["date"], file=stem, row=row,
                                     source_index=original, negative_fraction=round(float(neg[row]), 5),
                                     shift_dy=dy[row], shift_dx=dx[row]))
                kept += len(index)
            print(f"{split:5s} {site:10s} kept {kept}/{total} patches from {len(files)} acquisitions", flush=True)
            summary[site] = dict(split=split, kept=kept, total=total)
        with open(folder / "patches.csv", "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    shutil.rmtree(output, ignore_errors=True)
    tmp.replace(output)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", default="datasets/sen2venus")
    parser.add_argument("--output", default="datasets/sen2venus_npy")
    parser.add_argument("--train", nargs="+", default=SPLITS["train"])
    parser.add_argument("--val", nargs="+", default=SPLITS["val"])
    parser.add_argument("--test", nargs="+", default=SPLITS["test"])
    parser.add_argument("--max-negative", type=float, default=0.1)
    args = parser.parse_args()
    if set(args.train) & set(args.val) or set(args.train) & set(args.test) or set(args.val) & set(args.test):
        parser.error("A site may appear in only one split")
    prepare(args.source, args.output, dict(train=args.train, val=args.val, test=args.test), args.max_negative)


if __name__ == "__main__":
    main()
