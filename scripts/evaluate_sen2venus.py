"""Cross-sensor test: Sentinel-2 10 m -> VENµS 5 m on held-out SEN2VENµS sites.

Every method upsamples the real 10 m Sentinel-2 patch 2x and is scored against the real
VENµS 5 m patch acquired the same day. Two scorings are reported:

- raw: direct comparison.
- compensated ("aligned"): the prediction is moved by the patch's sub-pixel
  Sentinel-2/VENµS shift (estimated once, from bicubic vs. VENµS, in prepare_sen2venus.py)
  and each band's mean offset is removed before scoring; 4 px border excluded. This takes
  out the known misregistration and band offsets without searching per method, so a blurry
  method cannot win by matching the misregistration. Every method gets the same treatment.
- harmonised: as compensated, but against the VENµS reference rescaled per band to
  Sentinel-2 contrast and mean at 40 m (VENµS has ~5-9% less contrast). This scores
  spatial detail on Sentinel-2's own radiometry, which is what the app needs.

Usage: python scripts/evaluate_sen2venus.py --models new=checkpoints/real/rcan.pt old=checkpoints/demo/rcan.pt
"""
import _bootstrap  # noqa: F401  (puts src/ on sys.path)
import argparse
import csv
import json
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch

from satsr.data import SRPatchDataset
from satsr.models.evaluation_metrics import calculate_psnr, calculate_sam, calculate_ssim
from satsr.models.rcan import L1BSRSR



def compensate(prediction, target, shift, margin=4):
    """Apply the patch's sub-pixel shift (models/alignment.py) and remove per-band mean offsets.

    The shift is estimated once from bicubic vs. VENµS in prepare_sen2venus.py, so it does
    not depend on the method being scored. Returns float64 HWC arrays cropped by `margin`.
    """
    from satsr.models.alignment import shift_image
    tensor = torch.from_numpy(np.moveaxis(prediction.astype(np.float32), -1, 0))[None]
    aligned = np.moveaxis(shift_image(tensor, torch.tensor([shift]))[0].numpy(), 0, -1).astype(np.float64)
    aligned, core = aligned[margin:-margin, margin:-margin], target[margin:-margin, margin:-margin].astype(np.float64)
    aligned = aligned - (aligned - core).mean((0, 1))
    return np.clip(aligned, 0, 65535), core


def detail_gain(prediction, lr):
    """Std of (prediction - nearest-upsampled input): how much sub-pixel detail a method adds."""
    up = np.repeat(np.repeat(lr.astype(np.float64), 2, 0), 2, 1)
    return float((prediction.astype(np.float64) - up).std())


def metrics(a, b):
    return dict(psnr_db=calculate_psnr(a, b), ssim=calculate_ssim(a, b), sam_degrees=calculate_sam(a, b)[0])


def score(prediction, target, shift, harmonised):
    """raw: vs VENµS as is. compensated: aligned. harmonised: aligned, vs VENµS on S2 radiometry."""
    return (metrics(prediction, target), metrics(*compensate(prediction, target, shift)),
            metrics(*compensate(prediction, harmonised, shift)))


KINDS = ("raw", "compensated", "harmonised")


def score_all(predictions, target, shift, lr, harmonised):
    return {name: dict(**dict(zip(KINDS, score(p, target, shift, harmonised))), detail=detail_gain(p, lr))
            for name, p in predictions.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default="datasets/sen2venus_npy")
    parser.add_argument("--split", default="test")
    parser.add_argument("--models", nargs="*", default=[], help="name=path/to/rcan.pt")
    parser.add_argument("--sen2sr", action="store_true", help="Also score ESA SEN2SRLite (requirements/research.txt)")
    parser.add_argument("--limit", type=int, default=2000, help="Evenly spaced patches per split (0 = all)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", default="results/sen2venus_test.json")
    parser.add_argument("--jobs", type=int, default=8, help="Scoring processes")
    args = parser.parse_args()

    methods = {"bicubic": L1BSRSR.bicubic_upscale}
    provenance = {"bicubic": "PyTorch bicubic interpolation"}
    for item in args.models:
        name, path = item.split("=", 1)
        engine = L1BSRSR(path, args.device)
        methods[name], provenance[name] = engine.super_resolve, f"{engine.provenance} [{path}]"
    if args.sen2sr:
        from satsr.models.sen2sr_lite import SEN2SRLite
        sen2sr = SEN2SRLite(device=args.device)
        methods["sen2sr"], provenance["sen2sr"] = sen2sr.super_resolve, SEN2SRLite.provenance

    data = SRPatchDataset(args.data, args.split, harmonise=False)
    with open(Path(args.data)/args.split/"patches.csv", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    # patches.csv rows follow file order; SRPatchDataset sorts files, so key rows by (file, row).
    meta = {(r["file"], int(r["row"])): r for r in rows}
    indices = range(len(data)) if not args.limit or args.limit >= len(data) else \
        np.linspace(0, len(data)-1, args.limit).astype(int).tolist()
    results = []
    pending = deque()

    def collect(block=False):
        while pending and (block or pending[0][1].done() or len(pending) > 4*args.jobs):
            entry, future = pending.popleft()
            entry.update(future.result())
            results.append(entry)
            if len(results) % 500 == 1:
                print(f"{len(results)}/{len(indices)} " + "  ".join(
                    f"{n} {entry[n]['compensated']['psnr_db']:.2f}" for n in methods), flush=True)

    # Predictions run here (GPU); CPU-heavy scoring runs in worker processes.
    with ProcessPoolExecutor(args.jobs) as pool:
        for index in indices:
            lr, hr = data.arrays(int(index))
            file, row = data.index[int(index)]
            stem = data.paths[file][0].name[:-len("_lr.npy")]
            entry = dict(site=meta[(stem, row)]["site"], file=stem, row=row)
            predictions = {name: method(lr) for name, method in methods.items()}
            pending.append((entry, pool.submit(score_all, predictions, hr, data.shift(int(index)).tolist(), lr, data.harmonised_hr(int(index), hr))))
            collect()
        collect(block=True)

    def summarise(items):
        out = {}
        for name in methods:
            out[name] = {}
            for kind in KINDS:
                out[name][kind] = {k: round(float(np.mean([r[name][kind][k] for r in items])), 4)
                                   for k in ("psnr_db", "ssim", "sam_degrees")}
                if name != "bicubic":
                    gains = np.array([r[name][kind]["psnr_db"] - r["bicubic"][kind]["psnr_db"] for r in items])
                    out[name][kind]["psnr_gain_vs_bicubic_db"] = round(float(gains.mean()), 3)
                    out[name][kind]["patches_better_than_bicubic"] = f"{int((gains > 0).sum())}/{len(items)}"
            out[name]["detail_std_vs_nearest"] = round(float(np.mean([r[name]["detail"] for r in items])), 2)
        return out

    sites = sorted({r["site"] for r in results})
    report = dict(split=args.split, patches=len(results), provenance=provenance,
                  overall=summarise(results), per_site={s: summarise([r for r in results if r["site"] == s]) for s in sites},
                  protocol=__doc__.split("Usage:")[0].strip())
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, indent=2), encoding="utf-8")
    for kind in KINDS:
        print(f"\n{kind}:")
        for name, values in report["overall"].items():
            print(f"  {name:10s} " + "  ".join(f"{k}={v}" for k, v in values[kind].items()))
    data.close()


if __name__ == "__main__":
    main()
