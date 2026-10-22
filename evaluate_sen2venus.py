"""Cross-sensor test: Sentinel-2 10 m -> VENµS 5 m on held-out SEN2VENµS sites.

Every method upsamples the real 10 m Sentinel-2 patch 2x and is scored against the real
VENµS 5 m patch acquired the same day. Two scorings are reported:

- raw: direct comparison.
- compensated: the prediction is shifted by the best integer offset (|d| <= 2 px at 5 m)
  and each band's mean offset is removed before scoring. This takes out the known
  Sentinel-2/VENµS misregistration and band-response differences so that only spatial
  detail is compared. Every method, bicubic included, gets the same treatment.

Usage: python evaluate_sen2venus.py --models new=checkpoints/real/rcan.pt old=checkpoints/demo/rcan.pt
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from data import SRPatchDataset
from models.evaluation_metrics import calculate_psnr, calculate_sam, calculate_ssim
from models.rcan import L1BSRSR

ROOT = Path(__file__).resolve().parent


def compensate(prediction, target, radius=2):
    """Return (aligned, bias-corrected prediction, target) cropped by `radius`, as float64."""
    p, t = prediction.astype(np.float64), target.astype(np.float64)
    h, w = t.shape[:2]
    core = t[radius:h-radius, radius:w-radius]
    best, best_err = None, np.inf
    for dy in range(-radius, radius+1):
        for dx in range(-radius, radius+1):
            window = p[radius+dy:h-radius+dy, radius+dx:w-radius+dx]
            diff = window - core
            err = np.abs(diff - diff.mean((0, 1))).mean()
            if err < best_err:
                best, best_err = window - diff.mean((0, 1)), err
    return np.clip(best, 0, 65535), core


def score(prediction, target):
    raw = dict(psnr_db=calculate_psnr(prediction, target), ssim=calculate_ssim(prediction, target),
               sam_degrees=calculate_sam(prediction, target)[0])
    aligned, core = compensate(prediction, target)
    comp = dict(psnr_db=calculate_psnr(aligned, core), ssim=calculate_ssim(aligned, core),
                sam_degrees=calculate_sam(aligned, core)[0])
    return raw, comp


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default="datasets/sen2venus_npy")
    parser.add_argument("--split", default="test")
    parser.add_argument("--models", nargs="*", default=[], help="name=path/to/rcan.pt")
    parser.add_argument("--sen2sr", action="store_true", help="Also score ESA SEN2SRLite (requirements-research.txt)")
    parser.add_argument("--limit", type=int, default=2000, help="Evenly spaced patches per split (0 = all)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", default="results/sen2venus_test.json")
    args = parser.parse_args()

    methods = {"bicubic": L1BSRSR.bicubic_upscale}
    provenance = {"bicubic": "PyTorch bicubic interpolation"}
    for item in args.models:
        name, path = item.split("=", 1)
        engine = L1BSRSR(path, args.device)
        methods[name], provenance[name] = engine.super_resolve, f"{engine.provenance} [{path}]"
    if args.sen2sr:
        from models.sen2sr_lite import SEN2SRLite
        sen2sr = SEN2SRLite(device=args.device)
        methods["sen2sr"], provenance["sen2sr"] = sen2sr.super_resolve, SEN2SRLite.provenance

    data = SRPatchDataset(args.data, args.split)
    with open(Path(args.data)/args.split/"patches.csv", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    # patches.csv rows follow file order; SRPatchDataset sorts files, so key rows by (file, row).
    meta = {(r["file"], int(r["row"])): r for r in rows}
    indices = range(len(data)) if not args.limit or args.limit >= len(data) else \
        np.linspace(0, len(data)-1, args.limit).astype(int).tolist()
    results = []
    for count, index in enumerate(indices):
        lr, hr = data.arrays(int(index))
        file, row = data.index[int(index)]
        stem = data.paths[file][0].name[:-len("_lr.npy")]
        entry = dict(site=meta[(stem, row)]["site"], file=stem, row=row)
        for name, method in methods.items():
            raw, comp = score(method(lr), hr)
            entry[name] = dict(raw=raw, compensated=comp)
        results.append(entry)
        if count % 200 == 0:
            print(f"{count}/{len(indices)} " + "  ".join(f"{n} {entry[n]['compensated']['psnr_db']:.2f}" for n in methods), flush=True)

    def summarise(items):
        out = {}
        for name in methods:
            out[name] = {}
            for kind in ("raw", "compensated"):
                out[name][kind] = {k: round(float(np.mean([r[name][kind][k] for r in items])), 4)
                                   for k in ("psnr_db", "ssim", "sam_degrees")}
                if name != "bicubic":
                    gains = np.array([r[name][kind]["psnr_db"] - r["bicubic"][kind]["psnr_db"] for r in items])
                    out[name][kind]["psnr_gain_vs_bicubic_db"] = round(float(gains.mean()), 3)
                    out[name][kind]["patches_better_than_bicubic"] = f"{int((gains > 0).sum())}/{len(items)}"
        return out

    sites = sorted({r["site"] for r in results})
    report = dict(split=args.split, patches=len(results), provenance=provenance,
                  overall=summarise(results), per_site={s: summarise([r for r in results if r["site"] == s]) for s in sites},
                  protocol=__doc__.split("Usage:")[0].strip())
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, indent=2), encoding="utf-8")
    for kind in ("raw", "compensated"):
        print(f"\n{kind}:")
        for name, values in report["overall"].items():
            print(f"  {name:10s} " + "  ".join(f"{k}={v}" for k, v in values[kind].items()))
    data.close()


if __name__ == "__main__":
    main()
