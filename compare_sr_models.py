"""Compare bicubic, the project's RCAN and ESA's SEN2SRLite on real Sentinel-2.

Real Sentinel-2 has no same-day 5 m reference, so two standard checks are used:

1. Reduced-scale test (Wald protocol). Each real 10 m crop is area-averaged to
   20 m, every method upsamples it 2x back to 10 m, and the result is scored
   against the real 10 m image (PSNR, SSIM, spectral angle). Both models were
   built for 10 m input, so 20 m input is outside what they were trained on;
   this tests how much real detail each method recovers, at a different scale.
2. Native-scale checks. Each method upsamples the real 10 m crop to 5 m. With no
   reference, report (a) consistency: area-average the 5 m output back to 10 m
   and compare with the input, and (b) added detail: high-frequency (Laplacian)
   energy relative to bicubic. Added detail is not proof of correct detail.

    python compare_sr_models.py --output results/sr_comparison.json
"""
import argparse
import json
from pathlib import Path

import numpy as np

from download_mapping_benchmark import ROOT
from models.evaluation_metrics import calculate_edge_energy, reconstruction_metrics
from models.rcan import L1BSRSR
from models.sen2sr_lite import SEN2SRLite, block_mean

BENCHMARK = ROOT/"datasets"/"mapping_benchmark"
METHODS = ("bicubic", "rcan", "sen2sr")


class Upsamplers:
    """The three 2x methods behind one call: upsample(name, H x W x 4 uint16)."""

    def __init__(self, device="cpu", rcan_path=ROOT/"checkpoints"/"demo"/"rcan.pt"):
        self.rcan = L1BSRSR(rcan_path, device)
        self.sen2sr = SEN2SRLite(device=device)
        self.provenance = dict(bicubic="PyTorch bicubic interpolation (antialias off)",
                               rcan=self.rcan.provenance, sen2sr=SEN2SRLite.provenance)

    def __call__(self, method, image):
        if method == "bicubic":
            return L1BSRSR.bicubic_upscale(image)
        if method == "rcan":
            return self.rcan.super_resolve(image)
        if method == "sen2sr":
            return self.sen2sr.super_resolve(image)
        if method == "nearest":
            return np.repeat(np.repeat(image, 2, 0), 2, 1)
        raise ValueError(f"Unknown method {method}")


def as_u16(array):
    return np.clip(np.round(array), 0, 65535).astype(np.uint16)


def scenes():
    """All benchmark crops plus the five demo crops. None of them trained either SR model."""
    items = []
    for path in sorted(BENCHMARK.glob("*.npz")):
        data = np.load(path)
        items.append((path.stem, data["image"], data["valid"]))
    from geo_io import load_any_user_image
    for path in sorted((ROOT/"demo_images").glob("sentinel2_*.tif")):
        bands, image, _, _ = load_any_user_image(path)
        items.append(("demo_"+path.stem.split("_", 1)[1].rsplit("_", 1)[0], image, bands["B02"].valid_mask))
    return items


def evaluate_scene(up, image, valid):
    h, w = image.shape[0]//2*2, image.shape[1]//2*2
    image, valid = image[:h, :w], valid[:h, :w]
    reference = image.astype(np.float32)
    low = as_u16(block_mean(image, 2))
    low_valid = block_mean(valid.astype(np.float32), 2) > .999
    full_valid = np.repeat(np.repeat(low_valid, 2, 0), 2, 1) & valid
    row = dict(reduced_scale={}, native={})
    native_edges = {}
    for method in METHODS:
        rebuilt = up(method, low).astype(np.float32)
        row["reduced_scale"][method] = reconstruction_metrics(rebuilt, reference, valid_mask=full_valid)
        enhanced = up(method, image)
        back = block_mean(enhanced, 2).astype(np.float32)
        enhanced_valid = np.repeat(np.repeat(valid, 2, 0), 2, 1)
        native_edges[method] = calculate_edge_energy(enhanced, enhanced_valid)
        row["native"][method] = dict(consistency=reconstruction_metrics(back, reference, valid_mask=valid))
    for method in METHODS:
        row["native"][method]["edge_energy_vs_bicubic"] = round(native_edges[method]/(native_edges["bicubic"]+1e-9), 3)
    return row


def summarise(rows):
    summary = {}
    for protocol, keys in (("reduced_scale", ("psnr_db", "ssim", "sam_degrees")),):
        summary[protocol] = {m: {k: round(float(np.mean([r[protocol][m][k] for r in rows])), 4) for k in keys} for m in METHODS}
        for m in METHODS:
            gains = [r[protocol][m]["psnr_db"]-r[protocol]["bicubic"]["psnr_db"] for r in rows]
            summary[protocol][m]["psnr_gain_vs_bicubic_db"] = round(float(np.mean(gains)), 3)
            summary[protocol][m]["scenes_better_than_bicubic"] = int(sum(g > 0 for g in gains))
    summary["native"] = {m: dict(
        consistency_psnr_db=round(float(np.mean([r["native"][m]["consistency"]["psnr_db"] for r in rows])), 3),
        consistency_sam_degrees=round(float(np.mean([r["native"][m]["consistency"]["sam_degrees"] for r in rows])), 4),
        edge_energy_vs_bicubic=round(float(np.mean([r["native"][m]["edge_energy_vs_bicubic"] for r in rows])), 3))
        for m in METHODS}
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", default=str(ROOT/"results"/"sr_comparison.json"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--rcan", default=str(ROOT/"checkpoints"/"demo"/"rcan.pt"),
                        help="RCAN checkpoint to score, e.g. a retrained checkpoints/real/rcan.pt")
    args = parser.parse_args()
    up = Upsamplers(args.device, rcan_path=args.rcan)
    rows = {}
    for name, image, valid in scenes():
        rows[name] = evaluate_scene(up, image, valid)
        r = rows[name]["reduced_scale"]
        print(f"{name:28s} PSNR bicubic {r['bicubic']['psnr_db']:6.2f}  RCAN {r['rcan']['psnr_db']:6.2f}  "
              f"SEN2SR {r['sen2sr']['psnr_db']:6.2f}", flush=True)
    report = dict(description=__doc__.split("\n\n")[1].strip(), methods=up.provenance, rcan_checkpoint=args.rcan,
                  data_range=10000, scenes=len(rows), summary=summarise(list(rows.values())), per_scene=rows)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
