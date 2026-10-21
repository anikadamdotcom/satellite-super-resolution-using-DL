"""Does super-resolution help land-cover mapping? A controlled test on real data.

The same U-Net is trained from scratch, with the same seeds, steps and labels, on
four versions of the same real Sentinel-2 scenes, all on a 5 m grid:

    native   original 10 m pixels repeated 2x (no new information)
    bicubic  bicubic 2x interpolation
    rcan     this project's RCAN 2x
    sen2sr   ESA SEN2SRLite 4x, area-averaged to 5 m

Labels are ESA WorldCover 2021 (10 m), repeated onto the 5 m grid. Train,
validation and test sites are different places (see download_mapping_benchmark.py).
The checkpoint with the best validation mean IoU is scored once on the test sites.

Limits: WorldCover is a model output, not human ground truth, and it is 10 m, so
it cannot reward detail finer than 10 m. Results show whether SR input changes
agreement with a 10 m map; they cannot show sub-10 m mapping gains.

    python mapping_experiment.py --steps 1500 --seeds 0 1 2
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from compare_sr_models import Upsamplers
from download_mapping_benchmark import CLASSES, IGNORE, ROOT
from models.common import extract_features
from models.unet_segmentation import SatelliteSegmentationNet

BENCHMARK = ROOT/"datasets"/"mapping_benchmark"
INPUTS = ("native", "bicubic", "rcan", "sen2sr")
NUM_CLASSES = len(CLASSES)


def load_sites():
    manifest = json.loads((BENCHMARK/"manifest.json").read_text(encoding="utf-8"))
    sites = []
    for record in manifest["sites"]:
        data = np.load(BENCHMARK/f"{record['slug']}.npz")
        sites.append(dict(record=record, image=data["image"], labels=data["labels"], valid=data["valid"]))
    return sites


def prepare_inputs(sites, device, cache):
    """Upsample every site with every method once; cache features as float16."""
    cache = Path(cache)
    cache.mkdir(parents=True, exist_ok=True)
    up = None
    for site in sites:
        site["labels5"] = np.repeat(np.repeat(site["labels"], 2, 0), 2, 1)
        site["features"] = {}
        for method in INPUTS:
            path = cache/f"{site['record']['slug']}_{method}.npy"
            if not path.exists():
                if up is None:
                    up = Upsamplers(device)
                enhanced = up("nearest" if method == "native" else method, site["image"])
                np.save(path, np.moveaxis(extract_features(enhanced)[0], -1, 0).astype(np.float16))
            site["features"][method] = np.load(path, mmap_mode="r")
    return up.provenance if up else None


def class_weights(sites):
    counts = np.bincount(np.concatenate([s["labels"][s["labels"] != IGNORE].ravel() for s in sites]),
                         minlength=NUM_CLASSES).astype(float)
    frequency = counts/counts.sum()
    weights = 1/np.sqrt(np.maximum(frequency, 1e-4))
    return torch.tensor(weights/weights.mean(), dtype=torch.float32), frequency


def batch(sites, method, rng, size, count):
    xs, ys = [], []
    for _ in range(count):
        site = sites[rng.integers(len(sites))]
        h, w = site["labels5"].shape
        y, x = rng.integers(0, h-size+1), rng.integers(0, w-size+1)
        feats = np.asarray(site["features"][method][:, y:y+size, x:x+size], np.float32)
        label = site["labels5"][y:y+size, x:x+size]
        k = int(rng.integers(4))
        feats, label = np.rot90(feats, k, (1, 2)), np.rot90(label, k)
        if rng.random() < .5:
            feats, label = feats[:, :, ::-1], label[:, ::-1]
        xs.append(np.ascontiguousarray(feats)); ys.append(np.ascontiguousarray(label))
    return torch.from_numpy(np.stack(xs)), torch.from_numpy(np.stack(ys).astype(np.int64))


@torch.inference_mode()
def confusion(model, sites, method, device, tile=256):
    model.eval()
    matrix = np.zeros((NUM_CLASSES, NUM_CLASSES), np.int64)
    for site in sites:
        feats = np.asarray(site["features"][method], np.float32)
        label = site["labels5"]
        h, w = label.shape
        for y in range(0, h, tile):
            for x in range(0, w, tile):
                chunk = torch.from_numpy(np.ascontiguousarray(feats[:, y:y+tile, x:x+tile]))[None].to(device)
                pred = model(chunk).argmax(1)[0].cpu().numpy()
                target = label[y:y+tile, x:x+tile]
                keep = target != IGNORE
                matrix += np.bincount(NUM_CLASSES*target[keep].astype(int)+pred[keep],
                                      minlength=NUM_CLASSES**2).reshape(NUM_CLASSES, NUM_CLASSES)
    model.train()
    return matrix


def scores(matrix):
    tp = np.diag(matrix).astype(float)
    predicted, actual = matrix.sum(0), matrix.sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        iou = tp/(predicted+actual-tp)
        f1 = 2*tp/(predicted+actual)
    present = actual > 0
    return dict(overall_accuracy=float(tp.sum()/matrix.sum()), mean_iou=float(np.nanmean(iou[present])),
                mean_f1=float(np.nanmean(f1[present])),
                per_class_f1={c: (None if not present[i] or np.isnan(f1[i]) else round(float(f1[i]), 4)) for i, c in enumerate(CLASSES)},
                per_class_iou={c: (None if not present[i] or np.isnan(iou[i]) else round(float(iou[i]), 4)) for i, c in enumerate(CLASSES)})


def run(method, seed, splits, weights, args, device):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = SatelliteSegmentationNet(in_channels=8, num_classes=NUM_CLASSES).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    schedule = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, total_steps=args.steps)
    best, best_state, history = -1., None, []
    start = time.time()
    for step in range(1, args.steps+1):
        x, y = batch(splits["train"], method, rng, args.patch, args.batch)
        loss = F.cross_entropy(model(x.to(device)), y.to(device), weight=weights.to(device), ignore_index=IGNORE)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step(); schedule.step()
        if step % args.eval_every == 0 or step == args.steps:
            val = scores(confusion(model, splits["val"], method, device))
            history.append(dict(step=step, loss=round(loss.item(), 4), val_mean_iou=round(val["mean_iou"], 4)))
            if val["mean_iou"] > best:
                best, best_state = val["mean_iou"], {k: v.detach().clone() for k, v in model.state_dict().items()}
            print(f"  {method:8s} seed {seed} step {step:5d} loss {loss.item():.3f} val mIoU {val['mean_iou']:.4f} "
                  f"({time.time()-start:.0f}s)", flush=True)
    model.load_state_dict(best_state)
    per_site = {s["record"]["slug"]: confusion(model, [s], method, device) for s in splits["test"]}
    test_matrix = sum(per_site.values())
    return dict(method=method, seed=seed, best_val_mean_iou=round(best, 4), history=history,
                test=scores(test_matrix), test_confusion=test_matrix.tolist(),
                per_site={slug: round(float(np.trace(m)/m.sum()), 4) for slug, m in per_site.items()},
                seconds=round(time.time()-start, 1))


def summarise(runs):
    summary = {}
    for method in INPUTS:
        chosen = [r for r in runs if r["method"] == method]
        if not chosen:
            continue
        entry = {}
        for key in ("overall_accuracy", "mean_iou", "mean_f1"):
            values = np.array([r["test"][key] for r in chosen])
            entry[key] = dict(mean=round(float(values.mean()), 4), std=round(float(values.std()), 4))
        entry["per_class_f1"] = {c: round(float(np.mean([r["test"]["per_class_f1"][c] or 0 for r in chosen])), 4) for c in CLASSES}
        summary[method] = entry
    # Paired differences against bicubic: same seed, same test sites.
    for method in INPUTS:
        if method == "bicubic" or method not in summary or "bicubic" not in summary:
            continue
        diffs = {}
        for key in ("overall_accuracy", "mean_iou", "mean_f1"):
            pairs = [(r["test"][key], b["test"][key]) for r in runs for b in runs
                     if r["method"] == method and b["method"] == "bicubic" and r["seed"] == b["seed"]]
            d = np.array([a-b for a, b in pairs])
            diffs[key] = dict(mean=round(float(d.mean()), 4), min=round(float(d.min()), 4), max=round(float(d.max()), 4))
        summary[method]["difference_vs_bicubic"] = diffs
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--inputs", nargs="+", default=list(INPUTS), choices=INPUTS)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--patch", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--cache", default=str(ROOT/"datasets"/"mapping_benchmark_cache"))
    parser.add_argument("--output", default=str(ROOT/"results"/"mapping_experiment.json"))
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    sites = load_sites()
    provenance = prepare_inputs(sites, args.device, args.cache)
    splits = {s: [x for x in sites if x["record"]["split"] == s] for s in ("train", "val", "test")}
    weights, frequency = class_weights(splits["train"])
    output = Path(args.output)
    previous = json.loads(output.read_text(encoding="utf-8")) if output.exists() else {}
    runs = [r for r in previous.get("runs", []) if r["method"] in args.inputs or r["method"] not in INPUTS]
    done = {(r["method"], r["seed"]) for r in runs if r.get("steps") == args.steps}
    runs = [r for r in runs if r.get("steps") == args.steps]
    for seed in args.seeds:
        for method in args.inputs:
            if (method, seed) in done:
                continue
            print(f"Training on {method} input, seed {seed}", flush=True)
            result = run(method, seed, splits, weights, args, device)
            result["steps"] = args.steps
            runs.append(result)
            report = dict(description=__doc__.split("\n\n")[1].strip(), classes=CLASSES,
                          sites={s: [x["record"]["slug"] for x in v] for s, v in splits.items()},
                          train_class_frequency={c: round(float(f), 4) for c, f in zip(CLASSES, frequency)},
                          settings={k: v for k, v in vars(args).items() if k not in ("cache", "output")},
                          sr_methods=provenance or previous.get("sr_methods"), runs=runs, summary=summarise(runs))
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
            t = result["test"]
            print(f"  -> test OA {t['overall_accuracy']:.4f}  mIoU {t['mean_iou']:.4f}  mean F1 {t['mean_f1']:.4f}", flush=True)
    print(json.dumps(summarise(runs), indent=2))


if __name__ == "__main__":
    main()
