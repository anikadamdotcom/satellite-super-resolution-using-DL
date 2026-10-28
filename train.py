"""Train RCAN and U-Net, select on validation, and evaluate untouched test scenes."""
import argparse
import json
import random
import importlib.metadata
import hashlib
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from data import SatelliteDataset
from models.rcan import RCAN
from models.unet_segmentation import SatelliteSegmentationNet
from models.common import BAND_ORDER, NORMALIZATION
from models.evaluation_metrics import reconstruction_metrics, segmentation_metrics
from geo_io import _atomic_write


def save_checkpoint(path, model, task, config, source, epoch):
    checkpoint = dict(format_version=1, task=task, model_config=config,
        state_dict={k: v.detach().cpu() for k, v in model.state_dict().items()},
        band_order=BAND_ORDER, normalization=NORMALIZATION,
        data_source=source, epoch=epoch)
    _atomic_write(Path(path), lambda temporary: torch.save(checkpoint, temporary))


@torch.inference_mode()
def validate(model, loader, task, device):
    model.eval()
    loss, count = 0.0, 0
    for lr, hr, features, mask in loader:
        target = hr.to(device) if task == "sr" else mask.to(device)
        output = model((lr if task == "sr" else features).to(device))
        value = torch.nn.functional.l1_loss(output, target) if task == "sr" else torch.nn.functional.cross_entropy(output, target, ignore_index=255)
        if not torch.isfinite(value):
            raise RuntimeError("Non-finite validation loss; check input data")
        weight = target.numel() if task == "sr" else int((target != 255).sum())
        loss += value.item()*weight
        count += weight
    return loss/count


def train_task(task, train_loader, val_loader, args, device, source):
    config = dict(n_colors=4, scale=2, n_feats=32, n_resgroups=3, n_resblocks=4, residual=True) if task == "sr" else dict(in_channels=8, num_classes=5)
    model = (RCAN(**config) if task == "sr" else SatelliteSegmentationNet(**config)).to(device)
    if task == "sr":
        # Start at the bicubic baseline and learn the residual through gradient descent.
        torch.nn.init.zeros_(model.tail[-1].weight)
        torch.nn.init.zeros_(model.tail[-1].bias)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    history, best = [], float("inf")
    filename = Path(args.output)/("rcan.pt" if task == "sr" else "unet.pt")
    for epoch in range(1, args.epochs+1):
        model.train()
        total, count = 0.0, 0
        for lr, hr, features, mask in train_loader:
            optimizer.zero_grad(set_to_none=True)
            output = model((lr if task == "sr" else features).to(device))
            target = hr.to(device) if task == "sr" else mask.to(device)
            loss = torch.nn.functional.l1_loss(output, target) if task == "sr" else torch.nn.functional.cross_entropy(output, target, ignore_index=255)
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite training loss; check dataset scaling and labels.")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            weight = target.numel() if task == "sr" else int((target != 255).sum())
            total += loss.item()*weight
            count += weight
        val_loss = validate(model, val_loader, task, device)
        history.append(dict(epoch=epoch, train_loss=total/count, val_loss=val_loss))
        if val_loss < best:
            best = val_loss
            save_checkpoint(filename, model, task, config, source, epoch)
        print(f"{task} epoch {epoch:03d}/{args.epochs}: train={total/count:.5f} val={val_loss:.5f}", flush=True)
    return history


def compensated_l1(output, target, radius=2):
    """Per-patch L1 after the best integer shift (|d| <= radius HR px) and per-band mean offset.

    Real cross-sensor pairs (Sentinel-2 vs VENµS) are misregistered by about one 5 m pixel
    and differ slightly in band response. Plain L1 would teach the network to shift and
    recolour the image; this loss scores only spatial detail (cf. PROBA-V cPSNR).
    Returns one value per patch.
    """
    h, w = target.shape[-2:]
    core = target[..., radius:h-radius, radius:w-radius]
    best = None
    for dy in range(-radius, radius+1):
        for dx in range(-radius, radius+1):
            diff = output[..., radius+dy:h-radius+dy, radius+dx:w-radius+dx] - core
            loss = (diff - diff.mean((2, 3), keepdim=True)).abs().mean((1, 2, 3))
            best = loss if best is None else torch.minimum(best, loss)
    return best


def sr_objective(output, lr, hr, shift, args):
    """Training/validation loss for --task sr.

    "aligned" (default): L1 after the patch's known sub-pixel VENµS/Sentinel-2 shift and
    per-band mean offset are removed (models/alignment.py). Integer-only compensation leaves
    up to half a pixel of misregistration, which rewards blur. Because the aligned loss ignores
    band means, the consistency term (smoothly downsampled output vs. the Sentinel-2 input)
    anchors brightness and large-scale contrast to the measured input.
    """
    from models.alignment import aligned_l1
    if args.loss == "aligned":
        main = aligned_l1(output, hr, shift).mean()
    elif args.loss == "compensated":
        main = compensated_l1(output, hr, args.shift_radius).mean()
    else:
        main = torch.nn.functional.l1_loss(output, hr)
    if args.lr_consistency > 0:
        # Smooth (antialiased) downsampling approximates the Sentinel-2 footprint better than a 2x2 box.
        down = torch.nn.functional.interpolate(output, scale_factor=.5, mode="bicubic", align_corners=False, antialias=True)
        # RMSE (not L1): anchors the mean exactly, so overall brightness cannot drift.
        main = main + args.lr_consistency*torch.sqrt(torch.nn.functional.mse_loss(down, lr) + 1e-12)
    return main


@torch.inference_mode()
def evaluate_sr(model, loader, device, args, radius=2):
    """Mean raw and aligned (sub-pixel shift + band offset removed) L1/PSNR for model and bicubic."""
    from models.alignment import aligned_l1, shift_image
    model.eval()
    sums = {}
    count = 0
    for lr, hr, shift in loader:
        lr, hr, shift = (t.to(device, non_blocking=True) for t in (lr, hr, shift))
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=args.amp and device.type == "cuda"):
            output = model(lr)
        output = output.float().clamp(0, 6.5535)
        bicubic = torch.nn.functional.interpolate(lr, scale_factor=2, mode="bicubic", align_corners=False).clamp(0, 6.5535)
        for name, prediction in (("model", output), ("bicubic", bicubic)):
            raw_mse = ((prediction - hr)**2).mean((1, 2, 3))
            diff = (shift_image(prediction, shift) - hr)[..., 4:-4, 4:-4]
            aligned_mse = ((diff - diff.mean((2, 3), keepdim=True))**2).mean((1, 2, 3))
            values = dict(l1=(prediction - hr).abs().mean((1, 2, 3)), aligned_l1=aligned_l1(prediction, hr, shift),
                          psnr=10*torch.log10(1/raw_mse.clamp(min=1e-12)),
                          aligned_psnr=10*torch.log10(1/aligned_mse.clamp(min=1e-12)))
            for key, value in values.items():
                sums[f"{name}_{key}"] = sums.get(f"{name}_{key}", 0.0) + float(value.sum())
            if name == "model":
                sums["objective"] = sums.get("objective", 0.0) + float(sr_objective(prediction, lr, hr, shift, args))*len(lr)
        count += len(lr)
    return {key: value/count for key, value in sums.items()}


def train_sr_patches(args, device):
    """--task sr: RCAN only, on paired LR/HR patches (e.g. SEN2VENµS), no land-cover masks."""
    import csv
    import time
    from data import SRPatchDataset
    train_set = SRPatchDataset(args.data, "train", crop=args.crop, augment=True)
    val_set = SRPatchDataset(args.data, "val")
    test_set = SRPatchDataset(args.data, "test")
    if args.val_limit and len(val_set) > args.val_limit:
        val_set = torch.utils.data.Subset(val_set, np.linspace(0, len(val_set)-1, args.val_limit).astype(int).tolist())
    sites = {}
    for split in ("train", "val", "test"):
        with open(Path(args.data)/split/"patches.csv", encoding="utf-8") as handle:
            sites[split] = sorted({row["site"] for row in csv.DictReader(handle)})
    source = (f"SEN2VENµS real Sentinel-2 10 m / VENµS 5 m pairs (train sites: {', '.join(sites['train'])}; "
              f"val: {', '.join(sites['val'])}; test: {', '.join(sites['test'])})")
    workers = dict(num_workers=args.workers, pin_memory=device.type == "cuda", persistent_workers=args.workers > 0)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, drop_last=True, **workers)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, **workers)
    config = dict(n_colors=4, scale=2, n_feats=args.n_feats, n_resgroups=args.n_resgroups,
                  n_resblocks=args.n_resblocks, residual=True, consistent=args.consistent)
    if config["n_feats"]**2*config["n_resgroups"]*config["n_resblocks"] > 2_000_000:
        raise SystemExit("Architecture exceeds the app's inference budget (read_checkpoint)")
    model = RCAN(**config).to(device)
    torch.nn.init.zeros_(model.tail[-1].weight)   # start exactly at bicubic
    torch.nn.init.zeros_(model.tail[-1].bias)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    total_steps = args.epochs*len(train_loader)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, total_steps=total_steps,
                                                    pct_start=min(.05, 2000/total_steps), anneal_strategy="cos",
                                                    div_factor=10, final_div_factor=100)
    output = Path(args.output)
    history, best, best_epoch, start = [], float("inf"), 0, 1
    last = output/"last_state.pt"
    if args.resume and last.is_file():
        state = torch.load(last, map_location="cpu", weights_only=True)
        model.load_state_dict(state["model"]); optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        history, best, start = state["history"], state["best"], state["epoch"]+1
        best_epoch = min(history, key=lambda h: h["val_loss"])["epoch"] if history else 0
        print(f"resumed after epoch {state['epoch']}", flush=True)
    for epoch in range(start, args.epochs+1):
        model.train()
        began, total, count = time.time(), 0.0, 0
        for lr, hr, shift in train_loader:
            lr, hr, shift = (t.to(device, non_blocking=True) for t in (lr, hr, shift))
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=args.amp and device.type == "cuda"):
                prediction = model(lr)
            loss = sr_objective(prediction.float(), lr, hr, shift, args)
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite training loss; check dataset scaling.")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total += loss.item()*len(lr)
            count += len(lr)
        val = evaluate_sr(model, val_loader, device, args)
        history.append(dict(epoch=epoch, train_loss=total/count, val_loss=val["objective"], lr=scheduler.get_last_lr()[0],
                            seconds=round(time.time()-began, 1), **{f"val_{k}": v for k, v in val.items()}))
        if val["objective"] < best:
            best, best_epoch = val["objective"], epoch
            save_checkpoint(output/"rcan.pt", model, "sr", config, source, epoch)
        torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
                        history=history, best=best, epoch=epoch), last)
        (output/"history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        print(f"sr epoch {epoch:03d}/{args.epochs}: train={total/count:.5f} val={val['objective']:.5f} "
              f"aligned PSNR model/bicubic={val['model_aligned_psnr']:.2f}/{val['bicubic_aligned_psnr']:.2f} "
              f"raw PSNR model/bicubic={val['model_psnr']:.2f}/{val['bicubic_psnr']:.2f} ({time.time()-began:.0f}s)", flush=True)
        if args.early_stop and epoch - best_epoch >= args.early_stop:
            print(f"early stop: no validation improvement since epoch {best_epoch}", flush=True)
            break
    # Test sites are scored once, with the checkpoint selected on validation.
    from models.common import read_checkpoint
    checkpoint = read_checkpoint(output/"rcan.pt", "sr")
    model.load_state_dict(checkpoint["state_dict"])
    test = evaluate_sr(model, DataLoader(test_set, batch_size=args.batch_size, **workers), device, args)
    report = dict(task="sr", data_source=source, sites=sites, model_config=config, selected_epoch=checkpoint["epoch"],
                  settings={k: v for k, v in vars(args).items()}, split_sizes=dict(train=len(train_set), val=len(val_set), test=len(test_set)),
                  history=dict(sr=history), test=test,
                  evaluation_protocol=("Select by validation objective; evaluate held-out test sites once. Metrics on full "
                                       "128->256 px patches, reflectance scale 1.0 = 10000. 'aligned' = the patch's sub-pixel VENµS/Sentinel-2 "
                                       "shift (estimated once from bicubic vs. VENµS, independent of the method) and per-band mean "
                                       "offset removed, 4 px border excluded; applied identically to every method."))
    report["environment"] = {key: importlib.metadata.version(key) for key in ("torch", "numpy")}
    (output/"training_report.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(test, indent=2), flush=True)


def preflight_datasets(datasets):
    """Fail before training on bad patches, mixed batch sizes or exact split leakage."""
    seen = {}
    for split, dataset in datasets.items():
        shape = None
        for index in range(len(dataset)):
            lr, hr, mask = dataset.arrays(index)
            if shape is None:
                shape = hr.shape
            if hr.shape != shape:
                raise ValueError(f"{split} patches must have the same dimensions for batching")
            if not dataset.demo:
                fingerprint = hashlib.sha256(hr.tobytes()).digest()
                if fingerprint in seen and seen[fingerprint] != split:
                    raise ValueError(f"Identical HR imagery appears in {seen[fingerprint]} and {split}; split leakage")
                seen[fingerprint] = split


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--demo", action="store_true")
    source.add_argument("--data", help="Directory containing train/, val/, test/ NPZ scenes")
    parser.add_argument("--output", default=None)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument("--size", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    sr = parser.add_argument_group("--task sr (paired LR/HR patches from prepare_sen2venus.py, no masks)")
    sr.add_argument("--task", choices=("both", "sr"), default="both")
    sr.add_argument("--crop", type=int, default=96, help="Random LR training crop in pixels")
    sr.add_argument("--workers", type=int, default=4)
    sr.add_argument("--n-feats", type=int, default=32)
    sr.add_argument("--n-resgroups", type=int, default=3)
    sr.add_argument("--n-resblocks", type=int, default=4)
    sr.add_argument("--loss", choices=("aligned", "compensated", "l1"), default="aligned")
    sr.add_argument("--shift-radius", type=int, default=2)
    sr.add_argument("--lr-consistency", type=float, default=0.2, help="Weight of the input-consistency loss term")
    sr.add_argument("--consistent", action="store_true",
                    help="Exact 2x2-mean consistency layer (tested: stays below bicubic on validation)")
    sr.add_argument("--early-stop", type=int, default=0, help="Stop after N epochs without validation improvement (0 = off)")
    sr.add_argument("--val-limit", type=int, default=2000, help="Evenly spaced validation patches (0 = all)")
    sr.add_argument("--no-amp", dest="amp", action="store_false")
    sr.add_argument("--deterministic", action="store_true")
    sr.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.epochs < 1 or args.samples < 1 or args.batch_size < 1 or not np.isfinite(args.lr) or args.lr <= 0:
        parser.error("epochs, samples, batch-size and lr must be positive")
    if args.task == "sr" and not args.data:
        parser.error("--task sr requires --data")
    args.output = args.output or ("checkpoints/demo" if args.demo else "checkpoints/real")
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    device = torch.device(args.device)
    if args.task == "sr":
        torch.backends.cudnn.deterministic = args.deterministic
        torch.backends.cudnn.benchmark = not args.deterministic
        Path(args.output).mkdir(parents=True, exist_ok=True)
        return train_sr_patches(args, device)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    Path(args.output).mkdir(parents=True, exist_ok=True)
    datasets = {split: SatelliteDataset(args.data, split, args.demo,
                 args.samples if split == "train" else max(16, args.samples//4), args.size)
                for split in ("train", "val", "test")}
    preflight_datasets(datasets)
    loaders = {split: DataLoader(ds, batch_size=args.batch_size, shuffle=split == "train", num_workers=0)
               for split, ds in datasets.items()}
    source_name = "synthetic procedural demo (not real satellite validation)" if args.demo else "user supplied NPZ scenes (verify source independently)"
    history = {task: train_task(task, loaders["train"], loaders["val"], args, device, source_name)
               for task in ("sr", "segmentation")}
    # Select only by validation loss. Test scenes are evaluated once at the end.
    from models.rcan import L1BSRSR
    from models.unet_segmentation import LandCoverSegmenter
    sr = L1BSRSR(Path(args.output)/"rcan.pt", str(device))
    seg = LandCoverSegmenter(str(device), Path(args.output)/"unet.pt")
    recon, baseline, confusion = [], [], np.zeros((5, 5), dtype=np.int64)
    direct_confusion = np.zeros((5, 5), dtype=np.int64)
    for index in range(len(datasets["test"])):
        lr, hr, mask = datasets["test"].arrays(index)
        prediction = sr.super_resolve(lr)
        recon.append(reconstruction_metrics(prediction, hr, max_val=NORMALIZATION))
        baseline.append(reconstruction_metrics(sr.bicubic_upscale(lr), hr, max_val=NORMALIZATION))
        confusion += np.array(segmentation_metrics(seg.segment(prediction, apply_morphology=False, return_probabilities=False)["class_map"], mask)["confusion_matrix"])
        direct_confusion += np.array(segmentation_metrics(seg.segment(hr, apply_morphology=False, return_probabilities=False)["class_map"], mask)["confusion_matrix"])
    report = dict(data_source=source_name, settings=vars(args), split_sizes={k:len(v) for k,v in datasets.items()},
        history=history, test=dict(sr={k:float(np.mean([v[k] for v in recon])) for k in recon[0]},
        bicubic={k:float(np.mean([v[k] for v in baseline])) for k in baseline[0]},
        segmentation_pipeline=segmentation_metrics(confusion=confusion),
        segmentation_on_hr=segmentation_metrics(confusion=direct_confusion)))
    report["environment"] = {key: importlib.metadata.version(key) for key in
                             ("torch", "numpy", "rasterio", "PyQt5", "Pillow", "scipy")}
    report["evaluation_protocol"] = "Select by validation loss; evaluate held-out test once; no morphology."
    (Path(args.output)/"training_report.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(report["test"], indent=2), flush=True)


if __name__ == "__main__":
    main()
