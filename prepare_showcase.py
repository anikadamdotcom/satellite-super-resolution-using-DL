"""Prepare CPU results and an honest overview of the five real demo scenes."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import zipfile
import numpy as np
import torch
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from geo_io import _atomic_write, save_text
from models.unet_segmentation import REAL_CLASS_DEFINITIONS
from predict import run_prediction
from runtime_support import configure_cpu_threads

ROOT = Path(__file__).resolve().parent


def prepare(output):
    configure_cpu_threads()
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    records = json.loads((ROOT/"demo_images"/"manifest.json").read_text(encoding="utf-8-sig"))
    summary = []
    fig, axes = plt.subplots(len(records), 4, figsize=(13, 16), layout="constrained")
    try:
        for row, record in enumerate(records):
            folder = output/record["category"]
            report = run_prediction(SimpleNamespace(output=str(folder), image=str(ROOT/"demo_images"/record["file"]),
                bands=None, demo=False, models=str(ROOT/"checkpoints"/"demo"), device="cpu"))
            assert report["device"] == "cpu" and not report["metrics"]["is_ground_truth"]
            labels = np.load(folder/"class_ids.npy", allow_pickle=False)
            assert labels.shape == (512, 512) and np.isin(labels, [0, 1, 2, 3, 4, 5, 255]).all()
            entry = dict(category=record["category"], location=record["location"], input=record["file"],
                         device="cpu", predicted_percentages={key: value["percentage"] for key,value in report["statistics"].items()},
                         metrics=report["metrics"], classification_accuracy=None,
                         accuracy_note="Real-pretrained OpenEarthMap + measured-NIR checks; no independent Sentinel-2 ground truth.")
            summary.append(entry)
            for col, (name, title) in enumerate((("input.png", "Original: 256 x 256"),
                                                ("enhanced.png", "RCAN: 512 x 512"),
                                                ("land_cover.png", "Predicted classes"),
                                                ("overlay.png", "Predictions on image"))):
                with Image.open(folder/name) as image:
                    axes[row,col].imshow(np.asarray(image), interpolation="nearest")
                axes[row,col].set_xticks([]); axes[row,col].set_yticks([])
                if row == 0:
                    axes[row,col].set_title(title, fontsize=11)
                if col == 0:
                    axes[row,col].set_ylabel(record["category"].title()+"\n"+record["location"], fontsize=10)
            # Include every generated result in one verified offline backup.
            def write_archive(temp, source=folder):
                with zipfile.ZipFile(temp, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                    for file in sorted(source.iterdir()):
                        if file.is_file() and file.suffix != ".zip":
                            archive.write(file, file.name)
            _atomic_write(folder/"results.zip", write_archive)
            with zipfile.ZipFile(folder/"results.zip") as archive:
                if archive.testzip() is not None:
                    raise RuntimeError("Result archive integrity check failed")
        fig.suptitle("Real Sentinel-2 demonstration | CPU | OpenEarthMap land cover + NIR checks\n"
                     "Native-image classification; 2x RCAN display does not establish new observed detail", fontsize=13)
        legend = [Patch(color=info["hex"], label=info["name"]) for info in REAL_CLASS_DEFINITIONS.values()]
        fig.legend(handles=legend, loc="outside lower center", ncols=6, fontsize=10)
        _atomic_write(output/"overview.png", lambda temp: fig.savefig(temp, format="png", dpi=140))
        save_text(output/"summary.json", json.dumps(dict(torch_version=torch.__version__, scenes=summary), indent=2, allow_nan=False))
        readme = ["# Offline showcase results", "", "These outputs use the five genuine Sentinel-2 inputs and CPU inference.",
                  "Land cover uses real-pretrained OpenEarthMap weights and measured-NIR checks. RCAN is trained on real Sentinel-2/VENµS pairs (small measured gain over bicubic).",
                  "Road/building accuracy at Sentinel-2's 10 m resolution is unvalidated. Scores are not calibrated accuracy.", "",
                  "![Original, enhanced, predicted map and overlay](overview.png)", "",
                  "Each scene folder includes original/enhanced PNGs, maps, overlay, scores, arrays, GeoTIFFs, report and results.zip.",
                  "The ZIPs are backups for a live demonstration. summary.json records predicted shares without inventing accuracy.", "",
                  "See ../SHOWCASE_GUIDE.md for the live demonstration and presentation wording.", ""]
        save_text(output/"README.md", "\n".join(readme))
    finally:
        plt.close(fig)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(ROOT/"showcase"))
    args = parser.parse_args()
    prepare(args.output)
