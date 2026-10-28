"""Create a shareable figure from the recorded training run."""
import _bootstrap  # noqa: F401  (puts src/ on sys.path)
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", default="checkpoints/demo/training_report.json")
    parser.add_argument("--output", default="outputs/demo/training_results.png")
    args = parser.parse_args()
    report = json.loads(Path(args.report).read_text())
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
    fig.suptitle("Training and held-out evaluation\n" + report["data_source"], fontsize=14)
    for ax, task, title in zip(axes[0], ("sr", "segmentation"), ("RCAN: L1 reconstruction loss", "U-Net: cross-entropy loss")):
        rows = report["history"][task]
        ax.plot([r["epoch"] for r in rows], [r["train_loss"] for r in rows], label="Training")
        ax.plot([r["epoch"] for r in rows], [r["val_loss"] for r in rows], label="Validation")
        ax.set(title=title, xlabel="Epoch", ylabel="Loss"); ax.legend(); ax.grid(alpha=.2)
    tests = report["test"]
    axes[1, 0].bar(["Bicubic", "Trained RCAN"], [tests["bicubic"]["psnr_db"], tests["sr"]["psnr_db"]], color=["#8596ab", "#2a9d8f"])
    axes[1, 0].set(title="Reconstruction against held-out HR truth", ylabel="PSNR (dB)",
                   ylim=(0, max(45, tests["sr"]["psnr_db"]+5, tests["bicubic"]["psnr_db"]+5)))
    for i, value in enumerate([tests["bicubic"]["psnr_db"], tests["sr"]["psnr_db"]]):
        axes[1, 0].text(i, value+.5, f"{value:.2f}", ha="center")
    matrix = np.array(tests["segmentation_pipeline"]["confusion_matrix"], dtype=float)
    matrix /= np.maximum(matrix.sum(axis=1, keepdims=True), 1)
    ax = axes[1, 1]; ax.imshow(matrix, vmin=0, vmax=1, cmap="Blues")
    names = ["Buildings", "Roads", "Vegetation", "Water", "Bare land"]
    ax.set(xticks=range(5), yticks=range(5), xticklabels=names, yticklabels=names,
           title=f"RCAN -> U-Net test mIoU: {tests['segmentation_pipeline']['mean_iou']:.2%}",
           xlabel="Predicted class", ylabel="True class")
    ax.tick_params(axis="x", labelrotation=30)
    for y in range(5):
        for x in range(5):
            ax.text(x, y, f"{matrix[y,x]:.0%}", ha="center", va="center", color="white" if matrix[y,x]>.5 else "black")
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
