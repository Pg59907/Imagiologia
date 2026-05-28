"""
explore_dataset.py – Exploratory Data Analysis (EDA) for the ERCP dataset.

Generates figures required for the report:
  - Class distribution bar chart (all splits)
  - Sample image grid (with and without CLAHE)

Usage:
    python explore_dataset.py --data_dir ./dataset --output_dir ./outputs/eda
"""

import argparse
from pathlib import Path

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from src.dataset import ERCPDataset, apply_clahe
from src.config import Config

matplotlib.use("Agg")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir",   type=str, default="./dataset")
    p.add_argument("--output_dir", type=str, default="./outputs/eda")
    p.add_argument("--n_samples",  type=int, default=3,
                   help="Number of sample images per class for the image grid")
    return p.parse_args()


def plot_class_distribution(datasets: dict, class_names: list, output_path: str):
    """Bar chart of class sizes per split."""
    phases = list(datasets.keys())
    x      = np.arange(len(class_names))
    width  = 0.25
    colors = ["steelblue", "darkorange", "seagreen"]

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, (phase, color) in enumerate(zip(phases, colors)):
        dist   = datasets[phase].class_distribution()
        counts = [dist.get(c, 0) for c in class_names]
        bars   = ax.bar(x + i * width, counts, width, label=phase.capitalize(), color=color, alpha=0.85)
        for bar, count in zip(bars, counts):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                    str(count), ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x + width)
    ax.set_xticklabels(class_names, rotation=20, ha="right", fontsize=11)
    ax.set_ylabel("Number of images", fontsize=12)
    ax.set_title("Class Distribution per Split", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[EDA] Distribution chart saved → {output_path}")


def plot_sample_grid(train_ds: ERCPDataset, class_names: list,
                     n_samples: int, output_path: str):
    """
    Grid showing raw image vs CLAHE-enhanced image for each class.
    Layout: rows = classes, cols = [raw, clahe] × n_samples
    """
    from collections import defaultdict
    by_class = defaultdict(list)
    for path, lbl in zip(train_ds.image_paths, train_ds.labels):
        by_class[lbl].append(path)

    n_cls = len(class_names)
    fig, axes = plt.subplots(n_cls, n_samples * 2, figsize=(n_samples * 2 * 3, n_cls * 3.2))
    if n_cls == 1:
        axes = axes[np.newaxis, :]

    for row, cls_idx in enumerate(range(n_cls)):
        paths = by_class[cls_idx][:n_samples]
        for col, img_path in enumerate(paths):
            gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if gray is None:
                continue
            gray_clahe = apply_clahe(gray)

            ax_r = axes[row, col * 2]
            ax_r.imshow(cv2.resize(gray, (256, 256)), cmap="gray")
            ax_r.axis("off")
            if col == 0:
                ax_r.set_ylabel(class_names[cls_idx], fontsize=10, fontweight="bold")
            if row == 0:
                ax_r.set_title(f"Raw {col+1}", fontsize=9)

            ax_c = axes[row, col * 2 + 1]
            ax_c.imshow(cv2.resize(gray_clahe, (256, 256)), cmap="gray")
            ax_c.axis("off")
            if row == 0:
                ax_c.set_title(f"CLAHE {col+1}", fontsize=9)

    plt.suptitle("Sample Images: Raw vs CLAHE Enhancement", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[EDA] Sample grid saved → {output_path}")


def main():
    args = parse_args()
    out  = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Minimal config (no augmentation needed for EDA)
    cfg = Config(data_dir=args.data_dir, use_clahe=False)

    datasets = {}
    for phase in ["train", "val", "test"]:
        try:
            datasets[phase] = ERCPDataset(cfg.data_dir, phase, cfg)
        except FileNotFoundError as e:
            print(f"  [SKIP] {e}")

    if not datasets:
        print("No dataset splits found.  Check --data_dir.")
        return

    class_names = list(datasets.values())[0].class_names

    print("\n  Class distribution:")
    for phase, ds in datasets.items():
        dist  = ds.class_distribution()
        total = sum(dist.values())
        print(f"    {phase:>5} ({total:4d})  " +
              "  ".join(f"{c}: {n}" for c, n in dist.items()))

    plot_class_distribution(datasets, class_names, str(out / "class_distribution.png"))

    if "train" in datasets:
        plot_sample_grid(datasets["train"], class_names, args.n_samples,
                         str(out / "sample_images_clahe.png"))

    print(f"\n  EDA outputs saved to: {out.resolve()}")


if __name__ == "__main__":
    main()
