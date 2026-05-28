"""
visualize_gradcam.py – Generate Grad-CAM heatmaps from a trained model.

Usage:
    python visualize_gradcam.py --model_path ./outputs/best_model.pth --data_dir ./dataset

Outputs saved to --output_dir:
    gradcam_grid.png          – overview grid (all classes, n_per_class examples each)
    <ClassName>/sample_N.png  – individual original + overlay side-by-side
"""

import argparse
from collections import defaultdict
from pathlib import Path

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

from src.config import Config
from src.dataset import apply_clahe, IMAGENET_MEAN, IMAGENET_STD
from src.gradcam import GradCAM, overlay_heatmap
from src.model import create_model, get_gradcam_target_layer

matplotlib.use("Agg")


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Grad-CAM Visualisation")
    p.add_argument("--model_path",   type=str, required=True,
                   help="Path to .pth checkpoint (saved by train.py)")
    p.add_argument("--data_dir",     type=str, default="./dataset")
    p.add_argument("--output_dir",   type=str, default="./outputs/gradcam")
    p.add_argument("--n_per_class",  type=int, default=4,
                   help="Number of example images per class in the grid")
    p.add_argument("--img_size",     type=int, default=512)
    p.add_argument("--no_clahe",     action="store_true",
                   help="Disable CLAHE (should match training settings)")
    p.add_argument("--phase",        type=str, default="test",
                   choices=["train", "val", "test"])
    return p.parse_args()


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def load_model(model_path: str, device: torch.device):
    ckpt = torch.load(model_path, map_location=device)
    cfg_dict   = ckpt.get("cfg", {})
    model_name = cfg_dict.get("model_name", "tf_efficientnetv2_s")
    num_classes= cfg_dict.get("num_classes", 4)

    model = create_model(model_name, num_classes=num_classes, pretrained=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    print(f"  Loaded {model_name}  (val F1 = {ckpt.get('val_f1', '?'):.4f})")
    return model, model_name, cfg_dict


def prepare_image(img_path: str, img_size: int, use_clahe: bool, device: torch.device):
    """
    Returns:
        display_rgb : (H, W, 3) uint8 for matplotlib display
        inp         : (1, 3, H, W) normalised tensor for model
    """
    gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise IOError(f"Cannot read image: {img_path}")

    if use_clahe:
        gray = apply_clahe(gray)

    display_rgb = cv2.cvtColor(cv2.resize(gray, (img_size, img_size)), cv2.COLOR_GRAY2RGB)

    transform = T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    inp = transform(Image.fromarray(display_rgb)).unsqueeze(0).to(device)
    return display_rgb, inp


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_clahe = not args.no_clahe

    print(f"\n{'═'*55}")
    print(f"  Grad-CAM Visualisation")
    print(f"{'═'*55}")
    print(f"  Device     : {device}")
    print(f"  Checkpoint : {args.model_path}")
    print(f"  Phase      : {args.phase}")
    print(f"  CLAHE      : {use_clahe}")

    # ── Load model ───────────────────────────────────────────
    model, model_name, cfg_dict = load_model(args.model_path, device)
    target_layer = get_gradcam_target_layer(model, model_name)
    gradcam      = GradCAM(model, target_layer)

    # ── Collect images by class ───────────────────────────────
    class_names = cfg_dict.get("class_names",
                               ["Biliary_Leaks", "Lithiasis", "Normal", "Stricture"])
    phase_dir   = Path(args.data_dir) / args.phase

    by_class = defaultdict(list)
    for cls_idx, cls_name in enumerate(class_names):
        cls_dir = phase_dir / cls_name
        if not cls_dir.exists():
            print(f"  [WARNING] Folder not found: {cls_dir}")
            continue
        for p in sorted(cls_dir.iterdir()):
            if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}:
                by_class[cls_idx].append(str(p))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    n_per   = args.n_per_class
    n_cls   = len(class_names)
    img_sz  = args.img_size

    # ── Overview grid ─────────────────────────────────────────
    fig, axes = plt.subplots(n_cls, n_per * 2, figsize=(n_per * 2 * 3.2, n_cls * 3.5))
    if n_cls == 1:
        axes = axes[np.newaxis, :]

    for row, cls_idx in enumerate(range(n_cls)):
        paths = by_class[cls_idx][:n_per]
        for col, img_path in enumerate(paths):
            try:
                display_rgb, inp = prepare_image(img_path, img_sz, use_clahe, device)
                cam, pred_idx    = gradcam.generate(inp, class_idx=cls_idx)
                overlay          = overlay_heatmap(display_rgb, cam)
                correct          = pred_idx == cls_idx
            except Exception as e:
                print(f"  [ERROR] {img_path}: {e}")
                axes[row, col * 2].axis("off")
                axes[row, col * 2 + 1].axis("off")
                continue

            color = "limegreen" if correct else "red"

            ax_o = axes[row, col * 2]
            ax_o.imshow(display_rgb, cmap="gray")
            ax_o.axis("off")
            if col == 0:
                ax_o.set_ylabel(class_names[cls_idx], fontsize=10, fontweight="bold")
            if row == 0:
                ax_o.set_title(f"Original {col+1}", fontsize=9)

            ax_c = axes[row, col * 2 + 1]
            ax_c.imshow(overlay)
            ax_c.axis("off")
            for spine in ax_c.spines.values():
                spine.set_visible(True)
                spine.set_edgecolor(color)
                spine.set_linewidth(2.5)
            if row == 0:
                ax_c.set_title(f"Grad-CAM {col+1}", fontsize=9)
            pred_name = class_names[pred_idx] if pred_idx < n_cls else "?"
            ax_c.set_xlabel(
                f"{'✓' if correct else '✗'}  {pred_name}",
                fontsize=7, color=color, labelpad=2,
            )

        for col in range(len(paths), n_per):
            axes[row, col * 2].axis("off")
            axes[row, col * 2 + 1].axis("off")

    plt.suptitle(
        f"Grad-CAM — {model_name}   "
        f"(green = correct  |  red = incorrect prediction)",
        fontsize=12, fontweight="bold", y=1.01,
    )
    plt.tight_layout()
    grid_path = output_dir / "gradcam_grid.png"
    plt.savefig(str(grid_path), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Grid saved → {grid_path}")

    # ── Individual per-class images ───────────────────────────
    for cls_idx, cls_name in enumerate(class_names):
        cls_out = output_dir / cls_name
        cls_out.mkdir(exist_ok=True)

        for i, img_path in enumerate(by_class[cls_idx][:n_per]):
            try:
                display_rgb, inp = prepare_image(img_path, img_sz, use_clahe, device)
                cam, pred_idx    = gradcam.generate(inp, class_idx=cls_idx)
                overlay          = overlay_heatmap(display_rgb, cam)
                correct          = pred_idx == cls_idx
            except Exception as e:
                print(f"  [ERROR] {img_path}: {e}")
                continue

            fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4.5))

            ax1.imshow(display_rgb, cmap="gray")
            ax1.set_title("Original (CLAHE)" if use_clahe else "Original", fontsize=11)
            ax1.axis("off")

            ax2.imshow(overlay)
            status = "✓ correct" if correct else f"✗ predicted: {class_names[pred_idx]}"
            ax2.set_title(f"Grad-CAM  [{status}]", fontsize=11,
                          color="green" if correct else "red")
            ax2.axis("off")

            plt.suptitle(f"True class: {cls_name}", fontsize=12, fontweight="bold")
            plt.tight_layout()

            save_path = cls_out / f"sample_{i + 1:02d}.png"
            plt.savefig(str(save_path), dpi=130, bbox_inches="tight")
            plt.close()

        print(f"  {cls_name:<22} → {n_per} images saved to {cls_out}")

    gradcam.remove_hooks()
    print(f"\n  Done.  All Grad-CAM outputs in: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
