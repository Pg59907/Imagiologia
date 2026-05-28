"""
Gradient-weighted Class Activation Mapping (Grad-CAM).

Implementation using PyTorch forward / backward hooks.
Works for any CNN backbone where a spatial feature-map layer can be identified.

Reference: Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks
via Gradient-based Localization" (ICCV 2017).
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

matplotlib.use("Agg")  # non-interactive backend for saving figures


# ──────────────────────────────────────────────
# Core Grad-CAM class
# ──────────────────────────────────────────────

class GradCAM:
    """
    Hook-based Grad-CAM for CNN models.

    Usage:
        gradcam = GradCAM(model, target_layer)
        cam, pred_class = gradcam.generate(input_tensor)
        gradcam.remove_hooks()   # always clean up after use
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model        = model
        self.activations: Optional[torch.Tensor] = None
        self.gradients:   Optional[torch.Tensor] = None

        # Register hooks on the target layer
        self._fwd_handle = target_layer.register_forward_hook(self._save_activations)
        self._bwd_handle = target_layer.register_full_backward_hook(self._save_gradients)

    # ── Hook callbacks ───────────────────────────────────────

    def _save_activations(self, module, input, output):
        # For most CNN blocks the output is a 4-D tensor [B, C, H, W]
        if isinstance(output, torch.Tensor):
            self.activations = output.detach()
        else:
            # Some blocks return tuples; take the first tensor element
            for o in (output if isinstance(output, (list, tuple)) else [output]):
                if isinstance(o, torch.Tensor) and o.ndim == 4:
                    self.activations = o.detach()
                    break

    def _save_gradients(self, module, grad_input, grad_output):
        g = grad_output[0]
        if isinstance(g, torch.Tensor) and g.ndim == 4:
            self.gradients = g.detach()

    # ── Main generation method ───────────────────────────────

    def generate(
        self,
        input_tensor: torch.Tensor,
        class_idx: Optional[int] = None,
    ) -> Tuple[np.ndarray, int]:
        """
        Generate a Grad-CAM heatmap for the given input.

        Args:
            input_tensor : (1, C, H, W) preprocessed image tensor on the correct device
            class_idx    : class to explain; if None uses the predicted class

        Returns:
            cam       : (H', W') float32 array in [0, 1]  (same spatial size as feature map)
            class_idx : the class that was explained
        """
        self.model.eval()

        # Forward pass
        logits = self.model(input_tensor)

        if class_idx is None:
            class_idx = int(logits.argmax(dim=1).item())

        # Backward pass w.r.t. the chosen class score
        self.model.zero_grad()
        logits[0, class_idx].backward()

        if self.activations is None or self.gradients is None:
            raise RuntimeError(
                "Hooks did not capture activations/gradients. "
                "Make sure the target_layer is actually used in the forward pass."
            )

        # Global average pooling of gradients → importance weights  [1, C, 1, 1]
        weights = self.gradients.mean(dim=[2, 3], keepdim=True)

        # Weighted linear combination of feature maps  [H', W']
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)                     # keep only positive contributions
        cam = cam.squeeze().cpu().float().numpy()

        # Normalize to [0, 1]
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)
        else:
            cam = np.zeros_like(cam)

        return cam, class_idx

    def remove_hooks(self):
        """Remove forward and backward hooks.  Always call this after use."""
        self._fwd_handle.remove()
        self._bwd_handle.remove()


# ──────────────────────────────────────────────
# Visualisation helpers
# ──────────────────────────────────────────────

def overlay_heatmap(
    original_rgb: np.ndarray,
    cam: np.ndarray,
    alpha: float = 0.45,
) -> np.ndarray:
    """
    Overlay a Grad-CAM heatmap on the original image using the JET colormap.

    Args:
        original_rgb : (H, W, 3) uint8 RGB array
        cam          : (H', W') float32 array in [0, 1]
        alpha        : heatmap opacity (0 = invisible, 1 = fully opaque)

    Returns:
        (H, W, 3) uint8 RGB overlay
    """
    h, w = original_rgb.shape[:2]
    cam_resized = cv2.resize(cam, (w, h), interpolation=cv2.INTER_LINEAR)

    # Map CAM values to JET colors  [H, W, 3]  float in [0, 1]
    heatmap_rgb = plt.cm.jet(cam_resized)[:, :, :3]
    heatmap_rgb = (heatmap_rgb * 255).astype(np.float32)

    orig_f = original_rgb.astype(np.float32)
    overlay = (1.0 - alpha) * orig_f + alpha * heatmap_rgb
    return np.clip(overlay, 0, 255).astype(np.uint8)


def save_gradcam_grid(
    image_paths: List[str],
    labels: List[int],
    model: nn.Module,
    gradcam: "GradCAM",
    class_names: List[str],
    device: torch.device,
    val_transform,
    output_path: str,
    n_per_class: int = 4,
    img_size: int = 512,
    use_clahe: bool = True,
):
    """
    Generate a grid showing Original | Grad-CAM overlay for each class.

    Layout:  rows = classes,  cols = n_per_class × 2 (orig + overlay)
    Border colour: green = correct prediction, red = wrong prediction.

    Args:
        image_paths   : all image file paths (from dataset.image_paths)
        labels        : corresponding ground-truth labels
        model         : loaded, eval-mode nn.Module
        gradcam       : GradCAM instance (hooks already registered)
        class_names   : ordered list of class name strings
        device        : torch.device
        val_transform : torchvision transform pipeline (same as val/test)
        output_path   : where to save the PNG grid
        n_per_class   : number of example images per class
        img_size      : target image size (for CLAHE display resize)
        use_clahe     : whether to apply CLAHE for display image
    """
    from src.dataset import apply_clahe   # local import to avoid circular deps

    # Group image paths by ground-truth class
    by_class: dict = defaultdict(list)
    for path, lbl in zip(image_paths, labels):
        by_class[lbl].append(path)

    n_classes = len(class_names)
    fig, axes = plt.subplots(
        n_classes, n_per_class * 2,
        figsize=(n_per_class * 2 * 3.2, n_classes * 3.5),
    )
    # Ensure axes is always 2-D
    if n_classes == 1:
        axes = axes[np.newaxis, :]

    model.eval()

    for row, cls_idx in enumerate(range(n_classes)):
        paths = by_class[cls_idx][:n_per_class]

        for col, img_path in enumerate(paths):
            # ── Load + CLAHE for display ──
            gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if gray is None:
                continue
            if use_clahe:
                gray = apply_clahe(gray)
            display_rgb = cv2.cvtColor(
                cv2.resize(gray, (img_size, img_size)), cv2.COLOR_GRAY2RGB
            )

            # ── Prepare model input ──
            pil = Image.fromarray(display_rgb)
            inp = val_transform(pil).unsqueeze(0).to(device)

            # ── Grad-CAM ──
            try:
                cam, pred_idx = gradcam.generate(inp, class_idx=cls_idx)
            except RuntimeError as e:
                print(f"Grad-CAM failed for {img_path}: {e}")
                continue

            overlay = overlay_heatmap(display_rgb, cam)
            correct = pred_idx == cls_idx

            # ── Plot: original ──
            ax_o = axes[row, col * 2]
            ax_o.imshow(display_rgb, cmap="gray")
            ax_o.axis("off")
            if col == 0:
                ax_o.set_ylabel(class_names[cls_idx], fontsize=11, fontweight="bold")
            if row == 0:
                ax_o.set_title(f"Original {col + 1}", fontsize=9)

            # ── Plot: overlay ──
            ax_c = axes[row, col * 2 + 1]
            ax_c.imshow(overlay)
            ax_c.axis("off")
            color = "limegreen" if correct else "red"
            for spine in ax_c.spines.values():
                spine.set_visible(True)
                spine.set_edgecolor(color)
                spine.set_linewidth(2.5)
            if row == 0:
                ax_c.set_title(f"Grad-CAM {col + 1}", fontsize=9)
            pred_name = class_names[pred_idx] if pred_idx < len(class_names) else "?"
            status    = "✓" if correct else "✗"
            ax_c.set_xlabel(
                f"{status} Pred: {pred_name}", fontsize=7, color=color, labelpad=2
            )

        # Fill empty columns if fewer images than n_per_class
        for col in range(len(paths), n_per_class):
            axes[row, col * 2].axis("off")
            axes[row, col * 2 + 1].axis("off")

    plt.suptitle(
        "Grad-CAM Visualisations  (green = correct prediction, red = wrong)",
        fontsize=13, fontweight="bold", y=1.01,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Grad-CAM] Grid saved → {output_path}")
