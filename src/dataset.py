"""
Dataset module for ERCP image classification.

Reads PNG images from the fixed train/val/test folder hierarchy,
applies CLAHE contrast enhancement (X-ray specific), converts to
3-channel RGB for pretrained models, and applies augmentation.
"""

import os
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

# ImageNet statistics – used because backbones are pretrained on ImageNet
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


# ──────────────────────────────────────────────
# CLAHE helper
# ──────────────────────────────────────────────

def apply_clahe(gray: np.ndarray, clip_limit: float = 2.0, tile_size: int = 8) -> np.ndarray:
    """
    Apply Contrast Limited Adaptive Histogram Equalization to a grayscale image.
    Greatly improves local contrast in fluoroscopic X-rays.

    Args:
        gray        : uint8 grayscale array (H, W)
        clip_limit  : threshold for contrast limiting (higher → more enhancement)
        tile_size   : size of the grid for local histogram computation

    Returns:
        uint8 grayscale array after CLAHE
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
    return clahe.apply(gray)


# ──────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────

class ERCPDataset(Dataset):
    """
    Folder-based dataset for ERCP classification.

    Expected structure:
        <data_dir>/<phase>/<ClassName>/<image>.png
    """

    def __init__(self, data_dir: str, phase: str, cfg):
        """
        Args:
            data_dir : root directory containing train/val/test sub-folders
            phase    : one of 'train', 'val', 'test'
            cfg      : Config object
        """
        self.phase      = phase
        self.use_clahe  = cfg.use_clahe
        self.clahe_clip = cfg.clahe_clip_limit
        self.clahe_tile = cfg.clahe_tile_size

        root = Path(data_dir) / phase
        if not root.exists():
            raise FileNotFoundError(f"Dataset phase folder not found: {root}")

        self.class_names: List[str] = sorted(
            [d.name for d in root.iterdir() if d.is_dir()]
        )
        self.class_to_idx = {c: i for i, c in enumerate(self.class_names)}

        # Collect all image paths and labels
        valid_exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
        self.image_paths: List[str] = []
        self.labels: List[int] = []

        for cls in self.class_names:
            for p in sorted((root / cls).iterdir()):
                if p.suffix.lower() in valid_exts:
                    self.image_paths.append(str(p))
                    self.labels.append(self.class_to_idx[cls])

        # ── Augmentation transforms ──
        if phase == "train":
            self.transform = T.Compose([
                T.Resize((cfg.img_size, cfg.img_size)),
                T.RandomHorizontalFlip(p=0.5),
                T.RandomVerticalFlip(p=0.2),
                T.RandomRotation(degrees=15),
                T.ColorJitter(brightness=0.15, contrast=0.15),
                T.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.9, 1.1)),
                T.ToTensor(),
                T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ])
        else:
            self.transform = T.Compose([
                T.Resize((cfg.img_size, cfg.img_size)),
                T.ToTensor(),
                T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ])

    # ── Class weighting ──────────────────────────────────────

    def get_class_weights(self) -> torch.Tensor:
        """
        Compute inverse-frequency class weights for the loss function.
        Weight_c = N / (C * N_c)  where N=total samples, C=num classes, N_c=samples in class c.
        """
        counts = np.bincount(self.labels, minlength=len(self.class_names)).astype(float)
        counts = np.maximum(counts, 1.0)          # avoid division by zero
        weights = len(self.labels) / (len(self.class_names) * counts)
        return torch.FloatTensor(weights)

    def class_distribution(self) -> dict:
        counts = np.bincount(self.labels, minlength=len(self.class_names))
        return {c: int(counts[i]) for i, c in enumerate(self.class_names)}

    # ── Dataset protocol ─────────────────────────────────────

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path = self.image_paths[idx]
        label    = self.labels[idx]

        # Read as grayscale (fluoroscopic images are grayscale)
        gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if gray is None:
            raise IOError(f"Cannot read image: {img_path}")

        # Apply CLAHE
        if self.use_clahe:
            gray = apply_clahe(gray, self.clahe_clip, self.clahe_tile)

        # Convert grayscale → 3-channel RGB (required by pretrained models)
        rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        img = Image.fromarray(rgb)

        return self.transform(img), label
