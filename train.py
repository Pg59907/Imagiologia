"""
train.py – Main training entry point for ERCP image classification.

Usage examples:
    # Default  (EfficientNetV2-S, CLAHE enabled, class weights)
    python train.py --data_dir ./dataset

    # Compare with ConvNeXt
    python train.py --data_dir ./dataset --model_name convnext_tiny --run_name convnext_tiny

    # Ablation: no CLAHE
    python train.py --data_dir ./dataset --no_clahe --run_name efficientnetv2_no_clahe

    # Enable WandB experiment tracking
    python train.py --data_dir ./dataset --use_wandb --run_name run_01
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.config import Config
from src.dataset import ERCPDataset
from src.evaluate import (
    compute_all_metrics,
    plot_confusion_matrix,
    plot_roc_curves,
    plot_training_curves,
    print_metrics,
)
from src.losses import WeightedFocalLoss
from src.model import create_model
from src.trainer import train, validate


# ──────────────────────────────────────────────
# Reproducibility
# ──────────────────────────────────────────────

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="ERCP Classification Training")

    # Data
    p.add_argument("--data_dir",   type=str, default="./dataset",
                   help="Root folder with train/val/test sub-folders")
    p.add_argument("--img_size",   type=int, default=512,
                   help="Resize images to (img_size, img_size)")

    # Model
    p.add_argument("--model_name", type=str, default="tf_efficientnetv2_s",
                   help="timm model name  (tf_efficientnetv2_s | convnext_tiny | resnet50 | …)")
    p.add_argument("--drop_rate",  type=float, default=0.3,
                   help="Dropout rate before the classifier head")

    # Training
    p.add_argument("--epochs",      type=int,   default=60)
    p.add_argument("--batch_size",  type=int,   default=16)
    p.add_argument("--lr",          type=float, default=1e-4)
    p.add_argument("--weight_decay",type=float, default=1e-4)
    p.add_argument("--patience",    type=int,   default=10,
                   help="Early stopping patience (epochs without val F1 improvement)")
    p.add_argument("--num_workers", type=int,   default=4,
                   help="DataLoader workers (use 0 on Windows)")
    p.add_argument("--seed",        type=int,   default=42)

    # Preprocessing
    p.add_argument("--no_clahe",         action="store_true",
                   help="Disable CLAHE contrast enhancement")
    p.add_argument("--clahe_clip",       type=float, default=2.0)
    p.add_argument("--clahe_tile",       type=int,   default=8)

    # Loss
    p.add_argument("--no_class_weights", action="store_true",
                   help="Disable inverse-frequency class weighting in the loss")
    p.add_argument("--focal_gamma",      type=float, default=2.0,
                   help="Focal Loss gamma  (0 = plain cross-entropy)")

    # Output / tracking
    p.add_argument("--output_dir", type=str, default="./outputs")
    p.add_argument("--use_wandb",  action="store_true")
    p.add_argument("--run_name",   type=str, default=None)

    return p.parse_args()


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main():
    args = parse_args()

    cfg = Config(
        data_dir               = args.data_dir,
        img_size               = args.img_size,
        model_name             = args.model_name,
        drop_rate              = args.drop_rate,
        epochs                 = args.epochs,
        batch_size             = args.batch_size,
        lr                     = args.lr,
        weight_decay           = args.weight_decay,
        early_stopping_patience= args.patience,
        num_workers            = args.num_workers,
        seed                   = args.seed,
        use_clahe              = not args.no_clahe,
        clahe_clip_limit       = args.clahe_clip,
        clahe_tile_size        = args.clahe_tile,
        use_class_weights      = not args.no_class_weights,
        focal_gamma            = args.focal_gamma,
        output_dir             = args.output_dir,
        use_wandb              = args.use_wandb,
        run_name               = args.run_name or args.model_name,
    )

    set_seed(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'═'*55}")
    print(f"  ERCP Classification — {cfg.model_name}")
    print(f"{'═'*55}")
    print(f"  Device       : {device}")
    print(f"  CLAHE        : {cfg.use_clahe}")
    print(f"  Class weights: {cfg.use_class_weights}")
    print(f"  Focal γ      : {cfg.focal_gamma}")
    print(f"  Image size   : {cfg.img_size}×{cfg.img_size}")
    print(f"  Batch size   : {cfg.batch_size}")
    print(f"  LR           : {cfg.lr}")
    print(f"{'═'*55}\n")

    # ── Datasets ─────────────────────────────────────────────
    train_ds = ERCPDataset(cfg.data_dir, "train", cfg)
    val_ds   = ERCPDataset(cfg.data_dir, "val",   cfg)
    test_ds  = ERCPDataset(cfg.data_dir, "test",  cfg)

    print("  Class distribution:")
    for phase, ds in [("train", train_ds), ("val", val_ds), ("test", test_ds)]:
        dist = ds.class_distribution()
        total = sum(dist.values())
        parts = "  ".join(f"{c}: {n}" for c, n in dist.items())
        print(f"    {phase:>5} ({total:4d})  {parts}")
    print()

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=cfg.num_workers, pin_memory=device.type == "cuda")
    val_loader   = DataLoader(val_ds,   batch_size=cfg.batch_size, shuffle=False,
                              num_workers=cfg.num_workers, pin_memory=device.type == "cuda")
    test_loader  = DataLoader(test_ds,  batch_size=cfg.batch_size, shuffle=False,
                              num_workers=cfg.num_workers, pin_memory=device.type == "cuda")

    # ── Model ────────────────────────────────────────────────
    model = create_model(cfg.model_name, cfg.num_classes, cfg.pretrained, cfg.drop_rate)
    model = model.to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable parameters: {n_params:,}\n")

    # ── Loss ─────────────────────────────────────────────────
    class_weights = train_ds.get_class_weights().to(device) if cfg.use_class_weights else None
    if class_weights is not None:
        print("  Class weights:", {c: f"{w:.3f}" for c, w in zip(train_ds.class_names, class_weights.cpu().tolist())})
    criterion = WeightedFocalLoss(gamma=cfg.focal_gamma, weight=class_weights)

    # ── Optimiser + Scheduler ────────────────────────────────
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs, eta_min=1e-6)

    # ── Train ────────────────────────────────────────────────
    history, best_model_path = train(
        model, train_loader, val_loader, criterion, optimizer, scheduler, cfg, device
    )

    # ── Plot training curves ─────────────────────────────────
    out = Path(cfg.output_dir)
    plot_training_curves(history, str(out / "training_curves.png"))

    # ── Load best checkpoint for test evaluation ─────────────
    print(f"\n  Loading best checkpoint: {best_model_path}")
    ckpt = torch.load(best_model_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])

    # ── Test evaluation ──────────────────────────────────────
    _, _, test_labels, test_preds, test_probs = validate(model, test_loader, criterion, device)

    metrics = compute_all_metrics(test_labels, test_preds, test_probs, train_ds.class_names)
    print_metrics(metrics)

    plot_confusion_matrix(test_labels, test_preds, train_ds.class_names,
                          str(out / "confusion_matrix_test.png"))
    plot_roc_curves(test_labels, test_probs, train_ds.class_names,
                    str(out / "roc_curves_test.png"))

    # ── Save metrics JSON ────────────────────────────────────
    metrics_json = {
        k: (v if not isinstance(v, dict) else {kk: (float(vv) if vv is not None else None)
                                                for kk, vv in v.items()})
        for k, v in metrics.items()
        if k != "report"
    }
    metrics_json["accuracy"] = float(metrics_json["accuracy"])
    metrics_json["f1_macro"] = float(metrics_json["f1_macro"])
    if metrics_json["auc_macro"] is not None:
        metrics_json["auc_macro"] = float(metrics_json["auc_macro"])
    metrics_json["model"]   = cfg.model_name
    metrics_json["use_clahe"] = cfg.use_clahe

    metrics_path = out / "test_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics_json, f, indent=2)
    print(f"\n  Metrics JSON saved → {metrics_path}")
    print(f"  All outputs in    → {out.resolve()}")


if __name__ == "__main__":
    main()
