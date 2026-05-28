"""
Training loop with:
  - Per-epoch train / validation
  - Early stopping (monitored on val F1-macro)
  - Best-model checkpoint saving
  - Optional WandB logging
"""

import time
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
from sklearn.metrics import f1_score


# ──────────────────────────────────────────────
# Early stopping
# ──────────────────────────────────────────────

class EarlyStopping:
    """Stop training when val metric stops improving."""

    def __init__(self, patience: int = 10, min_delta: float = 1e-4):
        self.patience  = patience
        self.min_delta = min_delta
        self.counter   = 0
        self.best      = None

    def step(self, score: float) -> bool:
        """Call once per epoch.  Returns True when training should stop."""
        if self.best is None or score > self.best + self.min_delta:
            self.best   = score
            self.counter = 0
            return False
        self.counter += 1
        return self.counter >= self.patience


# ──────────────────────────────────────────────
# Single epoch helpers
# ──────────────────────────────────────────────

def _train_epoch(model, loader, criterion, optimizer, device) -> Tuple[float, float]:
    model.train()
    total_loss = 0.0
    all_preds, all_labels = [], []

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()
        logits = model(imgs)
        loss   = criterion(logits, labels)
        loss.backward()
        # Gradient clipping prevents occasional exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * len(labels)
        all_preds.extend(logits.argmax(1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    f1       = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return avg_loss, f1


@torch.no_grad()
def _val_epoch(model, loader, criterion, device) -> Tuple[float, float, np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    total_loss = 0.0
    all_preds, all_labels, all_probs = [], [], []

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        logits = model(imgs)
        loss   = criterion(logits, labels)

        total_loss += loss.item() * len(labels)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        all_preds.extend(logits.argmax(1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs)

    avg_loss = total_loss / len(loader.dataset)
    labels_arr = np.array(all_labels)
    preds_arr  = np.array(all_preds)
    probs_arr  = np.array(all_probs)
    f1 = f1_score(labels_arr, preds_arr, average="macro", zero_division=0)
    return avg_loss, f1, labels_arr, preds_arr, probs_arr


# ── Public aliases used in train.py and evaluate scripts ──
def validate(model, loader, criterion, device):
    return _val_epoch(model, loader, criterion, device)


# ──────────────────────────────────────────────
# Full training loop
# ──────────────────────────────────────────────

def train(model, train_loader, val_loader, criterion, optimizer, scheduler, cfg, device) -> Tuple[Dict, str]:
    """
    Train the model and return (history dict, path to best checkpoint).

    Args:
        model        : nn.Module
        train_loader : DataLoader for training split
        val_loader   : DataLoader for validation split
        criterion    : loss function
        optimizer    : torch optimizer
        scheduler    : learning-rate scheduler
        cfg          : Config dataclass
        device       : torch.device

    Returns:
        history          : dict with lists of train/val loss and F1 per epoch
        best_model_path  : path (str) of the saved best checkpoint
    """
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    best_model_path = str(output_dir / "best_model.pth")

    stopper     = EarlyStopping(patience=cfg.early_stopping_patience)
    best_val_f1 = -1.0
    history     = {"train_loss": [], "val_loss": [], "train_f1": [], "val_f1": []}

    # ── Optional WandB ──────────────────────────────────────
    wandb_run = None
    if cfg.use_wandb:
        try:
            import wandb
            wandb_run = wandb.init(
                project=cfg.wandb_project,
                name=cfg.run_name or cfg.model_name,
                config=cfg.to_dict(),
            )
        except ImportError:
            print("[WandB] Not installed – skipping experiment tracking.")
        except Exception as e:
            print(f"[WandB] Init failed: {e}")

    print(f"\n{'─'*65}")
    print(f"  Training  │  Model: {cfg.model_name}  │  Device: {device}")
    print(f"{'─'*65}")
    print(f"  {'Epoch':>6}  │  {'Train Loss':>10}  │  {'Val Loss':>8}  │  "
          f"{'Train F1':>8}  │  {'Val F1':>7}  │  {'Time':>5}")
    print(f"{'─'*65}")

    for epoch in range(1, cfg.epochs + 1):
        t0 = time.time()

        train_loss, train_f1 = _train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_f1, _, _, _ = _val_epoch(model, val_loader, criterion, device)

        scheduler.step()
        elapsed = time.time() - t0

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_f1"].append(train_f1)
        history["val_f1"].append(val_f1)

        marker = " ★" if val_f1 > best_val_f1 else ""
        print(f"  {epoch:>6}  │  {train_loss:>10.4f}  │  {val_loss:>8.4f}  │  "
              f"{train_f1:>8.4f}  │  {val_f1:>7.4f}  │  {elapsed:>4.1f}s{marker}")

        if wandb_run:
            wandb_run.log({
                "epoch": epoch,
                "train/loss": train_loss,
                "val/loss":   val_loss,
                "train/f1_macro": train_f1,
                "val/f1_macro":   val_f1,
                "lr": scheduler.get_last_lr()[0],
            })

        # ── Save best checkpoint ─────────────────────────────
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save({
                "epoch":             epoch,
                "model_state_dict":  model.state_dict(),
                "val_f1":            val_f1,
                "cfg":               cfg.to_dict(),   # save as dict for forward-compatibility
            }, best_model_path)

        if stopper.step(val_f1):
            print(f"\n  Early stopping triggered at epoch {epoch} "
                  f"(no improvement for {cfg.early_stopping_patience} epochs).")
            break

    print(f"{'─'*65}")
    print(f"  Best val F1-macro : {best_val_f1:.4f}  (baseline: 0.738)")
    print(f"  Checkpoint saved  : {best_model_path}")

    if wandb_run:
        wandb_run.finish()

    return history, best_model_path
