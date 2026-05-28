"""
Weighted Focal Loss for imbalanced multi-class classification.

Focal Loss (Lin et al., 2017) down-weights well-classified examples,
forcing the model to focus on hard/misclassified cases.
Per-class weights further compensate for dataset imbalance.

Combined:  FL(p_t) = -α_c · (1 − p_t)^γ · log(p_t)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class WeightedFocalLoss(nn.Module):
    """
    Focal Loss with optional per-class weighting.

    Args:
        gamma  : focusing parameter (0 = standard CE, 2 = recommended default)
        weight : 1-D FloatTensor of per-class weights (computed from class frequencies)
    """

    def __init__(self, gamma: float = 2.0, weight: torch.Tensor = None):
        super().__init__()
        self.gamma = gamma
        # Register as buffer so it moves to the right device automatically
        if weight is not None:
            self.register_buffer("weight", weight)
        else:
            self.weight = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits  : (B, C) raw model output (before softmax)
            targets : (B,)   ground-truth class indices

        Returns:
            Scalar loss value
        """
        # Standard cross-entropy with class weights, no reduction yet
        ce_loss = F.cross_entropy(logits, targets, weight=self.weight, reduction="none")

        # p_t = probability assigned to the correct class
        pt = torch.exp(-ce_loss)

        # Focal weight
        focal_loss = (1.0 - pt) ** self.gamma * ce_loss

        return focal_loss.mean()
