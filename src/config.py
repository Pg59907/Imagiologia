from dataclasses import dataclass, field, asdict
from typing import Optional, List


@dataclass
class Config:
    # ── Data ──────────────────────────────────────────────────
    data_dir: str = "./dataset"
    img_size: int = 512
    num_classes: int = 4
    class_names: List[str] = field(
        default_factory=lambda: ["Biliary_Leaks", "Lithiasis", "Normal", "Stricture"]
    )

    # ── CLAHE ─────────────────────────────────────────────────
    use_clahe: bool = True
    clahe_clip_limit: float = 2.0
    clahe_tile_size: int = 8

    # ── Model ─────────────────────────────────────────────────
    model_name: str = "tf_efficientnetv2_s"   # timm model string
    pretrained: bool = True
    drop_rate: float = 0.3

    # ── Training ──────────────────────────────────────────────
    epochs: int = 60
    batch_size: int = 16
    lr: float = 1e-4
    weight_decay: float = 1e-4
    early_stopping_patience: int = 10
    num_workers: int = 4
    seed: int = 42

    # ── Loss ──────────────────────────────────────────────────
    use_class_weights: bool = True
    focal_gamma: float = 2.0

    # ── Output ────────────────────────────────────────────────
    output_dir: str = "./outputs"

    # ── WandB (optional) ──────────────────────────────────────
    use_wandb: bool = False
    wandb_project: str = "ercp-classification"
    run_name: Optional[str] = None

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict):
        # drop unknown keys for forward-compatibility
        valid = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in d.items() if k in valid})
