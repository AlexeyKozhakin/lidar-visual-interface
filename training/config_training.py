"""
Training configuration.

All hyperparameters and paths for model training.
"""

import dataclasses
import json
import os
from dataclasses import dataclass


@dataclass
class TrainingConfig:
    """Configuration for model training."""

    # Data paths
    raw_las_dir: str = "data/stpls3d/raw"
    cut_las_dir: str = "data/stpls3d/las_cut"
    tensor_dir: str = "data/stpls3d/tensors"
    img_features_dir: str = "data/stpls3d/img_features"
    img_class_dir: str = "data/stpls3d/img_class"

    # Training hyperparameters
    num_epochs: int = 100
    batch_size: int = 8
    learning_rate: float = 1e-3
    train_ratio: float = 0.8

    # Model
    encoder_name: str = "resnet34"
    num_classes: int = 20
    in_channels: int = 3

    # Hardware
    device: str = "auto"

    # Reproducibility
    seed: int = 42

    # Output
    checkpoint_dir: str = "checkpoints"
    metrics_csv: str = "checkpoints/metrics_log.csv"
    epoch_summary_csv: str = "checkpoints/epoch_summary.csv"
    best_model_path: str = "checkpoints/best_model.pth"
    config_json_path: str = "checkpoints/training_config.json"
    confusion_matrix_dir: str = "checkpoints/confusion_matrices"
    plots_dir: str = "checkpoints/plots"

    def resolve_device(self) -> str:
        if self.device == "auto":
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        return self.device

    def to_dict(self) -> dict:
        """Serialize config to a JSON-compatible dictionary."""
        return dataclasses.asdict(self)

    def save_json(self, path: str = None):
        """Save full config as JSON for reproducibility."""
        path = path or self.config_json_path
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
