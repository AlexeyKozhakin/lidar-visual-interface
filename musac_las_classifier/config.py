"""
Unified pipeline configuration.

Single PipelineConfig dataclass replaces all scattered config modules.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from musac_las_classifier.constants import (
    CHANNELS_VISUALIZATION,
    CLASS_COLORS,
    CLASS_COLORS_BINARY,
    DEFAULT_BATCH_SIZE,
    DEFAULT_CONTOUR_THICKNESS,
    DEFAULT_K_NN,
    DEFAULT_M_TENSOR_SIZE,
    DEFAULT_MIN_POLYGON_AREA,
    DEFAULT_NUM_POINTS_LIM,
    DEFAULT_TILE_SIZE,
    FEATURE_INPUT_TENSOR,
    FEATURE_OUTPUT_TENSOR,
    NUM_CLASSES_BINARY,
    NUM_CLASSES_MULTICLASS,
)

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """Configuration for the LAS classification pipeline.

    Attributes:
        tile_size: Tile size in meters for spatial partitioning.
        k_nn: Number of nearest neighbors for KNN encoding.
        m_tensor_size: Grid resolution (M x M) for 2D tensor.
        num_points_lim: Target number of points per tile (resampled).
        batch_size: Batch size for model inference.
        device: Torch device ('cuda', 'cpu', or 'auto' for auto-detect).
        checkpoint_path: Path to model checkpoint (.pth).
        encoder_weights_path: Path to ResNet34 encoder weights (.pth).
        num_classes: Number of output classes (2 for binary, 20 for multiclass).
        class_colors: Mapping from class index to RGB color.
        feature_input_tensor: Input channel mapping for raw LAS data.
        feature_output_tensor: Output channel mapping for computed features.
        channels_visualization: Which feature channels to use as model input.
        min_polygon_area: Minimum contour area for polygon extraction.
        contour_thickness: Line thickness for contour visualization.
        train_mode: If True, uses training-mode tile naming conventions.
    """

    # Preprocessing
    tile_size: int = DEFAULT_TILE_SIZE
    k_nn: int = DEFAULT_K_NN
    m_tensor_size: int = DEFAULT_M_TENSOR_SIZE
    num_points_lim: int = DEFAULT_NUM_POINTS_LIM

    # Model
    batch_size: int = DEFAULT_BATCH_SIZE
    device: str = "auto"
    checkpoint_path: Optional[str] = None
    encoder_weights_path: Optional[str] = None
    num_classes: int = NUM_CLASSES_MULTICLASS

    # Class definitions
    class_colors: Dict[int, List[int]] = field(default_factory=lambda: dict(CLASS_COLORS))

    # Feature encoding
    feature_input_tensor: Dict[str, int] = field(
        default_factory=lambda: dict(FEATURE_INPUT_TENSOR)
    )
    feature_output_tensor: Dict[str, int] = field(
        default_factory=lambda: dict(FEATURE_OUTPUT_TENSOR)
    )
    channels_visualization: Dict[str, int] = field(
        default_factory=lambda: dict(CHANNELS_VISUALIZATION)
    )

    # Postprocessing
    min_polygon_area: int = DEFAULT_MIN_POLYGON_AREA
    contour_thickness: int = DEFAULT_CONTOUR_THICKNESS

    # Mode
    train_mode: bool = False

    def resolve_device(self) -> str:
        """Resolve 'auto' device to actual torch device string."""
        if self.device == "auto":
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        return self.device

    def validate(self) -> None:
        """Validate configuration, raise ValueError on problems."""
        if self.checkpoint_path is not None:
            p = Path(self.checkpoint_path)
            if not p.exists():
                raise FileNotFoundError(f"Model checkpoint not found: {p}")

        if self.encoder_weights_path is not None:
            p = Path(self.encoder_weights_path)
            if not p.exists():
                raise FileNotFoundError(f"Encoder weights not found: {p}")

        if self.num_classes < 2:
            raise ValueError(f"num_classes must be >= 2, got {self.num_classes}")

        if self.m_tensor_size < 1:
            raise ValueError(f"m_tensor_size must be >= 1, got {self.m_tensor_size}")

    @classmethod
    def for_binary(cls, checkpoint_path: str, encoder_weights_path: str, **kwargs):
        """Create config for binary building segmentation."""
        return cls(
            checkpoint_path=checkpoint_path,
            encoder_weights_path=encoder_weights_path,
            num_classes=NUM_CLASSES_BINARY,
            class_colors=dict(CLASS_COLORS_BINARY),
            **kwargs,
        )

    @classmethod
    def for_multiclass(cls, checkpoint_path: str, encoder_weights_path: str, **kwargs):
        """Create config for 20-class multiclass segmentation."""
        return cls(
            checkpoint_path=checkpoint_path,
            encoder_weights_path=encoder_weights_path,
            num_classes=NUM_CLASSES_MULTICLASS,
            class_colors=dict(CLASS_COLORS),
            **kwargs,
        )
