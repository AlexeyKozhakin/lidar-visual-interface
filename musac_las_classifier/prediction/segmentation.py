"""
Unified segmentation inference for both binary and multiclass models.

Processes directories of PNG feature images through a U-Net model
and saves prediction masks.
"""

import logging
import os
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from musac_las_classifier.constants import CLASS_COLORS, DEFAULT_BATCH_SIZE
from musac_las_classifier.prediction.models import load_segmentation_model

logger = logging.getLogger(__name__)


class _PredictionDataset(Dataset):
    """Dataset for loading PNG feature images for inference."""

    def __init__(self, image_dir, transform=None):
        self.image_dir = Path(image_dir)
        self.transform = transform
        self.image_filenames = sorted(self.image_dir.glob("*.png"))

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        image_path = self.image_filenames[idx]
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, image_path.name


def _class_to_rgb(class_mask, class_to_color):
    """Convert a class index mask to an RGB image."""
    h, w = class_mask.shape
    rgb_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for class_idx, color in class_to_color.items():
        rgb_mask[class_mask == class_idx] = color
    return rgb_mask


def predict_tiles(input_directory, output_directory,
                  checkpoint_path, encoder_weights_path,
                  num_classes=None, class_colors=None,
                  batch_size=DEFAULT_BATCH_SIZE, device="auto"):
    """Run segmentation inference on a directory of feature images.

    For multiclass models (num_classes > 2): saves RGB-colored prediction masks.
    For binary models (num_classes == 2): saves grayscale binary masks.

    Args:
        input_directory: Directory with PNG feature images.
        output_directory: Directory to save prediction masks.
        checkpoint_path: Path to model checkpoint.
        encoder_weights_path: Path to encoder weights.
        num_classes: Number of classes (auto-detected from class_colors if None).
        class_colors: Class-to-RGB mapping (defaults to STPLS3D 20-class).
        batch_size: Inference batch size.
        device: 'cuda', 'cpu', or 'auto'.
    """
    if class_colors is None:
        class_colors = CLASS_COLORS
    if num_classes is None:
        num_classes = len(class_colors)

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    os.makedirs(output_directory, exist_ok=True)

    transform = T.Compose([T.ToTensor()])
    dataset = _PredictionDataset(input_directory, transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    model = load_segmentation_model(
        checkpoint_path, encoder_weights_path, num_classes, device
    )

    logger.info(
        "Running inference: %d images, num_classes=%d, device=%s",
        len(dataset), num_classes, device,
    )

    is_binary = num_classes == 2

    with torch.no_grad():
        for images, filenames in dataloader:
            images = images.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()

            for pred, filename in zip(preds, filenames):
                if is_binary:
                    pred_image = (pred * 255).astype(np.uint8)
                    pred_pil = Image.fromarray(pred_image, mode="L")
                else:
                    rgb_mask = _class_to_rgb(pred, class_colors)
                    pred_pil = Image.fromarray(rgb_mask)

                pred_pil.save(os.path.join(output_directory, filename))

    logger.info("Predictions saved to %s", output_directory)
