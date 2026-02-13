"""
Model loading utilities.

Handles loading U-Net + ResNet34 models for both binary and multiclass segmentation.
"""

import logging

import segmentation_models_pytorch as smp
import torch

logger = logging.getLogger(__name__)


def load_segmentation_model(checkpoint_path, encoder_weights_path,
                            num_classes, device="cpu"):
    """Load a U-Net segmentation model from checkpoint.

    Args:
        checkpoint_path: Path to the trained model checkpoint (.pth).
        encoder_weights_path: Path to ResNet34 encoder weights (.pth).
        num_classes: Number of output classes (2 for binary, 20 for multiclass).
        device: Torch device string ('cpu' or 'cuda').

    Returns:
        Loaded model in eval mode on the specified device.
    """
    logger.info("Loading model: %s (num_classes=%d)", checkpoint_path, num_classes)

    encoder_state_dict = torch.load(
        encoder_weights_path, map_location=device, weights_only=False
    )

    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=3,
        classes=num_classes,
    )

    model.encoder.load_state_dict(encoder_state_dict)
    model.load_state_dict(
        torch.load(checkpoint_path, map_location=device, weights_only=False)
    )

    model.to(device)
    model.eval()

    logger.info("Model loaded on %s", device)
    return model
