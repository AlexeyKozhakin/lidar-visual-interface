"""
Data preparation script for training.

Orchestrates: raw LAS files -> tiling -> KNN encoding -> feature images + class images.

Usage:
    python training/prepare_data.py
    python training/prepare_data.py --raw-dir data/stpls3d/raw --output-dir data/stpls3d
"""

import argparse
import logging
import os

from musac_las_classifier.constants import (
    CHANNELS_VISUALIZATION,
    CLASS_COLORS,
    FEATURE_INPUT_TENSOR,
    FEATURE_OUTPUT_TENSOR,
)
from musac_las_classifier.preprocessing.encoding import encode_las_to_tensors
from musac_las_classifier.preprocessing.imaging import tensors_to_images
from musac_las_classifier.preprocessing.tiling import tile_las_files

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def _generate_class_images(tensor_dir, output_dir, class_colors=None):
    """Generate class mask images from tensors (channel index 6 = class).

    This is specific to training data where ground-truth labels are available.
    """
    import numpy as np
    from PIL import Image

    if class_colors is None:
        class_colors = CLASS_COLORS

    os.makedirs(output_dir, exist_ok=True)
    filenames = [f for f in os.listdir(tensor_dir) if f.endswith(".npy")]

    for filename in filenames:
        data = np.load(os.path.join(tensor_dir, filename))
        class_channel = data[:, :, FEATURE_OUTPUT_TENSOR["class"]].astype(int)

        h, w = class_channel.shape
        rgb = np.zeros((h, w, 3), dtype=np.uint8)
        for cls_id, color in class_colors.items():
            rgb[class_channel == cls_id] = color

        name, _ = os.path.splitext(filename)
        Image.fromarray(rgb).save(os.path.join(output_dir, f"{name}.png"))

    logger.info("Generated %d class images in %s", len(filenames), output_dir)


def main():
    parser = argparse.ArgumentParser(description="Prepare training data from raw LAS")
    parser.add_argument(
        "--raw-dir", default="data/stpls3d/raw",
        help="Directory with raw LAS files",
    )
    parser.add_argument(
        "--output-dir", default="data/stpls3d",
        help="Base output directory",
    )
    parser.add_argument("--tile-size", type=int, default=250)
    parser.add_argument("--num-points", type=int, default=30000)
    parser.add_argument("--grid-size", type=int, default=512)
    parser.add_argument("--k-nn", type=int, default=4)
    args = parser.parse_args()

    cut_dir = os.path.join(args.output_dir, "las_cut")
    tensor_dir = os.path.join(args.output_dir, "tensors")
    features_dir = os.path.join(args.output_dir, "img_features")
    class_dir = os.path.join(args.output_dir, "img_class")

    # Step 1: Tile raw LAS files
    logger.info("Step 1/4: Tiling LAS files...")
    tile_las_files(
        args.raw_dir, cut_dir,
        tile_size=args.tile_size,
        train_mode=True,
    )

    # Step 2: Encode to feature tensors
    logger.info("Step 2/4: Encoding to feature tensors...")
    encode_las_to_tensors(
        cut_dir, tensor_dir,
        feature_input_tensor=FEATURE_INPUT_TENSOR,
        feature_output_tensor=FEATURE_OUTPUT_TENSOR,
        num_points_lim=args.num_points,
        M=args.grid_size,
        K=args.k_nn,
    )

    # Step 3: Generate feature images (model input)
    logger.info("Step 3/4: Generating feature images...")
    tensors_to_images(
        tensor_dir, features_dir,
        feature_output_tensor=FEATURE_OUTPUT_TENSOR,
        channels_visualization=CHANNELS_VISUALIZATION,
    )

    # Step 4: Generate class mask images (ground truth)
    logger.info("Step 4/4: Generating class mask images...")
    _generate_class_images(tensor_dir, class_dir)

    logger.info("Data preparation complete!")
    logger.info("  Features: %s", features_dir)
    logger.info("  Masks:    %s", class_dir)
    logger.info("Ready for training: python training/train.py --features-dir %s --masks-dir %s",
                features_dir, class_dir)


if __name__ == "__main__":
    main()
