"""
Convert feature tensors to PNG images for model input.

Selects specified channels from (M, M, 7) tensors, normalizes to [0, 255],
and saves as 3-channel PNG images.
"""

import logging
import os
from multiprocessing import Pool

import numpy as np
from PIL import Image

from musac_las_classifier.constants import CHANNELS_VISUALIZATION, FEATURE_OUTPUT_TENSOR

logger = logging.getLogger(__name__)


def tensor_to_image(input_dir, filename, feature_output_tensor, channels_visualization,
                    output_dir):
    """Convert a single .npy tensor to a 3-channel PNG image.

    Args:
        input_dir: Directory containing .npy files.
        filename: Name of the .npy file.
        feature_output_tensor: Output channel name-to-index mapping.
        channels_visualization: Which channels to visualize (name -> RGB index).
        output_dir: Directory to save PNG images.
    """
    file_path = os.path.join(input_dir, filename)
    data = np.load(file_path)

    os.makedirs(output_dir, exist_ok=True)

    image_data = np.zeros((data.shape[0], data.shape[1], 3), dtype=np.uint8)

    for channel_name, rgb_index in channels_visualization.items():
        if channel_name not in feature_output_tensor:
            logger.warning(
                "Channel '%s' not found in feature_output_tensor, skipping",
                channel_name,
            )
            continue

        channel_data = data[:, :, feature_output_tensor[channel_name]]
        max_val = np.max(channel_data)
        if max_val > 0:
            channel_normalized = (channel_data / max_val * 255).astype(np.uint8)
        else:
            channel_normalized = np.zeros_like(channel_data, dtype=np.uint8)

        channel_normalized = np.clip(channel_normalized, 0, 255)
        image_data[:, :, rgb_index] = channel_normalized

    name, _ = os.path.splitext(filename)
    output_path = os.path.join(output_dir, f"{name}.png")
    Image.fromarray(image_data).save(output_path)


def tensors_to_images(input_dir, output_dir,
                      feature_output_tensor=None, channels_visualization=None,
                      parallel=False):
    """Convert all .npy tensors in a directory to PNG images.

    Args:
        input_dir: Directory with .npy tensor files.
        output_dir: Directory to save PNG images.
        feature_output_tensor: Output channel mapping (defaults to standard).
        channels_visualization: Channels to visualize (defaults to z_mean, n_z, n_r).
        parallel: Use multiprocessing if True.
    """
    if feature_output_tensor is None:
        feature_output_tensor = FEATURE_OUTPUT_TENSOR
    if channels_visualization is None:
        channels_visualization = CHANNELS_VISUALIZATION

    os.makedirs(output_dir, exist_ok=True)
    filenames = [f for f in os.listdir(input_dir) if f.endswith(".npy")]

    if not filenames:
        logger.warning("No .npy files found in %s", input_dir)
        return

    logger.info("Converting %d tensors to images", len(filenames))

    if parallel and len(filenames) > 1:
        num_processes = min(os.cpu_count() or 1, len(filenames))
        with Pool(processes=num_processes) as pool:
            pool.starmap(
                tensor_to_image,
                [
                    (input_dir, f, feature_output_tensor, channels_visualization, output_dir)
                    for f in filenames
                ],
            )
    else:
        for filename in filenames:
            tensor_to_image(
                input_dir, filename, feature_output_tensor, channels_visualization, output_dir
            )
