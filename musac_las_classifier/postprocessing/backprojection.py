"""
2D-to-3D back-projection of segmentation predictions onto LAS point clouds.

Maps pixel-level class predictions from 2D images back to original 3D points
using nearest-neighbor interpolation.
"""

import logging
import os

import laspy
import numpy as np
from PIL import Image
from scipy.interpolate import NearestNDInterpolator

from musac_las_classifier.constants import CLASS_COLORS

logger = logging.getLogger(__name__)


def backproject_mask_to_las(las_file_path, image_file_path, output_las_path,
                            class_colors=None, write_rgb=True):
    """Project 2D class predictions back onto a 3D LAS point cloud.

    Each 3D point is assigned a class label (and optionally RGB color)
    based on the nearest pixel in the prediction mask.

    Args:
        las_file_path: Path to the original LAS file.
        image_file_path: Path to the prediction mask image (RGB or grayscale).
        output_las_path: Path to save the classified LAS file.
        class_colors: Dict mapping class_id -> [R, G, B].
        write_rgb: If True, also writes RGB color to the LAS file.
    """
    if class_colors is None:
        class_colors = CLASS_COLORS

    color_to_class = {tuple(v): k for k, v in class_colors.items()}

    las = laspy.read(las_file_path)
    x, y, z = np.array(las.x), np.array(las.y), np.array(las.z)

    image = np.array(Image.open(image_file_path))
    img_height, img_width = image.shape[:2]

    # Normalize LAS coordinates to [0, 1]
    x_min, y_min = np.min(x), np.min(y)
    x_shifted = x - x_min
    y_shifted = y - y_min
    x_max_shifted = np.max(x_shifted)
    y_max_shifted = np.max(y_shifted)
    if x_max_shifted == 0:
        x_max_shifted = 1.0
    if y_max_shifted == 0:
        y_max_shifted = 1.0
    x_scaled = x_shifted / x_max_shifted
    y_scaled = y_shifted / y_max_shifted

    # Build image grid in [0, 1] coordinates
    xi = np.linspace(0, 1, img_width)
    yi = np.linspace(0, 1, img_height)
    xi, yi = np.meshgrid(xi, yi)
    xi_flat = xi.ravel()
    yi_flat = yi.ravel()

    if image.ndim == 3:
        colors_flat = image.reshape(-1, 3)
    else:
        # Grayscale — expand to 3 channels
        colors_flat = np.stack([image.ravel()] * 3, axis=-1)

    interpolator = NearestNDInterpolator(
        np.column_stack((xi_flat, yi_flat)), colors_flat
    )
    nearest_colors = interpolator(x_scaled, y_scaled)

    # Map colors to class indices
    classifications = np.zeros(len(nearest_colors), dtype=np.uint8)
    rgb_values = np.zeros((len(nearest_colors), 3), dtype=np.uint16)

    for i, color in enumerate(nearest_colors):
        color_tuple = tuple(np.round(color).astype(int))
        classifications[i] = color_to_class.get(color_tuple, 0)
        if write_rgb and color_tuple in color_to_class:
            rgb_values[i] = (np.array(color_tuple) * 256).astype(np.uint16)

    # Create output LAS preserving original data
    new_las = laspy.create(
        point_format=las.point_format,
        file_version=str(las.header.version),
    )
    for dimension in las.point_format.dimension_names:
        if hasattr(las, dimension):
            setattr(new_las, dimension, getattr(las, dimension))

    new_las.classification = classifications

    if write_rgb and "red" in new_las.point_format.dimension_names:
        new_las.red = rgb_values[:, 0]
        new_las.green = rgb_values[:, 1]
        new_las.blue = rgb_values[:, 2]

    os.makedirs(os.path.dirname(output_las_path) or ".", exist_ok=True)
    new_las.write(output_las_path)
    logger.info("Classified LAS saved: %s", output_las_path)
