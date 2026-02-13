"""
Building polygon extraction from binary segmentation masks.

Detects contours in binary prediction masks and exports them as
ESRI Shapefiles for GIS integration.
"""

import logging
import os

import cv2
import numpy as np
import shapefile
from PIL import Image

from musac_las_classifier.constants import DEFAULT_CONTOUR_THICKNESS, DEFAULT_MIN_POLYGON_AREA

logger = logging.getLogger(__name__)


def extract_polygons(input_path, output_image_path, output_shp_path,
                     min_area=DEFAULT_MIN_POLYGON_AREA,
                     contour_thickness=DEFAULT_CONTOUR_THICKNESS):
    """Extract building polygons from a binary mask image.

    Args:
        input_path: Path to binary prediction mask (grayscale PNG).
        output_image_path: Path to save contour visualization image.
        output_shp_path: Path to save Shapefile (without extension).
        min_area: Minimum contour area in pixels to keep.
        contour_thickness: Line thickness for contour visualization.
    """
    image = Image.open(input_path).convert("L")
    image_np = np.array(image)

    _, binary = cv2.threshold(image_np, 127, 255, cv2.THRESH_BINARY_INV)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) >= min_area]

    logger.info(
        "Found %d contours (%d after filtering)",
        len(contours), len(filtered_contours),
    )

    # Visualization image
    h, w = binary.shape
    output_image = np.ones((h, w, 3), dtype=np.uint8) * 255
    cv2.drawContours(output_image, filtered_contours, -1, (150, 150, 150), cv2.FILLED)
    cv2.drawContours(output_image, filtered_contours, -1, (255, 0, 0), contour_thickness)
    Image.fromarray(output_image).save(output_image_path)

    # Shapefile export
    shp_writer = shapefile.Writer(output_shp_path, shapeType=shapefile.POLYGON)
    shp_writer.field("ID", "N")

    for i, contour in enumerate(filtered_contours):
        points = contour.squeeze()
        if len(points.shape) == 1:
            continue
        points_list = points.tolist()
        # Flip Y axis for geographic coordinate convention
        flipped_points = [(x, h - y) for (x, y) in points_list]
        if flipped_points[0] != flipped_points[-1]:
            flipped_points.append(flipped_points[0])
        shp_writer.poly([flipped_points])
        shp_writer.record(i)

    shp_writer.close()
    logger.info("Shapefile saved: %s.shp", output_shp_path)


def extract_polygons_from_directory(input_dir, output_image_dir, output_shp_dir,
                                    min_area=DEFAULT_MIN_POLYGON_AREA,
                                    contour_thickness=DEFAULT_CONTOUR_THICKNESS):
    """Extract polygons from all mask images in a directory.

    Args:
        input_dir: Directory with binary mask images.
        output_image_dir: Directory to save contour visualization images.
        output_shp_dir: Directory to save Shapefiles.
        min_area: Minimum contour area.
        contour_thickness: Contour line thickness.
    """
    os.makedirs(output_image_dir, exist_ok=True)
    os.makedirs(output_shp_dir, exist_ok=True)

    for filename in os.listdir(input_dir):
        if filename.lower().endswith((".png", ".jpg", ".jpeg")):
            input_path = os.path.join(input_dir, filename)
            name_wo_ext = os.path.splitext(filename)[0]
            output_image_path = os.path.join(output_image_dir, f"{name_wo_ext}.png")
            output_shp_path = os.path.join(output_shp_dir, name_wo_ext)

            logger.info("Processing: %s", filename)
            extract_polygons(
                input_path, output_image_path, output_shp_path,
                min_area=min_area, contour_thickness=contour_thickness,
            )
