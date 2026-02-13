"""
Tile stitching for predicted image tiles.

Reassembles individual 512x512 prediction tiles into full scene mosaics
based on coordinate information encoded in filenames.
"""

import logging
import os

from PIL import Image
from tqdm import tqdm

logger = logging.getLogger(__name__)


def _parse_coordinates(filename):
    """Extract (y, x) tile coordinates from filename like '453000_3974000.png'."""
    base = os.path.basename(filename)
    name, _ = os.path.splitext(base)
    parts = name.split("_")
    y, x = map(int, parts[-2:])
    return y, x


def stitch_tiles_by_basename(input_folder, output_folder):
    """Stitch tiles grouped by basename into full scene images.

    Files are grouped by the prefix before the first underscore.
    Each group is stitched into a single joined image.

    Args:
        input_folder: Directory containing tile PNG files.
        output_folder: Directory to save joined images.
    """
    os.makedirs(output_folder, exist_ok=True)

    files = [f for f in os.listdir(input_folder) if f.endswith(".png")]
    if not files:
        logger.warning("No PNG files found in %s", input_folder)
        return

    # Group by basename
    basename_groups = {}
    for file in files:
        basename = file.split("_")[0]
        basename_groups.setdefault(basename, []).append(file)

    logger.info("Found %d basename groups", len(basename_groups))

    for basename, file_list in basename_groups.items():
        logger.info("Stitching %s: %d tiles", basename, len(file_list))

        coordinates = {}
        for file in file_list:
            try:
                x, y = _parse_coordinates(file)
                coordinates[(x, y)] = file
            except (ValueError, IndexError) as e:
                logger.warning("Could not parse coordinates from %s: %s", file, e)

        if not coordinates:
            continue

        x_coords = sorted({x for x, y in coordinates.keys()})
        y_coords = sorted({y for x, y in coordinates.keys()})

        step_x = x_coords[1] - x_coords[0] if len(x_coords) > 1 else 0
        step_y = y_coords[1] - y_coords[0] if len(y_coords) > 1 else 0

        min_x, max_x = min(x_coords), max(x_coords)
        min_y, max_y = min(y_coords), max(y_coords)

        num_x = ((max_x - min_x) // step_x) + 1 if step_x != 0 else 1
        num_y = ((max_y - min_y) // step_y) + 1 if step_y != 0 else 1

        first_image = Image.open(
            os.path.join(input_folder, coordinates[(min_x, min_y)])
        )
        tile_width, tile_height = first_image.size

        final_image = Image.new("RGB", (num_x * tile_width, num_y * tile_height))
        default_tile = Image.new("RGB", (tile_width, tile_height), (0, 0, 0))

        for y in tqdm(range(num_y), desc=f"Stitching {basename}", leave=False):
            for x in range(num_x):
                coord_x = min_x + x * step_x
                coord_y = min_y + y * step_y
                start_x = x * tile_width
                start_y = y * tile_height

                if (coord_x, coord_y) in coordinates:
                    tile = Image.open(
                        os.path.join(input_folder, coordinates[(coord_x, coord_y)])
                    )
                else:
                    tile = default_tile

                final_image.paste(tile, (start_x, start_y))

        output_path = os.path.join(output_folder, f"{basename}_joined.png")
        final_image.save(output_path)
        logger.info("Saved %s", output_path)


def stitch_tiles(input_folder, output_folder):
    """Stitch all tiles in a folder into a single joined image.

    Args:
        input_folder: Directory containing tile PNG files.
        output_folder: Directory to save the joined image as 'joined.png'.
    """
    os.makedirs(output_folder, exist_ok=True)

    files = [f for f in os.listdir(input_folder) if f.endswith(".png")]
    coordinates = {}

    for file in files:
        try:
            x, y = _parse_coordinates(file)
            coordinates[(x, y)] = file
        except (ValueError, IndexError):
            continue

    if not coordinates:
        raise ValueError("No images found for stitching")

    x_coords = sorted({x for x, y in coordinates.keys()})
    y_coords = sorted({y for x, y in coordinates.keys()})

    step_x = x_coords[1] - x_coords[0] if len(x_coords) > 1 else 0
    step_y = y_coords[1] - y_coords[0] if len(y_coords) > 1 else 0

    min_x, max_x = min(x_coords), max(x_coords)
    min_y, max_y = min(y_coords), max(y_coords)

    num_x = ((max_x - min_x) // step_x) + 1 if step_x != 0 else 1
    num_y = ((max_y - min_y) // step_y) + 1 if step_y != 0 else 1

    first_image = Image.open(
        os.path.join(input_folder, coordinates[(min_x, min_y)])
    )
    tile_width, tile_height = first_image.size

    final_image = Image.new("RGB", (num_x * tile_width, num_y * tile_height))
    default_tile = Image.new("RGB", (tile_width, tile_height), (0, 0, 0))

    for y in tqdm(range(num_y), desc="Stitching rows", leave=False):
        for x in range(num_x):
            coord_x = min_x + x * step_x
            coord_y = min_y + y * step_y
            start_x = x * tile_width
            start_y = y * tile_height

            if (coord_x, coord_y) in coordinates:
                tile = Image.open(
                    os.path.join(input_folder, coordinates[(coord_x, coord_y)])
                )
            else:
                tile = default_tile

            final_image.paste(tile, (start_x, start_y))

    output_path = os.path.join(output_folder, "joined.png")
    final_image.save(output_path)
    logger.info("Saved %s", output_path)
