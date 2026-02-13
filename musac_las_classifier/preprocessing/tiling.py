"""
Spatial tiling of LAS files into fixed-size blocks.

Splits large LAS point clouds into 250m x 250m (configurable) tiles
for downstream processing.
"""

import logging
import os
from multiprocessing import Pool

import laspy

logger = logging.getLogger(__name__)


def split_las(input_path, output_dir, tile_size=250, train_mode=False):
    """Split a single LAS file into square tiles.

    Args:
        input_path: Path to input LAS file.
        output_dir: Directory to save output tiles.
        tile_size: Tile side length in meters.
        train_mode: If True, use training naming convention (prefix with base name).
    """
    os.makedirs(output_dir, exist_ok=True)
    las = laspy.read(input_path)
    base_name = os.path.splitext(os.path.basename(input_path))[0]

    if not train_mode:
        try:
            x0_km, y0_km = map(int, base_name.split("_"))
        except ValueError:
            logger.warning(
                "Could not parse coordinates from filename '%s', using (0, 0)",
                base_name,
            )
            x0_km, y0_km = 0, 0
    else:
        x0_km, y0_km = 0, 0

    x0 = x0_km * 1000
    y0 = y0_km * 1000
    x_min, y_min = x0, y0

    xs = las.x
    ys = las.y

    for i in range(0, 1000, tile_size):
        for j in range(0, 1000, tile_size):
            tile_x_min = x_min + i
            tile_x_max = tile_x_min + tile_size
            tile_y_min = y_min + j
            tile_y_max = tile_y_min + tile_size

            mask = (
                (xs >= tile_x_min)
                & (xs < tile_x_max)
                & (ys >= tile_y_min)
                & (ys < tile_y_max)
            )
            selected_points = las.points[mask]

            if len(selected_points) > 0:
                new_las = laspy.LasData(las.header)
                new_las.points = selected_points
                if train_mode:
                    output_name = f"{base_name}_{x0_km}_{y0_km}_{tile_x_min}_{tile_y_min}.las"
                else:
                    output_name = f"{x0_km}_{y0_km}_{tile_x_min}_{tile_y_min}.las"
                output_path = os.path.join(output_dir, output_name)
                new_las.write(output_path)
                logger.info("Saved %s", output_path)


def _process_file(filename, input_directory, output_directory, tile_size, train_mode):
    """Process a single LAS file (worker function for parallel execution)."""
    input_file = os.path.join(input_directory, filename)
    split_las(input_file, output_directory, tile_size=tile_size, train_mode=train_mode)


def tile_las_files(input_directory, output_directory, tile_size=250,
                   train_mode=False, parallel=False):
    """Tile all LAS files in a directory.

    Args:
        input_directory: Directory containing LAS files.
        output_directory: Directory to save tiled output.
        tile_size: Tile side length in meters.
        train_mode: If True, use training naming convention.
        parallel: If True, use multiprocessing.
    """
    os.makedirs(output_directory, exist_ok=True)
    filenames = [f for f in os.listdir(input_directory) if f.lower().endswith(".las")]

    if not filenames:
        logger.warning("No LAS files found in %s", input_directory)
        return

    logger.info("Found %d LAS files to tile", len(filenames))

    if parallel and len(filenames) > 1:
        num_processes = min(os.cpu_count() or 1, len(filenames))
        logger.info("Using %d processes", num_processes)
        with Pool(processes=num_processes) as pool:
            pool.starmap(
                _process_file,
                [
                    (f, input_directory, output_directory, tile_size, train_mode)
                    for f in filenames
                ],
            )
    else:
        for filename in filenames:
            _process_file(
                filename, input_directory, output_directory, tile_size, train_mode
            )
