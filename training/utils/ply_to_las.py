"""
Utilities for converting STPLS3D PLY files to normalized/stretched LAS data.

The pipeline implemented here:
1. Read PLY vertex attributes.
2. Normalize XY translation so that min(x)=0 and min(y)=0.
3. Stretch XY to the nearest multiple of tile size.
4. Split into fixed-size tiles.
5. Downsample only when tile density is above the configured target.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import laspy
import numpy as np
from plyfile import PlyData


CLASS_CANDIDATES = ("class", "classification", "scalar_Label", "label")
RGB_CANDIDATES = (("red", "green", "blue"), ("r", "g", "b"))


@dataclass
class SceneData:
    """In-memory point cloud with metadata after normalization/stretch."""

    stem: str
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    classification: np.ndarray
    red: np.ndarray
    green: np.ndarray
    blue: np.ndarray
    min_x_before_shift: float
    min_y_before_shift: float
    size_x_before_stretch: float
    size_y_before_stretch: float
    target_x: float
    target_y: float
    sx: float
    sy: float


def _find_field(dtype_names: Iterable[str], candidates: Iterable[str]) -> str:
    lower_map = {name.lower(): name for name in dtype_names}
    for candidate in candidates:
        if candidate.lower() in lower_map:
            return lower_map[candidate.lower()]
    raise KeyError(f"Missing required field. Tried candidates: {tuple(candidates)}")


def _find_rgb_fields(dtype_names: Iterable[str]) -> Tuple[str, str, str]:
    lower_map = {name.lower(): name for name in dtype_names}
    for r_name, g_name, b_name in RGB_CANDIDATES:
        if r_name in lower_map and g_name in lower_map and b_name in lower_map:
            return lower_map[r_name], lower_map[g_name], lower_map[b_name]
    raise KeyError("Missing RGB fields. Expected (red,green,blue) or (r,g,b).")


def _normalize_rgb(values: np.ndarray) -> np.ndarray:
    """
    Convert RGB channel to uint16.

    - If source values fit into uint8 range, scale to 16-bit using *257.
    - Otherwise clip to uint16 range.
    """
    values = np.asarray(values)
    max_value = float(np.nanmax(values))
    if max_value <= 255.0:
        return np.clip(values, 0, 255).astype(np.uint16) * 257
    return np.clip(values, 0, 65535).astype(np.uint16)


def _stable_seed(base_seed: int, *parts: Any) -> int:
    key = "|".join(str(part) for part in parts).encode("utf-8")
    digest = hashlib.blake2b(key, digest_size=8).digest()
    value = int.from_bytes(digest, byteorder="little", signed=False)
    return (value ^ int(base_seed)) & 0xFFFFFFFF


def _compute_target_size(size: float, tile_size: int) -> float:
    if size <= 0:
        return float(tile_size)
    tiles = int(math.ceil(size / float(tile_size)))
    return float(tiles * tile_size)


def read_and_prepare_scene(ply_path: Path, tile_size: int) -> SceneData:
    """Read PLY and return normalized + stretched arrays."""
    ply_data = PlyData.read(str(ply_path))
    if "vertex" not in ply_data:
        raise ValueError(f"{ply_path} has no 'vertex' element.")

    vertex = ply_data["vertex"].data
    names = vertex.dtype.names or ()
    if not names:
        raise ValueError(f"{ply_path} has no vertex fields.")

    class_field = _find_field(names, CLASS_CANDIDATES)
    r_field, g_field, b_field = _find_rgb_fields(names)

    x = np.asarray(vertex["x"], dtype=np.float64)
    y = np.asarray(vertex["y"], dtype=np.float64)
    z = np.asarray(vertex["z"], dtype=np.float64)
    cls = np.asarray(vertex[class_field], dtype=np.int64)
    red = _normalize_rgb(vertex[r_field])
    green = _normalize_rgb(vertex[g_field])
    blue = _normalize_rgb(vertex[b_field])

    if x.size == 0:
        raise ValueError(f"{ply_path} has zero points.")
    if np.isnan(x).any() or np.isnan(y).any() or np.isnan(z).any():
        raise ValueError(f"{ply_path} contains NaN coordinates.")

    cls_u8 = np.clip(cls, 0, 255).astype(np.uint8)

    min_x = float(np.min(x))
    min_y = float(np.min(y))
    x_shifted = x - min_x
    y_shifted = y - min_y

    size_x = float(np.max(x_shifted))
    size_y = float(np.max(y_shifted))
    target_x = _compute_target_size(size_x, tile_size)
    target_y = _compute_target_size(size_y, tile_size)

    sx = 1.0 if size_x == 0 else target_x / size_x
    sy = 1.0 if size_y == 0 else target_y / size_y
    x_stretched = x_shifted * sx
    y_stretched = y_shifted * sy

    return SceneData(
        stem=ply_path.stem,
        x=x_stretched,
        y=y_stretched,
        z=z,
        classification=cls_u8,
        red=red,
        green=green,
        blue=blue,
        min_x_before_shift=min_x,
        min_y_before_shift=min_y,
        size_x_before_stretch=size_x,
        size_y_before_stretch=size_y,
        target_x=target_x,
        target_y=target_y,
        sx=sx,
        sy=sy,
    )


def write_las(
    las_path: Path,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    classification: np.ndarray,
    red: np.ndarray,
    green: np.ndarray,
    blue: np.ndarray,
) -> None:
    """Write arrays to LAS (point format 3, version 1.2)."""
    las = laspy.create(file_version="1.2", point_format=3)
    las.header.scales = np.array([0.001, 0.001, 0.001], dtype=np.float64)
    las.header.offsets = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    las.x = x
    las.y = y
    las.z = z
    las.classification = classification
    las.red = red
    las.green = green
    las.blue = blue
    las_path.parent.mkdir(parents=True, exist_ok=True)
    las.write(str(las_path))


def tile_scene(
    scene: SceneData,
    tile_size: int,
    target_points_per_tile: int,
    seed: int,
    tiles_output_dir: Path,
) -> Dict[str, Any]:
    """
    Split scene into tiles and write LAS files.

    Returns:
        Dict with aggregate file stats and per-tile records.
    """
    nx = int(math.ceil(scene.target_x / float(tile_size)))
    ny = int(math.ceil(scene.target_y / float(tile_size)))
    expected_tiles = nx * ny

    eps = 1e-9
    records: List[Dict[str, Any]] = []
    actual_tiles = 0
    empty_tiles = 0
    low_density_tiles = 0
    downsampled_tiles = 0
    total_points_after = 0
    non_empty_after_counts: List[int] = []

    for ix in range(nx):
        x0 = ix * tile_size
        x1 = x0 + tile_size
        x_mask = (
            (scene.x >= x0) & (scene.x <= x1 + eps)
            if ix == nx - 1
            else (scene.x >= x0) & (scene.x < x1)
        )

        for iy in range(ny):
            y0 = iy * tile_size
            y1 = y0 + tile_size
            y_mask = (
                (scene.y >= y0) & (scene.y <= y1 + eps)
                if iy == ny - 1
                else (scene.y >= y0) & (scene.y < y1)
            )
            mask = x_mask & y_mask
            idx = np.flatnonzero(mask)
            points_before = int(idx.size)

            if points_before == 0:
                empty_tiles += 1
                records.append(
                    {
                        "file_name": scene.stem,
                        "tile_id": f"{scene.stem}_{x0}_{y0}",
                        "ix": ix,
                        "iy": iy,
                        "bbox_min_x": x0,
                        "bbox_min_y": y0,
                        "bbox_max_x": x1,
                        "bbox_max_y": y1,
                        "points_before": 0,
                        "points_after": 0,
                        "is_low_density": False,
                        "is_downsampled": False,
                    }
                )
                continue

            is_downsampled = False
            if points_before > target_points_per_tile:
                tile_seed = _stable_seed(seed, scene.stem, ix, iy)
                rng = np.random.default_rng(tile_seed)
                idx = rng.choice(idx, size=target_points_per_tile, replace=False)
                is_downsampled = True
                downsampled_tiles += 1

            points_after = int(idx.size)
            is_low_density = points_after < target_points_per_tile
            if is_low_density:
                low_density_tiles += 1

            tile_name = f"{scene.stem}_{x0}_{y0}.las"
            tile_path = tiles_output_dir / tile_name
            write_las(
                tile_path,
                scene.x[idx],
                scene.y[idx],
                scene.z[idx],
                scene.classification[idx],
                scene.red[idx],
                scene.green[idx],
                scene.blue[idx],
            )

            actual_tiles += 1
            total_points_after += points_after
            non_empty_after_counts.append(points_after)
            records.append(
                {
                    "file_name": scene.stem,
                    "tile_id": f"{scene.stem}_{x0}_{y0}",
                    "ix": ix,
                    "iy": iy,
                    "bbox_min_x": x0,
                    "bbox_min_y": y0,
                    "bbox_max_x": x1,
                    "bbox_max_y": y1,
                    "points_before": points_before,
                    "points_after": points_after,
                    "is_low_density": is_low_density,
                    "is_downsampled": is_downsampled,
                }
            )

    if non_empty_after_counts:
        min_points = int(np.min(non_empty_after_counts))
        max_points = int(np.max(non_empty_after_counts))
        mean_points = float(np.mean(non_empty_after_counts))
    else:
        min_points = 0
        max_points = 0
        mean_points = 0.0

    return {
        "expected_tiles": expected_tiles,
        "actual_tiles": actual_tiles,
        "empty_tiles": empty_tiles,
        "low_density_tiles": low_density_tiles,
        "downsampled_tiles": downsampled_tiles,
        "total_points_after": total_points_after,
        "min_points_in_tile": min_points,
        "max_points_in_tile": max_points,
        "mean_points_in_tile": mean_points,
        "per_tile_records": records,
    }
