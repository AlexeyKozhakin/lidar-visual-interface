"""
KNN-based 3D-to-2D feature encoding.

Converts LAS point clouds into 7-channel 2D tensors via K-nearest-neighbor
aggregation on a regular grid.
"""

import logging
import os
from multiprocessing import Pool

import laspy
import numpy as np
from scipy.ndimage import uniform_filter
from scipy.spatial import cKDTree

from musac_las_classifier.constants import (
    DEFAULT_K_NN,
    DEFAULT_M_TENSOR_SIZE,
    DEFAULT_NUM_POINTS_LIM,
    FEATURE_INPUT_TENSOR,
    FEATURE_OUTPUT_TENSOR,
)

logger = logging.getLogger(__name__)


def smooth_2d(data, window_size):
    """2D smoothing with boundary reflection."""
    return uniform_filter(data, size=window_size, mode="reflect")


def load_las_to_numpy(file_path, num_points_lim=DEFAULT_NUM_POINTS_LIM):
    """Load a LAS file and return a normalized numpy array.

    Returns an (N, 7) array with columns [x, y, z, r, g, b, class].
    Coordinates are shifted to start at 0. RGB is normalized to [0, 256].
    Points are resampled to exactly num_points_lim.
    """
    las = laspy.read(file_path)

    max_color_value = np.max([
        np.max(las.red - np.min(las.red)),
        np.max(las.green - np.min(las.green)),
        np.max(las.blue - np.min(las.blue)),
    ])
    if max_color_value == 0:
        max_color_value = 1  # avoid division by zero

    points = np.vstack((
        las.x - np.min(las.x),
        las.y - np.min(las.y),
        las.z - np.min(las.z),
        (las.red - np.min(las.red)) / max_color_value * 256,
        (las.green - np.min(las.green)) / max_color_value * 256,
        (las.blue - np.min(las.blue)) / max_color_value * 256,
    )).T

    classes = np.array(las.classification, dtype=np.int64)
    num_points = points.shape[0]

    if num_points >= num_points_lim:
        indices = np.random.choice(num_points, num_points_lim, replace=False)
    else:
        logger.info(
            "File '%s': %d points < limit %d, resampling with replacement",
            file_path, num_points, num_points_lim,
        )
        indices = np.random.choice(num_points, num_points_lim, replace=True)

    sampled_points = points[indices]
    sampled_classes = classes[indices]
    return np.hstack((sampled_points, sampled_classes.reshape(-1, 1)))


def get_knn_data(data, M, K, knn_eps=0.0, knn_workers=1):
    """Find K nearest neighbors for each point on an M x M grid.

    Args:
        data: (N, 7) array of point data.
        M: Grid resolution.
        K: Number of nearest neighbors.

    Returns:
        Tuple of (knn_data, grid) where knn_data is (M, M, K, 7)
        and grid is (M, M, 2).
    """
    N, D = data.shape

    x_min, x_max = np.min(data[:, 0]), np.max(data[:, 0])
    y_min, y_max = np.min(data[:, 1]), np.max(data[:, 1])

    x_lin = np.linspace(x_min, x_max, M)
    y_lin = np.linspace(y_min, y_max, M)
    grid_x, grid_y = np.meshgrid(x_lin, y_lin)
    grid = np.stack([grid_x, grid_y], axis=-1)

    tree = cKDTree(data[:, :2])
    grid_flat = grid.reshape(-1, 2)
    query_kwargs = {"k": K}
    if knn_eps and knn_eps > 0:
        query_kwargs["eps"] = knn_eps
    if knn_workers is not None:
        query_kwargs["workers"] = knn_workers

    try:
        _, knn_indices = tree.query(grid_flat, **query_kwargs)
    except TypeError:
        # Backward compatibility with older SciPy versions that do not
        # support "workers".
        query_kwargs.pop("workers", None)
        _, knn_indices = tree.query(grid_flat, **query_kwargs)
    knn_data = data[knn_indices].reshape(M, M, K, D)

    return knn_data, grid


def compute_features(data_knn, grid, feature_input_tensor, feature_output_tensor,
                     m_tensor_size=DEFAULT_M_TENSOR_SIZE):
    """Compute feature channels from KNN data.

    Args:
        data_knn: (M, M, K, D) KNN neighbor data.
        grid: (M, M, 2) grid coordinates.
        feature_input_tensor: Input channel index mapping.
        feature_output_tensor: Output channel index mapping.
        m_tensor_size: Grid resolution (used for smoothing window).

    Returns:
        (M, M, C) feature tensor.
    """
    M, _, K, D = data_knn.shape
    C = len(feature_output_tensor)

    data_result = np.zeros((M, M, C), dtype=np.float32)

    idx_x = feature_input_tensor["x"]
    idx_y = feature_input_tensor["y"]
    idx_z = feature_input_tensor["z"]
    idx_r = feature_input_tensor["r"]
    idx_g = feature_input_tensor["g"]
    idx_b = feature_input_tensor["b"]
    idx_class = feature_input_tensor["class"]

    # z_mean: terrain-corrected elevation
    if "z_mean" in feature_output_tensor:
        z_feature = np.mean(data_knn[:, :, :, idx_z], axis=2)
        smoothed_profile = smooth_2d(z_feature, m_tensor_size)
        z_feature_adjusted = z_feature - smoothed_profile
        z_feature_adjusted -= np.min(z_feature_adjusted)
        data_result[:, :, feature_output_tensor["z_mean"]] = z_feature_adjusted

    # z_std: elevation standard deviation
    if "z_std" in feature_output_tensor:
        data_result[:, :, feature_output_tensor["z_std"]] = np.std(
            data_knn[:, :, :, idx_z], axis=2
        )

    # n_r: radial component of surface normal proxy
    if "n_r" in feature_output_tensor:
        std_x = np.std(data_knn[:, :, :, idx_x], axis=2)
        std_y = np.std(data_knn[:, :, :, idx_y], axis=2)
        std_z = np.std(data_knn[:, :, :, idx_z], axis=2)
        denom = np.sqrt(std_x**2 + std_y**2 + std_z**2)
        denom = np.where(denom == 0, 1e-8, denom)
        data_result[:, :, feature_output_tensor["n_r"]] = (
            np.sqrt(std_x**2 + std_y**2) / denom
        )

    # n_z: vertical component of surface normal proxy
    if "n_z" in feature_output_tensor:
        std_x = np.std(data_knn[:, :, :, idx_x], axis=2)
        std_y = np.std(data_knn[:, :, :, idx_y], axis=2)
        std_z = np.std(data_knn[:, :, :, idx_z], axis=2)
        denom = np.sqrt(std_x**2 + std_y**2 + std_z**2)
        denom = np.where(denom == 0, 1e-8, denom)
        data_result[:, :, feature_output_tensor["n_z"]] = std_z / denom

    # dist_mean: mean distance to grid center
    if "dist_mean" in feature_output_tensor:
        dist = np.sqrt(
            (data_knn[:, :, :, idx_x] - grid[:, :, 0, None]) ** 2
            + (data_knn[:, :, :, idx_y] - grid[:, :, 1, None]) ** 2
        )
        data_result[:, :, feature_output_tensor["dist_mean"]] = np.mean(dist, axis=2)

    # RGB: mean color values
    if "r" in feature_output_tensor:
        data_result[:, :, feature_output_tensor["r"]] = np.mean(
            data_knn[:, :, :, idx_r], axis=-1
        )
    if "g" in feature_output_tensor:
        data_result[:, :, feature_output_tensor["g"]] = np.mean(
            data_knn[:, :, :, idx_g], axis=-1
        )
    if "b" in feature_output_tensor:
        data_result[:, :, feature_output_tensor["b"]] = np.mean(
            data_knn[:, :, :, idx_b], axis=-1
        )

    # class: nearest neighbor class label
    if "class" in feature_output_tensor:
        data_result[:, :, feature_output_tensor["class"]] = data_knn[:, :, 0, idx_class]

    return data_result


def _process_single_file(filename, input_directory, output_directory,
                         feature_input_tensor, feature_output_tensor,
                         num_points_lim, M, K, knn_eps, knn_workers):
    """Encode a single LAS file to a feature tensor."""
    input_file = os.path.join(input_directory, filename)
    name, _ = os.path.splitext(filename)
    output_file = os.path.join(output_directory, name)

    data_org = load_las_to_numpy(input_file, num_points_lim=num_points_lim)
    data_knn, grid = get_knn_data(
        data_org,
        M,
        K,
        knn_eps=knn_eps,
        knn_workers=knn_workers,
    )
    logger.info("Encoding %s", filename)
    data_result = compute_features(
        data_knn, grid, feature_input_tensor, feature_output_tensor, m_tensor_size=M
    )
    np.save(output_file, data_result)


def encode_las_to_tensors(input_directory, output_directory,
                          feature_input_tensor=None, feature_output_tensor=None,
                          num_points_lim=DEFAULT_NUM_POINTS_LIM,
                          M=DEFAULT_M_TENSOR_SIZE, K=DEFAULT_K_NN,
                          parallel=False, knn_eps=0.0, knn_workers=1):
    """Encode all LAS files in a directory to feature tensors.

    Args:
        input_directory: Directory with LAS files.
        output_directory: Directory to save .npy tensors.
        feature_input_tensor: Input channel mapping (defaults to standard).
        feature_output_tensor: Output channel mapping (defaults to standard).
        num_points_lim: Target points per tile.
        M: Grid resolution.
        K: Number of nearest neighbors.
        parallel: Use multiprocessing if True.
    """
    if feature_input_tensor is None:
        feature_input_tensor = FEATURE_INPUT_TENSOR
    if feature_output_tensor is None:
        feature_output_tensor = FEATURE_OUTPUT_TENSOR

    os.makedirs(output_directory, exist_ok=True)
    filenames = [f for f in os.listdir(input_directory) if f.lower().endswith(".las")]

    if not filenames:
        logger.warning("No LAS files found in %s", input_directory)
        return

    logger.info("Encoding %d LAS files to tensors", len(filenames))

    args_list = [
        (f, input_directory, output_directory,
         feature_input_tensor, feature_output_tensor,
         num_points_lim, M, K, knn_eps, knn_workers)
        for f in filenames
    ]

    if parallel and len(filenames) > 1:
        num_processes = min(os.cpu_count() or 1, len(filenames))
        logger.info("Using %d processes", num_processes)
        with Pool(processes=num_processes) as pool:
            pool.starmap(_process_single_file, args_list)
    else:
        for args in args_list:
            _process_single_file(*args)
