import os
import laspy
import numpy as np
from multiprocessing import Pool
from scipy.ndimage import uniform_filter


def smooth_2d(data, window_size):
    """Performs 2D smoothing of array with boundary reflection."""
    return uniform_filter(data, size=window_size, mode='reflect')

def get_mesh_grid(data, M):
    """
    Creates uniform grid of size M x M for (x, y) data.
    
    data: numpy array of dimension (N, 2) — coordinates (x, y).
    M: grid size (M x M).
    
    Returns:
    numpy array of dimension (M, M, 2), containing grid coordinates (x, y).
    """
    # Extract x and y coordinates
    x_coords = data[:, 0]
    y_coords = data[:, 1]

    # Define grid boundaries
    x_min, x_max = np.min(x_coords), np.max(x_coords)
    y_min, y_max = np.min(y_coords), np.max(y_coords)

    # Create uniform grid coordinates
    x_grid = np.linspace(x_min, x_max, M)
    y_grid = np.linspace(y_min, y_max, M)

    # Create meshgrid and combine x and y coordinates
    x_mesh, y_mesh = np.meshgrid(x_grid, y_grid)

    # Combine x and y coordinates along the last axis
    grid = np.stack((x_mesh, y_mesh), axis=-1)  # Dimension (M, M, 2)

    return grid


def load_las_to_numpy(file_path, num_points_lim=4096, log_path="log.txt"):
    """
    Processes one LAS file and returns a sample of points and classes as numpy arrays.
    file_path: Path to LAS file.
    num_points_lim: Number of points to sample.
    """
    try:
        las = laspy.read(file_path)

        # Find maximum value among all color channels
        max_color_value = np.max([(np.max(las.red-np.min(las.red))), 
                                  np.max(las.green - np.min(las.green)), 
                                  np.max(las.blue - np.min(las.blue))])
        min_color_value = np.min([np.min(las.red), np.min(las.green), np.min(las.blue)])
        print('max_colour channel =', max_color_value)
        print('min_colour channel =', min_color_value)

        # Extract coordinates and color data
        points = np.vstack((
            las.x - np.min(las.x),                 # Normalize X
            las.y - np.min(las.y),                 # Normalize Y
            las.z - np.min(las.z),                 # Normalize Z
            (las.red - np.min(las.red)) / max_color_value * 256,       # Normalize color (R)
            (las.green - np.min(las.green))/ max_color_value * 256,     # Normalize color (G)
            (las.blue - np.min(las.blue))/ max_color_value * 256       # Normalize color (B)
        )).T  # Dimension (N, 6)

        # Extract classes
        classes = np.array(las.classification, dtype=np.int64)  # Array with classes (N,)

        # Check number of points
        num_points = points.shape[0]
        # if num_points > num_points_lim:
        #     # Случайная выборка точек
        #     indices = np.random.choice(num_points, num_points_lim, replace=False)
        #     sampled_points = points[indices]
        #     sampled_classes = classes[indices]

        #     # Объединяем координаты и классы
        #     return np.hstack((sampled_points, sampled_classes.reshape(-1, 1)))  # (num_points_lim, 7)
        # else:
        #     print(f"Количество точек в файле меньше лимита {num_points_lim}, пропуск файла.")
        #     return None
        if num_points > num_points_lim:
            # Случайная выборка точек
            indices = np.random.choice(num_points, num_points_lim, replace=False)
            sampled_points = points[indices]
            sampled_classes = classes[indices]
            return np.hstack((sampled_points, sampled_classes.reshape(-1, 1)))
        
        else:
            # Logging and message
            msg = f"File '{file_path}': number of points ({num_points}) is less than limit ({num_points_lim}). Resampling will be performed."
            print(msg)
            
            # Write to log file
            with open(log_path, "a", encoding="utf-8") as log_file:
                log_file.write(msg + "\n")

            # Resampling with replacement
            indices = np.random.choice(num_points, num_points_lim, replace=True)
            sampled_points = points[indices]
            sampled_classes = classes[indices]
            return np.hstack((sampled_points, sampled_classes.reshape(-1, 1)))        
    except Exception as e:
        print(f"Ошибка при обработке файла {file_path}: {e}")
        return None

from scipy.spatial import cKDTree

def get_knn_data(data, M, K):
    """
    Search for K nearest neighbors for uniform grid of points.
    
    Args:
        data (np.ndarray): Input tensor of size (N, 7), where N is number of points.
        M (int): Grid size (MxM).
        K (int): Number of nearest neighbors.

    Returns:
        tuple: 
            - np.ndarray: Tensor of nearest neighbors of size (M, M, 7, K).
            - np.ndarray: Coordinate grid of size (M, M, 2).
    """
    N, D = data.shape
    assert 3 <= D <= 7, "Tensor `data` must have from 3 to 7 features (x, y, z, r, g, b, class)."
    
    # Minimum and maximum values of x and y coordinates
    x_min, x_max = np.min(data[:, 0]), np.max(data[:, 0])
    y_min, y_max = np.min(data[:, 1]), np.max(data[:, 1])
    
    # Generate uniform grid (M, M, 2)
    x_lin = np.linspace(x_min, x_max, M)
    y_lin = np.linspace(y_min, y_max, M)
    grid_x, grid_y = np.meshgrid(x_lin, y_lin)
    grid = np.stack([grid_x, grid_y], axis=-1)  # (M, M, 2)

    # Prepare data for KD-tree
    data_coords = data[:, :2]  # (N, 2)
    tree = cKDTree(data_coords)

    # Search for K nearest neighbors for each grid point (M*M, 2)
    grid_flat = grid.reshape(-1, 2)
    dists, knn_indices = tree.query(grid_flat, k=K)  # (M*M, K)

    # Extract nearest neighbor data (M*M, K, D)
    knn_data = data[knn_indices]  # (M*M, K, D)

    # Transform to shape (M, M, 7, K)
    knn_data = knn_data.reshape(M, M, K, D)#.transpose(0, 1, 3, 2)  # (M, M, D, K)

    return knn_data, grid

from scipy.stats import mode


def fast_mode(array, axis=2):
    """
    Вычисляет моду по оси axis на чистом NumPy.
    array: входной массив размерности (M, M, K).
    
    Возвращает:
    - Массив с модой (M, M)
    """
    M, N, K = array.shape  # Размеры входного массива
    reshaped_array = array.reshape(-1, K)  # Преобразуем в (M*M, K)

    # Вычисляем моду с использованием np.bincount
    mode_result = np.apply_along_axis(lambda x: np.bincount(x.astype(int)).argmax(), 1, reshaped_array)
    
    return mode_result.reshape(M, N)  # Возвращаем обратно в форму (M, M)

def fast_median(array):
    """
    Вычисляет медиану по последней оси массива (M, M, K).
    
    :param array: Входной массив размерности (M, M, K).
    :return: Массив (M, M) с медианными значениями.
    """
    M, N, K = array.shape  # Размерности входного массива
    reshaped_array = array.reshape(-1, K)  # Преобразуем в (M*M, K)

    # Calculate median for each row
    median_result = np.median(reshaped_array, axis=1)

    return median_result.reshape(M, N)  # Transform back to (M, M)


def compute_features(data_knn, grid, feature_input_tensor, feature_output_tensor):
    """
    Forms data_result tensor based on data_knn (M, M, K, D) and grid (M, M, 2).

    Arguments:
    - data_knn: numpy array of dimension (M, M, K, D), storing K nearest neighbors.
    - grid: numpy array of dimension (M, M, 2), containing x, y coordinates.
    - feature_input_tensor: dictionary with input channel indices.
    - feature_output_tensor: dictionary with output channel indices.

    Returns:
    - data_result: numpy array of dimension (M, M, C), containing computed features.
    """
    print('shape:', data_knn.shape)
    print('len feature:', len(feature_output_tensor))
    M, _, K, D = data_knn.shape
    C = len(feature_output_tensor)  # Number of output channels
    print('D=',D)
    if D < len(feature_input_tensor):
        print('D=',D)
        print('len feature:', len(feature_output_tensor))
        print('shape:', data_knn.shape)
        raise ValueError("Tensor dimensionality is smaller than specified in the configuration file.")



    # Allocate memory for output tensor
    data_result = np.zeros((M, M, C), dtype=np.float32)

    # Extract input channel indices
    idx_x = feature_input_tensor["x"]
    idx_y = feature_input_tensor["y"]
    idx_z = feature_input_tensor["z"]
    idx_r = feature_input_tensor["r"]
    idx_g = feature_input_tensor["g"]
    idx_b = feature_input_tensor["b"]
    idx_class = feature_input_tensor["class"]

    # --- Compute features ---
    
    # z_mean: mean value along K axis for z-coordinate
    if "z_mean" in feature_output_tensor:
        z_feature = np.mean(data_knn[:, :, :, idx_z], axis=2)

        # # Profile smoothing
        window_size = 512  # Set appropriate window
        smoothed_profile = smooth_2d(z_feature, window_size)

        # # Feature correction: subtract smoothed profile and make non-negative
        z_feature_adjusted = z_feature - smoothed_profile
        z_feature_adjusted -= np.min(z_feature_adjusted)

        # Save result to final tensor
        data_result[:, :, feature_output_tensor["z_mean"]] = z_feature_adjusted
    
    # z_std: standard deviation along K axis for z-coordinate
    if "z_std" in feature_output_tensor:
        data_result[:, :, feature_output_tensor["z_std"]] = np.std(data_knn[:, :, :, idx_z], axis=2)

    # n_r: radial component of surface normal
    if "n_r" in feature_output_tensor:
        std_x = np.std(data_knn[:, :, :, idx_x], axis=2)
        std_y = np.std(data_knn[:, :, :, idx_y], axis=2)
        std_z = np.std(data_knn[:, :, :, idx_z], axis=2)
        data_result[:, :, feature_output_tensor["n_r"]] = (std_x**2+std_y**2)**(1/2)/(std_x**2+std_y**2+std_z**2)**(1/2)

    # n_z: vertical component of surface normal
    if "n_z" in feature_output_tensor:
        std_x = np.std(data_knn[:, :, :, idx_x], axis=2)
        std_y = np.std(data_knn[:, :, :, idx_y], axis=2)
        std_z = np.std(data_knn[:, :, :, idx_z], axis=2)
        data_result[:, :, feature_output_tensor["n_z"]] = std_z/(std_x**2+std_y**2+std_z**2)**(1/2)                
    
    # dist_mean: mean distance from K neighbors to central point from grid
    if "dist_mean" in feature_output_tensor:
        dist = np.sqrt(
            (data_knn[:, :, :, idx_x] - grid[:, :, 0, None]) ** 2 +
            (data_knn[:, :, :, idx_y] - grid[:, :, 1, None]) ** 2
        )
        data_result[:, :, feature_output_tensor["dist_mean"]] = np.mean(dist, axis=2)
    
    # r, g, b: mode of values among K neighbors (optimized version)
    # if "r" in feature_output_tensor:
    #     data_result[:, :, feature_output_tensor["r"]] = fast_mode(data_knn[:, :, :, idx_r])
    # if "g" in feature_output_tensor:
    #     data_result[:, :, feature_output_tensor["g"]] = fast_mode(data_knn[:, :, :, idx_g])
    # if "b" in feature_output_tensor:
    #     data_result[:, :, feature_output_tensor["b"]] = fast_mode(data_knn[:, :, :, idx_b])

    if "r" in feature_output_tensor:
        data_result[:, :, feature_output_tensor["r"]] = np.mean(data_knn[:, :, :, idx_r], axis=-1)
    if "g" in feature_output_tensor:
        data_result[:, :, feature_output_tensor["g"]] = np.mean(data_knn[:, :, :, idx_g], axis=-1)
    if "b" in feature_output_tensor:
        data_result[:, :, feature_output_tensor["b"]] = np.mean(data_knn[:, :, :, idx_b], axis=-1)

    # if "r" in feature_output_tensor:
    #     data_result[:, :, feature_output_tensor["r"]] = fast_median(data_knn[:, :, :, idx_r])
    # if "g" in feature_output_tensor:
    #     data_result[:, :, feature_output_tensor["g"]] = fast_median(data_knn[:, :, :, idx_g])
    # if "b" in feature_output_tensor:
    #     data_result[:, :, feature_output_tensor["b"]] = fast_median(data_knn[:, :, :, idx_b])        

    # class: mode among K neighbors
    if "class" in feature_output_tensor:
        data_result[:, :, feature_output_tensor["class"]] = data_knn[:, :, 0, idx_class]
        print(data_knn[:, :, :, idx_class])
    
    return data_result


def main_parallel_transform_to_tensor(input_directory, output_directory,
                                          feature_input_tensor, feature_output_tensor,
                                           num_points_lim, M, K):
    
    """
    Parallel processing of all LAS files in directory.

    :param input_directory: Directory with source LAS files
    :param output_directory: Directory to save processed files
    :param feature_input_tensor: Input tensor configuration
    :param feature_output_tensor: Output tensor configuration
    :param num_points_lim: Point limit per file
    :param M: Tensor size
    :param K: Number of nearest neighbors
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_directory, exist_ok=True)

    # Get list of .las files
    filenames = [f for f in os.listdir(input_directory) if f.endswith('.las')]

    # Get memory information
    #mem = psutil.virtual_memory()

    # Available memory in bytes
    #mem_for_tensor_needed = 2
    #available_memory = mem.available
    #how_many_possible_processors = max(1, int(np.floor(available_memory/ (1024 ** 3)/2)))
    num_processes = min(os.cpu_count(), len(filenames))
    print(num_processes)
    print(filenames)
    with Pool(processes=num_processes) as pool:
        pool.starmap(process_transform, [(filename, input_directory, output_directory,
                                          feature_input_tensor, feature_output_tensor,num_points_lim, 
                                          M, K) for filename in filenames])
        
def main_not_parallel_transform_to_tensor(input_directory, output_directory,
                                          feature_input_tensor, feature_output_tensor,
                                           num_points_lim, M, K):
    
    """
    Sequential processing of all LAS files in directory.

    :param input_directory: Directory with source LAS files
    :param output_directory: Directory to save processed files
    :param feature_input_tensor: Input tensor configuration
    :param feature_output_tensor: Output tensor configuration
    :param num_points_lim: Point limit per file
    :param M: Tensor size
    :param K: Number of nearest neighbors
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_directory, exist_ok=True)

    # Get list of .las files
    filenames = [f for f in os.listdir(input_directory) if f.endswith('.las')]

    for filename in filenames:
        process_transform(filename, input_directory, output_directory,
                      feature_input_tensor, feature_output_tensor, num_points_lim, M, K)
                
        
def process_transform(filename, input_directory, output_directory,
                      feature_input_tensor, feature_output_tensor, num_points_lim, M, K):
    input_file = os.path.join(input_directory, filename)
    name, _ = os.path.splitext(filename)
    output_file = os.path.join(output_directory, name)
    data_org = load_las_to_numpy(input_file, num_points_lim=num_points_lim)

    data_knn, grid = get_knn_data(data_org, M, K)
    print(f'file {filename} is processing')
    data_result = compute_features(data_knn, grid, feature_input_tensor, feature_output_tensor)
    np.save(output_file, data_result)  # Save

if __name__ == "__main__":
    import config_preprocessing as cp
    import time
    #input_directory = cp.path_input_las_for_2d  # Path to directory with LAS files
    input_directory = cp.path_input_las_for_2d  # Path to directory with LAS files
    output_directory = cp.path_out_tensors
    M = cp.M_tensor_size
    K = cp.K_nn
    start = time.time()
    main_parallel_transform_to_tensor(input_directory, output_directory,
                                          cp.feature_input_tensor, cp.feature_output_tensor, 
                                          cp.num_points_lim, M, K)
    end = time.time()
    print(round((end-start)))