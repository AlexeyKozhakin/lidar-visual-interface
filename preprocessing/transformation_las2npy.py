import os
import laspy
import numpy as np
from multiprocessing import Pool
from scipy.ndimage import uniform_filter


def smooth_2d(data, window_size):
    """Выполняет двумерное сглаживание массива с отражением границ."""
    return uniform_filter(data, size=window_size, mode='reflect')

def get_mesh_grid(data, M):
    """
    Создает равномерную сетку размером M x M для данных (x, y).
    
    data: numpy массив размерности (N, 2) — координаты (x, y).
    M: размер сетки (M x M).
    
    Возвращает:
    numpy массив размерности (M, M, 2), содержащий координаты сетки (x, y).
    """
    # Извлекаем координаты x и y
    x_coords = data[:, 0]
    y_coords = data[:, 1]

    # Определяем границы сетки
    x_min, x_max = np.min(x_coords), np.max(x_coords)
    y_min, y_max = np.min(y_coords), np.max(y_coords)

    # Создаем равномерные координаты сетки
    x_grid = np.linspace(x_min, x_max, M)
    y_grid = np.linspace(y_min, y_max, M)

    # Создаем meshgrid и объединяем x и y координаты
    x_mesh, y_mesh = np.meshgrid(x_grid, y_grid)

    # Объединяем x и y координаты вдоль последней оси
    grid = np.stack((x_mesh, y_mesh), axis=-1)  # Размерность (M, M, 2)

    return grid


def load_las_to_numpy(file_path, num_points_lim=4096):
    """
    Обрабатывает один LAS файл и возвращает выборку точек и классов в виде numpy-массивов.
    file_path: Путь к LAS файлу.
    num_points_lim: Количество точек для выборки.
    """
    try:
        las = laspy.read(file_path)

        # Находим максимальное значение среди всех цветовых каналов
        max_color_value = np.max([(np.max(las.red-np.min(las.red))), 
                                  np.max(las.green - np.min(las.green)), 
                                  np.max(las.blue - np.min(las.blue))])
        min_color_value = np.min([np.min(las.red), np.min(las.green), np.min(las.blue)])
        print('max_colour channel =', max_color_value)
        print('min_colour channel =', min_color_value)

        # Извлечение координат и цветовых данных
        points = np.vstack((
            las.x - np.min(las.x),                 # Нормировка X
            las.y - np.min(las.y),                 # Нормировка Y
            las.z - np.min(las.z),                 # Нормировка Z
            (las.red - np.min(las.red)) / max_color_value * 256,       # Нормировка цвета (R)
            (las.green - np.min(las.green))/ max_color_value * 256,     # Нормировка цвета (G)
            (las.blue - np.min(las.blue))/ max_color_value * 256       # Нормировка цвета (B)
        )).T  # Размерность (N, 6)

        # Извлечение классов
        classes = np.array(las.classification, dtype=np.int64)  # Массив с классами (N,)

        # Проверка количества точек
        num_points = points.shape[0]
        if num_points > num_points_lim:
            # Случайная выборка точек
            indices = np.random.choice(num_points, num_points_lim, replace=False)
            sampled_points = points[indices]
            sampled_classes = classes[indices]

            # Объединяем координаты и классы
            return np.hstack((sampled_points, sampled_classes.reshape(-1, 1)))  # (num_points_lim, 7)
        else:
            print(f"Количество точек в файле меньше лимита {num_points_lim}, пропуск файла.")
            return None
    except Exception as e:
        print(f"Ошибка при обработке файла {file_path}: {e}")
        return None

from scipy.spatial import cKDTree

def get_knn_data(data, M, K):
    """
    Поиск K ближайших соседей для равномерной сетки точек.
    
    Args:
        data (np.ndarray): Входной тензор размером (N, 7), где N — количество точек.
        M (int): Размер сетки (MxM).
        K (int): Количество ближайших соседей.

    Returns:
        tuple: 
            - np.ndarray: Тензор ближайших соседей размером (M, M, 7, K).
            - np.ndarray: Сетка координат размером (M, M, 2).
    """
    N, D = data.shape
    assert 3 <= D <= 7, "Тензор `data` должен иметь от 3 до 7 признаков (x, y, z, r, g, b, class)."
    
    # Минимальные и максимальные значения координат x и y
    x_min, x_max = np.min(data[:, 0]), np.max(data[:, 0])
    y_min, y_max = np.min(data[:, 1]), np.max(data[:, 1])
    
    # Генерация равномерной сетки (M, M, 2)
    x_lin = np.linspace(x_min, x_max, M)
    y_lin = np.linspace(y_min, y_max, M)
    grid_x, grid_y = np.meshgrid(x_lin, y_lin)
    grid = np.stack([grid_x, grid_y], axis=-1)  # (M, M, 2)

    # Подготовка данных для KD-дерева
    data_coords = data[:, :2]  # (N, 2)
    tree = cKDTree(data_coords)

    # Поиск K ближайших соседей для каждой точки сетки (M*M, 2)
    grid_flat = grid.reshape(-1, 2)
    dists, knn_indices = tree.query(grid_flat, k=K)  # (M*M, K)

    # Извлечение данных ближайших соседей (M*M, K, D)
    knn_data = data[knn_indices]  # (M*M, K, D)

    # Преобразование к форме (M, M, 7, K)
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

    # Вычисляем медиану по каждой строке
    median_result = np.median(reshaped_array, axis=1)

    return median_result.reshape(M, N)  # Преобразуем обратно в (M, M)


def compute_features(data_knn, grid, feature_input_tensor, feature_output_tensor):
    """
    Формирует тензор data_result на основе data_knn (M, M, K, D) и grid (M, M, 2).

    Аргументы:
    - data_knn: numpy массив размерности (M, M, K, D), хранящий K ближайших соседей.
    - grid: numpy массив размерности (M, M, 2), содержащий x, y координаты.
    - feature_input_tensor: словарь с индексами входных каналов.
    - feature_output_tensor: словарь с индексами выходных каналов.

    Возвращает:
    - data_result: numpy массив размерности (M, M, C), содержащий вычисленные признаки.
    """
    print('shape:', data_knn.shape)
    print('len feature:', len(feature_output_tensor))
    M, _, K, D = data_knn.shape
    C = len(feature_output_tensor)  # Число выходных каналов
    print('D=',D)
    if D < len(feature_input_tensor):
        print('D=',D)
        print('len feature:', len(feature_output_tensor))
        print('shape:', data_knn.shape)
        raise ValueError("Tensor dimensionality is smaller than specified in the configuration file.")



    # Выделяем память под выходной тензор
    data_result = np.zeros((M, M, C), dtype=np.float32)

    # Извлекаем индексы входных каналов
    idx_x = feature_input_tensor["x"]
    idx_y = feature_input_tensor["y"]
    idx_z = feature_input_tensor["z"]
    idx_r = feature_input_tensor["r"]
    idx_g = feature_input_tensor["g"]
    idx_b = feature_input_tensor["b"]
    idx_class = feature_input_tensor["class"]

    # --- Вычисляем признаки ---
    
    # z_mean: среднее значение по оси K для z-координаты
    if "z_mean" in feature_output_tensor:
        z_feature = np.mean(data_knn[:, :, :, idx_z], axis=2)

        # # Сглаживание профиля
        window_size = 512  # Задайте подходящее окно
        smoothed_profile = smooth_2d(z_feature, window_size)

        # # Коррекция фичи: вычитание сглаженного профиля и приведение к неотрицательному виду
        z_feature_adjusted = z_feature - smoothed_profile
        z_feature_adjusted -= np.min(z_feature_adjusted)

        # Сохранение результата в итоговый тензор
        data_result[:, :, feature_output_tensor["z_mean"]] = z_feature_adjusted
    
    # z_std: стандартное отклонение по оси K для z-координаты
    if "z_std" in feature_output_tensor:
        data_result[:, :, feature_output_tensor["z_std"]] = np.std(data_knn[:, :, :, idx_z], axis=2)

    # z_std: стандартное отклонение по оси K для z-координаты
    if "n_r" in feature_output_tensor:
        std_x = np.std(data_knn[:, :, :, idx_x], axis=2)
        std_y = np.std(data_knn[:, :, :, idx_y], axis=2)
        std_z = np.std(data_knn[:, :, :, idx_z], axis=2)
        data_result[:, :, feature_output_tensor["n_r"]] = (std_x**2+std_y**2)**(1/2)/(std_x**2+std_y**2+std_z**2)**(1/2)

    # z_std: стандартное отклонение по оси K для z-координаты
    if "n_z" in feature_output_tensor:
        std_x = np.std(data_knn[:, :, :, idx_x], axis=2)
        std_y = np.std(data_knn[:, :, :, idx_y], axis=2)
        std_z = np.std(data_knn[:, :, :, idx_z], axis=2)
        data_result[:, :, feature_output_tensor["n_z"]] = std_z/(std_x**2+std_y**2+std_z**2)**(1/2)                
    
    # dist_mean: среднее расстояние от K соседей до центральной точки из grid
    if "dist_mean" in feature_output_tensor:
        dist = np.sqrt(
            (data_knn[:, :, :, idx_x] - grid[:, :, 0, None]) ** 2 +
            (data_knn[:, :, :, idx_y] - grid[:, :, 1, None]) ** 2
        )
        data_result[:, :, feature_output_tensor["dist_mean"]] = np.mean(dist, axis=2)
    
    # r, g, b: мода значений среди K соседей (оптимизированный вариант)
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

    #class: мода среди K соседей
    if "class" in feature_output_tensor:
        data_result[:, :, feature_output_tensor["class"]] = data_knn[:, :, 0, idx_class]
        print(data_knn[:, :, :, idx_class])
    
    return data_result


def main_parallel_transform_to_tensor(input_directory, output_directory,
                                          feature_input_tensor, feature_output_tensor,
                                           num_points_lim, M, K):
    
    """
    Параллельная нарезка всех LAS-файлов в директории.

    :param input_directory: Директория с исходными LAS-файлами
    :param output_directory: Директория для сохранения нарезанных файлов
    :param tile_size: Размер tile (в метрах)
    :param num_processes: Количество процессов для параллельной обработки
    """
    # Создаем выходную директорию, если ее нет
    os.makedirs(output_directory, exist_ok=True)

    # Получаем список файлов .las
    filenames = [f for f in os.listdir(input_directory) if f.endswith('.las')]

    # Получаем информацию о памяти
    #mem = psutil.virtual_memory()

    # Доступная память в байтах
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
        
def process_transform(filename, input_directory, output_directory,
                      feature_input_tensor, feature_output_tensor, num_points_lim, M, K):
    input_file = os.path.join(input_directory, filename)
    name, _ = os.path.splitext(filename)
    output_file = os.path.join(output_directory, name)
    data_org = load_las_to_numpy(input_file, num_points_lim=num_points_lim)

    data_knn, grid = get_knn_data(data_org, M, K)
    print(f'file {filename} is processing')
    data_result = compute_features(data_knn, grid, feature_input_tensor, feature_output_tensor)
    np.save(output_file, data_result)  # Сохранение

if __name__ == "__main__":
    import config_preprocessing as cp
    import time
    #input_directory = cp.path_input_las_for_2d  # Указать путь к каталогу с LAS-файлами
    input_directory = cp.path_input_las_for_2d  # Указать путь к каталогу с LAS-файлами
    output_directory = cp.path_out_tensors
    M = cp.M_tensor_size
    K = cp.K_nn
    start = time.time()
    main_parallel_transform_to_tensor(input_directory, output_directory,
                                          cp.feature_input_tensor, cp.feature_output_tensor, 
                                          cp.num_points_lim, M, K)
    end = time.time()
    print(round((end-start)))