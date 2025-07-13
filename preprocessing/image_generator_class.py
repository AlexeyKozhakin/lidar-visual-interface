import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from multiprocessing import Pool

def visual_tensor_class(input_dir, filename,
                        map_classes,
                        output_dir):
    """
    Функция для визуализации и сохранения выбранных каналов тензора как изображения.

    Аргументы:
    - input_dir: директория с входными данными (не используется в данной функции, но можно для логирования).
    - filename: имя файла для сохранения изображений.
    - data: тензор размерности (M, M, C), где C - количество каналов.
    - feature_output_tensor: словарь, содержащий соответствие между названиями каналов и их индексами.
    - channels_visualisation: словарь с названиями каналов для визуализации и их индексами в тензоре.
    - output_dir: директория для сохранения изображений.
    """
    file_path = os.path.join(input_dir, filename)
    data = np.load(file_path)  # Загрузка
    # Создаем выходную директорию, если она не существует
    os.makedirs(output_dir, exist_ok=True)

    # Сначала создаем пустое изображение для 3 канала (RGB)
    image_data = np.zeros((data.shape[0], data.shape[1], 3), dtype=np.uint8)
    class_map = data[:, :, 6].astype(np.uint8)  # (H, W)

    # Шаг 1: создаём массив LUT (таблица соответствия), где индекс — класс, а значение — цвет [R, G, B]
    max_class = max(map_classes.keys())
    lut = np.zeros((max_class + 1, 3), dtype=np.uint8)
    for k, v in map_classes.items():
        lut[k] = v
    # Шаг 2: применяем LUT к каждому пикселю
    image_data = lut[class_map]  # (H, W, 3)

    name, _ = os.path.splitext(filename)
    output_path = os.path.join(output_dir, f"{name}.png")
    img = Image.fromarray(image_data)
    img.save(output_path)


def main_not_parallel_tensor_to_image(input_dir, output_dir,
                                      feature_output_tensor, channels_visualisation):

    """
    Параллельная нарезка всех LAS-файлов в директории.

    :param input_directory: Директория с исходными LAS-файлами
    :param output_directory: Директория для сохранения нарезанных файлов
    :param tile_size: Размер tile (в метрах)
    :param num_processes: Количество процессов для параллельной обработки
    """
    # Создаем выходную директорию, если ее нет
    os.makedirs(output_dir, exist_ok=True)

    # Получаем список файлов .las
    filenames = [f for f in os.listdir(input_dir) if f.endswith('.npy')]

    for filename in filenames:
        visual_tensor(input_dir, filename, feature_output_tensor, channels_visualisation, output_dir)
