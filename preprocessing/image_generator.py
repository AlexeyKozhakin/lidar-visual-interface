import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from multiprocessing import Pool

def visual_tensor(input_dir, filename, feature_output_tensor, channels_visualisation, output_dir):
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

    # Проходим по каналам, которые нужно визуализировать
    for channel_name, channel_idx in channels_visualisation.items():
        if channel_name in feature_output_tensor:
            # Извлекаем нужный канал
            channel_data = data[:, :, feature_output_tensor[channel_name]]
            print('chanle feature',feature_output_tensor[channel_name])

            # Нормализация канала
            channel_data_normalized = channel_data / np.max(channel_data, axis=(0, 1), keepdims=True)*255
            #channel_data_normalized = channel_data
            print(channel_data_normalized.max())
            print(channel_data_normalized.min())

            # Ограничение значений в диапазоне от 0 до 255 и преобразование в целые числа
            channel_data_normalized = np.clip(channel_data_normalized, 0, 255).astype(np.uint8)

#            Записываем данные в соответствующий канал изображения (например, r, g, или b)
            # if channel_name == "r":
            #     image_data[:, :, 0] = channel_data_normalized  # Канал R
            # elif channel_name == "g":
            #     image_data[:, :, 1] = channel_data_normalized  # Канал G
            # elif channel_name == "b":
            #     image_data[:, :, 2] = channel_data_normalized  # Канал B

            # if channel_name == "class":
            #     image_data[:, :, 0] = channel_data_normalized  # Канал R
            # elif channel_name == "class":
            #     image_data[:, :, 1] = channel_data_normalized  # Канал G
            # elif channel_name == "class":
            #     image_data[:, :, 2] = channel_data_normalized  # Канал B                

            #Записываем данные в соответствующий канал изображения (например, r, g, или b)
            if channel_name == channel_name:
                image_data[:, :, channels_visualisation[channel_name]] = channel_data_normalized  # Канал R

                # Сохраняем изображение как PNG
    name, _ = os.path.splitext(filename)            
    output_path = os.path.join(output_dir, f"{name}.png")
    img = Image.fromarray(image_data)
    img.save(output_path)

def main_parallel_tensor_to_image(input_dir, output_dir,
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

    num_processes = 1 #min(os.cpu_count(), len(filenames))
    print(num_processes)
    print(os.cpu_count())
    print(len(filenames))
    with Pool(processes=num_processes) as pool:
        pool.starmap(visual_tensor, [(input_dir, filename,
                                      feature_output_tensor, channels_visualisation, output_dir) for filename in filenames])



if __name__ == "__main__":
    import config_preprocessing as cp
    import time
    
    start = time.time()
    # Пример вызова функции
    input_dir = cp.path_tensor_to_visual
    output_dir = cp.path_image
    # Создаем выходную директорию, если ее нет
    os.makedirs(output_dir, exist_ok=True)

    # Получаем список файлов .npy
    filenames = [f for f in os.listdir(input_dir) if f.endswith('.npy')]
    print(filenames)
    for filename in filenames:
        file_path = os.path.join(input_dir, filename)
        data = np.load(file_path)  # Загрузка
        # Вызов функции для визуализации
        visual_tensor(input_dir, filename,
                      cp.feature_input_tensor, 
                      cp.feature_output_tensor, cp.channels_visualisation, output_dir)
    end = time.time()
    print(round((end-start)))