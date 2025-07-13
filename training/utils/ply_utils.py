import os
import numpy as np
from multiprocessing import Pool
import laspy
from plyfile import PlyData

def get_filenames_without_extension(directory):
    # Получаем список всех файлов в указанной директории
    filenames = os.listdir(directory)

    # Удаляем расширение у каждого файла
    filenames_without_extension = [os.path.splitext(filename)[0] for filename in filenames]

    return filenames_without_extension

def main_parallel_ply2las(num_workers = 1, ply_dir='', las_dir=''):
    filenames = get_filenames_without_extension(ply_dir)

    # Создание списка аргументов для передачи каждому процессу
    file_data_list = [(file, ply_dir, las_dir) for file in filenames]

    # Параллельная обработка с использованием пула процессов
    with Pool(processes=num_workers) as pool:
        pool.map(process_file_ply2las_rgb, file_data_list)

def process_file_ply2las_rgb(file_data):
    """
    Обработка одного файла PLY и его сохранение как LAS.
    """
    file, ply_dir, las_dir = file_data
    print(f'Now processing {file}')
    ply_file = os.path.join(ply_dir, f"{file}.ply")
    las_file = os.path.join(las_dir, f"{file}.las")
    ply_to_las_rgb(ply_file, las_file, dataset='stpls3d')

def ply_to_las_rgb(ply_file_path, las_file_path, dataset='toronto3d'):
    """
    Конвертирует PLY файл в LAS файл с координатами, цветом и метками классификации.

    :param ply_file_path: Путь к PLY файлу.
    :param las_file_path: Путь для сохранения выходного LAS файла.
    :param dataset: Название датасета для определения атрибутов.
    """
    # Открываем PLY файл
    ply_data = PlyData.read(ply_file_path)

    # Доступ к элементам vertex
    vertex_data = ply_data['vertex'].data

    # Извлекаем координаты x, y, z
    x = vertex_data['x']
    y = vertex_data['y']
    z = vertex_data['z']

    # Извлекаем метки классификации на основе датасета
    if dataset == 'toronto3d':
        labels = vertex_data['scalar_Label']  # Метки классификации
    elif dataset == 'stpls3d':
        labels = vertex_data['class']

    # Проверяем, присутствуют ли цветовые атрибуты в PLY файле
    if 'red' in vertex_data.dtype.names and 'green' in vertex_data.dtype.names and 'blue' in vertex_data.dtype.names:
        r = vertex_data['red']
        g = vertex_data['green']
        b = vertex_data['blue']
    else:
        raise ValueError("Цветовые атрибуты (red, green, blue) отсутствуют в PLY файле.")

    # Создаем массив с координатами точек (x, y, z)
    points = np.vstack([x, y, z]).T

    # Создание LAS файла с версией 1.2 и форматом точек 3
    las_file = laspy.create(file_version="1.2", point_format=3)

    # Установка координат в LAS файл
    las_file.x = (points[:, 0]-np.min(points[:, 0]))/np.max((points[:, 0]-np.min(points[:, 0])))*499
    las_file.y = (points[:, 1]-np.min(points[:, 1]))/np.max((points[:, 1]-np.min(points[:, 1])))*499
    las_file.z = points[:, 2]

    # Установка меток классификации в LAS файл
    las_file.classification = labels.astype(np.uint8)  # Преобразуем метки в uint8

    # Установка цветовых атрибутов в LAS файл
    las_file.red = r.astype(np.uint16)  # Значения r, g, b должны быть в диапазоне от 0 до 65535
    las_file.green = g.astype(np.uint16)
    las_file.blue = b.astype(np.uint16)

    # Сохранение LAS файла
    las_file.write(las_file_path)

    print(f"Файл {las_file_path} успешно создан.")