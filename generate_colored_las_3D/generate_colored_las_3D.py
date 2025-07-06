import laspy
import numpy as np
from PIL import Image
from scipy.interpolate import NearestNDInterpolator
import os

def mask_to_las_with_class_nn_rgb(las_file_path, image_file_path, output_las_path,
                                  class_colors,
                                  grid_size=500):
    # Шаг 1. Чтение исходного LAS файла
    las = laspy.read(las_file_path)

    # Извлечение координат X, Y, Z
    x, y, z = las.x, las.y, las.z

    # Шаг 2. Чтение изображения маски
    image = Image.open(image_file_path)
    image = np.array(image)

    # Предположим, что изображение имеет размер (grid_size, grid_size, 3)
    img_height, img_width, _ = image.shape

    # Шаг 3. Сдвигаем координаты X, Y так, чтобы они начинались с 0
    x_min, y_min = np.min(x), np.min(y)
    x_shifted = x - x_min
    y_shifted = y - y_min

    # Масштабируем координаты LAS файла к диапазону от 0 до 1 (нормализация)
    x_scaled = x_shifted / np.max(x_shifted)
    y_scaled = y_shifted / np.max(y_shifted)

    # Шаг 4. Преобразуем индексы пикселей изображения в координаты от 0 до 1
    xi = np.linspace(0, 1, img_width)
    yi = np.linspace(0, 1, img_height)
    xi, yi = np.meshgrid(xi, yi)

    # Преобразуем координаты сетки изображения и соответствующие цвета в 1D массивы для интерполяции
    xi_flat = xi.ravel()
    yi_flat = yi.ravel()
    colors_flat = image.reshape(-1, 3)  # Преобразуем цвета изображения в плоский массив

    # Шаг 5. Создаем интерполятор на основе ближайших соседей
    interpolator = NearestNDInterpolator(np.column_stack((xi_flat, yi_flat)), colors_flat)

    # Шаг 6. Применяем интерполяцию для каждой точки из LAS файла
    nearest_colors = interpolator(x_scaled, y_scaled)

    # Преобразуем карту цветов классов в более удобную для поиска структуру
    color_to_class = {tuple(v): k for k, v in class_colors.items()}

    # Шаг 8. Определяем класс и RGB для каждой точки на основании ближайшего цвета
    classifications = np.zeros(len(nearest_colors), dtype=np.uint8)
    rgb_values = np.zeros((len(nearest_colors), 3), dtype=np.uint16)  # для RGB значений

    for i, color in enumerate(nearest_colors):
        # Приведение цветов к целым числам для сопоставления
        color = tuple(np.round(color).astype(int))
        classifications[i] = color_to_class.get(color, 0)  # Класс по умолчанию 0 (Unclassified)

        # Записываем RGB-значения для текущего класса
        if color in color_to_class:
            rgb = np.array(color) * 256  # Преобразуем цвета в 16-битное значение для LAS
            rgb_values[i] = rgb.astype(np.uint16)
        else:
            rgb_values[i] = (0, 0, 0)  # Если класс не найден, ставим черный цвет

    # Шаг 9. Создание нового LAS файла с нужными данными (x, y, z, classification, rgb)
    new_las = laspy.create(point_format=las.point_format, file_version=las.header.version)

    # Переносим x, y, z, classification
    new_las.x = x
    new_las.y = y
    new_las.z = z
    new_las.classification = classifications

    # Проверим, поддерживает ли исходный файл LAS сохранение RGB
    if 'red' in new_las.point_format.dimension_names:
        new_las.red = rgb_values[:, 0]  # Записываем красный канал
        new_las.green = rgb_values[:, 1]  # Записываем зеленый канал
        new_las.blue = rgb_values[:, 2]  # Записываем синий канал
    else:
        # Добавим RGB каналы, если они отсутствуют
        new_las.point_format.add_extra_dimension(name='red', dtype=np.uint16)
        new_las.point_format.add_extra_dimension(name='green', dtype=np.uint16)
        new_las.point_format.add_extra_dimension(name='blue', dtype=np.uint16)

        new_las.red = rgb_values[:, 0]
        new_las.green = rgb_values[:, 1]
        new_las.blue = rgb_values[:, 2]

    # Шаг 10. Сохранение обновленного LAS файла
    new_las.write(output_las_path)

    print(f'Файл {output_las_path} успешно создан с классами точек и RGB значениями.')

# === main ===
if __name__ == "__main__":
    # Пример словаря цветов классов
    class_colors = {
        0: [0, 0, 0],
        1: [180, 180, 180],
        2: [0, 255, 0],
        3: [255, 255, 0],
        4: [255, 0, 0],
        5: [135, 206, 250],
        6: [135, 206, 251],
        7: [135, 206, 252],
        8: [135, 206, 253],
        9: [135, 206, 254],
        10: [0, 0, 1],
        11: [0, 0, 2],
        12: [0, 0, 3],
        13: [190, 153, 153],
        14: [190, 153, 154],
        15: [0, 0, 4],
        16: [0, 0, 5],
        17: [180, 180, 181],
        18: [0, 0, 6],
        19: [0, 254, 0],
    }

    # Пути
    las_file_path = "temp/las/446_3972.las"
    image_file_path = "temp\img_features_join_multi_class\joined.png"
    output_las_path = "temp/output_file.las"

    # Вызов функции
    mask_to_las_with_class_nn_rgb(
        las_file_path=las_file_path,
        image_file_path=image_file_path,
        output_las_path=output_las_path,
        class_colors=class_colors
    )
