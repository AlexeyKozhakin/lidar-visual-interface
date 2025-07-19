import laspy
import numpy as np
from PIL import Image
from scipy.interpolate import NearestNDInterpolator
import os

def mask_to_las_with_class_only(las_file_path, image_file_path, output_las_path,
                                class_colors, grid_size=500):
    """
    Добавляет только классификацию к LAS файлу, сохраняя оригинальные RGB значения.
    
    Args:
        las_file_path: путь к исходному LAS файлу
        image_file_path: путь к изображению с предсказаниями
        output_las_path: путь для сохранения результата
        class_colors: словарь соответствия цветов классам
        grid_size: размер сетки (по умолчанию 500)
    """
    # Step 1. Чтение исходного LAS файла
    las = laspy.read(las_file_path)

    # Извлечение X, Y, Z координат и преобразование в numpy массивы
    x, y, z = np.array(las.x), np.array(las.y), np.array(las.z)

    # Step 2. Чтение изображения с предсказаниями
    image = Image.open(image_file_path)
    image = np.array(image)

    # Предполагаем, что изображение имеет размер (grid_size, grid_size, 3)
    img_height, img_width, _ = image.shape

    # Step 3. Сдвиг X, Y координат так, чтобы они начинались с 0
    x_min, y_min = np.min(x), np.min(y)
    x_shifted = x - x_min
    y_shifted = y - y_min

    # Масштабирование координат LAS файла в диапазон от 0 до 1 (нормализация)
    x_scaled = x_shifted / np.max(x_shifted)
    y_scaled = y_shifted / np.max(y_shifted)

    # Step 4. Преобразование индексов пикселей изображения в координаты от 0 до 1
    xi = np.linspace(0, 1, img_width)
    yi = np.linspace(0, 1, img_height)
    xi, yi = np.meshgrid(xi, yi)

    # Преобразование координат сетки изображения и соответствующих цветов в 1D массивы для интерполяции
    xi_flat = xi.ravel()
    yi_flat = yi.ravel()
    colors_flat = image.reshape(-1, 3)  # Преобразование цветов изображения в плоский массив

    # Step 5. Создание интерполятора на основе ближайших соседей
    interpolator = NearestNDInterpolator(np.column_stack((xi_flat, yi_flat)), colors_flat)

    # Step 6. Применение интерполяции для каждой точки из LAS файла
    nearest_colors = interpolator(x_scaled, y_scaled)

    # Преобразование карты цветов классов в более удобную структуру для поиска
    color_to_class = {tuple(v): k for k, v in class_colors.items()}

    # Step 7. Определение класса для каждой точки на основе ближайшего цвета
    classifications = np.zeros(len(nearest_colors), dtype=np.uint8)

    for i, color in enumerate(nearest_colors):
        # Преобразование цветов в целые числа для сопоставления
        color = tuple(np.round(color).astype(int))
        classifications[i] = color_to_class.get(color, 0)  # Класс по умолчанию 0 (Не классифицировано)

    # Step 8. Создание нового LAS файла с требуемыми данными (x, y, z, classification)
    new_las = laspy.create(point_format=las.point_format, file_version=str(las.header.version))

    # Копирование всех данных из исходного файла
    for dimension in las.point_format.dimension_names:
        if hasattr(las, dimension):
            setattr(new_las, dimension, getattr(las, dimension))

    # Обновление только классификации
    new_las.classification = classifications
    
    # Масштабирование координат (деление на 10)
    new_las.x = np.array(new_las.x) / 10 + x_min
    new_las.y = np.array(new_las.y) / 10 + y_min
    new_las.z = np.array(new_las.z) / 10

    # Step 9. Сохранение обновленного LAS файла
    new_las.write(output_las_path)

    print(f'Файл {output_las_path} успешно создан с добавленной классификацией точек.')

def create_output_directory():
    """Создает выходной каталог если он не существует."""
    output_dir = "generate_class_las_3D"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Создан каталог: {output_dir}")
    return output_dir

# === main ===
if __name__ == "__main__":
    # Импорт конфигурации
    from config_class_las import CLASS_COLORS, LAS_FILE_PATH, IMAGE_FILE_PATH, OUTPUT_DIRECTORY, OUTPUT_SUFFIX

    LAS_FILE_PATH = r"C:\Users\alexe\VSCprojects\lidar-visual-interface\project_batch_15_07_2025_19_34_40\454_3973\las\454_3973\454_3973.las"
    IMAGE_FILE_PATH = r"C:\Users\alexe\VSCprojects\lidar-visual-interface\project_batch_15_07_2025_19_34_40\454_3973\img_predict_multi_class_join\454_3973\joined.png"

    # Создание выходного каталога
    output_dir = create_output_directory()
    
    # Формирование имени выходного файла
    base_name = os.path.splitext(os.path.basename(LAS_FILE_PATH))[0]
    output_las_path = os.path.join(output_dir, f"{base_name}{OUTPUT_SUFFIX}.las")

    # Вызов функции
    mask_to_las_with_class_only(
        las_file_path=LAS_FILE_PATH,
        image_file_path=IMAGE_FILE_PATH,
        output_las_path=output_las_path,
        class_colors=CLASS_COLORS
    ) 