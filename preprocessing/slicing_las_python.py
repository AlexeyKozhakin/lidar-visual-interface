import laspy
import os
from multiprocessing import Pool


def split_las(input_path, output_dir, tile_size=250):
    os.makedirs(output_dir, exist_ok=True)
    # Читаем исходный las файл
    las = laspy.read(input_path)

    # Берем имя файла без расширения
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    
    # Парсим начальные координаты X и Y из имени файла
    x0_km, y0_km = map(int, base_name.split("_"))
    x0 = x0_km * 1000
    y0 = y0_km * 1000

    # Получаем диапазоны
    x_min = x0
    x_max = x0 + 1000
    y_min = y0
    y_max = y0 + 1000

    # Получаем координаты всех точек
    xs = las.x
    ys = las.y

    # Перебираем все 250м квадраты
    for i in range(0, 1000, tile_size):
        for j in range(0, 1000, tile_size):
            tile_x_min = x_min + i
            tile_x_max = tile_x_min + tile_size
            tile_y_min = y_min + j
            tile_y_max = tile_y_min + tile_size

            # Фильтруем точки, попадающие в этот тайл
            mask = (xs >= tile_x_min) & (xs < tile_x_max) & (ys >= tile_y_min) & (ys < tile_y_max)
            selected_points = las.points[mask]

            if len(selected_points) > 0:
                # Создаем новый las объект
                new_las = laspy.LasData(las.header)
                new_las.points = selected_points

                # Формируем имя нового файла
                output_name = f"{x0_km}_{y0_km}_{tile_x_min}_{tile_y_min}.las"
                output_path = os.path.join(output_dir, output_name)

                # Сохраняем
                new_las.write(output_path)

                print(f"Saved {output_path}")

def process_file_cut_tiles(filename, input_directory, output_directory, tile_size=250):
    """
    Нарезает один LAS-файл на tiles с помощью lastile.

    :param filename: Имя обрабатываемого LAS-файла
    :param input_directory: Директория с исходными файлами
    :param output_directory: Директория для сохранения нарезанных файлов
    :param tile_size: Размер tile (в метрах)
    """
    input_file = os.path.join(input_directory, filename)
    #name, _ = os.path.splitext(filename)
    #output_subdir = os.path.join(output_directory, name)
    # Создаем подкаталог для текущего файла, если он не существует
    split_las(input_file, output_directory, tile_size=tile_size)


def main_parallel_cut_tiles(input_directory, output_directory, tile_size=250):
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
    print(filenames)


    num_processes = min(os.cpu_count(), len(filenames))
    print('num_processes',num_processes)
    print('cpu = ',os.cpu_count())
    with Pool(processes=num_processes) as pool:
        pool.starmap(process_file_cut_tiles, [(filename, input_directory, output_directory, tile_size) for filename in filenames])


def main_not_parallel_cut_tiles(input_directory, output_directory, tile_size=250):
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
    print(filenames)


    for filename in filenames:
        process_file_cut_tiles(filename, input_directory, output_directory, tile_size=tile_size)


if __name__ == '__main__':
    # Пример использования
    input_directory = r'temp\las'
    output_directory = r'temp\las_cut'
    main_parallel_cut_tiles(input_directory, output_directory, tile_size=500)