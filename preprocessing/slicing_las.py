import os
import subprocess
from multiprocessing import Pool

def process_file_cut_tiles(filename, input_directory, output_directory, tile_size=64):
    """
    Нарезает один LAS-файл на tiles с помощью lastile.

    :param filename: Имя обрабатываемого LAS-файла
    :param input_directory: Директория с исходными файлами
    :param output_directory: Директория для сохранения нарезанных файлов
    :param tile_size: Размер tile (в метрах)
    """
    input_file = os.path.join(input_directory, filename)
    name, _ = os.path.splitext(filename)
    output_subdir = os.path.join(output_directory, name)
    # Создаем подкаталог для текущего файла, если он не существует
    os.makedirs(output_subdir, exist_ok=True)

    # Формируем команду для lastile
    command = [
        'lastile',
        '-i', input_file,        # Входной файл
        '-tile_size', str(tile_size),  # Размер tiles
        '-o', output_subdir       # Директория сохранения
    ]

    # Выполняем команду
    subprocess.run(command)
    print(f"[✔] {filename} успешно нарезан и сохранен в {output_subdir}")
    os.rmdir(output_subdir)

def main_parallel_cut_tiles(input_directory, output_directory, tile_size=64):
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

    num_processes = min(os.cpu_count(), len(filenames))
    with Pool(processes=num_processes) as pool:
        pool.starmap(process_file_cut_tiles, [(filename, input_directory, output_directory, tile_size) for filename in filenames])
                                               

if __name__ == "__main__":
    import config_preprocessing as cp
    import time
    input_directory = cp.path_las_before_cut  # Указать путь к каталогу с LAS-файлами
    output_directory = cp.path_las_after_cut  # Указать путь к каталогу с LAS-файлами
    # Создаем выходную директорию, если она не существует
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    start = time.time()
    # Запуск нарезки в 4 потока
    main_parallel_cut_tiles(input_directory, output_directory, tile_size=cp.las_cut_size)
    end = time.time()
    print(round((end-start)/60,1))