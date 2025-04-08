import os
import numpy as np
from tqdm import tqdm
from PIL import Image

def parse_coordinates(filename):
    """
    Извлекает координаты из имени файла, например: '453000_3974000.png'.
    """
    base = os.path.basename(filename)
    name, _ = os.path.splitext(base)
    parts = name.split('_')
    y, x = map(int, parts[-2:])
    return y, x

def stitch_images(input_folder, output_file):
    """
    Склеивает изображения из папки на основе их нумерации.
    """
    files = [f for f in os.listdir(input_folder) if f.endswith('.png')]
    coordinates = {}

    for file in files:
        x, y = parse_coordinates(file)
        coordinates[(x, y)] = file

    if not coordinates:
        raise ValueError("В папке нет изображений для склейки")

    x_coords = sorted({x for x, y in coordinates.keys()})
    y_coords = sorted({y for x, y in coordinates.keys()})

    if len(x_coords) < 1 or len(y_coords) < 1:
        raise ValueError("Недостаточно координат для определения сетки")

    step_x = x_coords[1] - x_coords[0] if len(x_coords) > 1 else 0
    step_y = y_coords[1] - y_coords[0] if len(y_coords) > 1 else 0

    min_x, max_x = min(x_coords), max(x_coords)
    min_y, max_y = min(y_coords), max(y_coords)

    num_x = ((max_x - min_x) // step_x) + 1 if step_x != 0 else 1
    num_y = ((max_y - min_y) // step_y) + 1 if step_y != 0 else 1

    first_image = Image.open(os.path.join(input_folder, coordinates[(min_x, min_y)]))
    tile_width, tile_height = first_image.size

    final_width = num_x * tile_width
    final_height = num_y * tile_height

    final_image = Image.new('RGB', (final_width, final_height))
    default_tile = Image.new('RGB', (tile_width, tile_height), (0, 0, 0))

    for y in tqdm(range(num_y), desc="Склеивание строк"):
        for x in range(num_x):
            coord_x = min_x + x * step_x
            coord_y = min_y + y * step_y

            start_x = x * tile_width
            start_y = y * tile_height

            if (coord_x, coord_y) in coordinates:
                tile = Image.open(os.path.join(input_folder, coordinates[(coord_x, coord_y)]))
            else:
                tile = default_tile

            final_image.paste(tile, (start_x, start_y))

    output_path = os.path.join(output_file, 'joined.png')
    final_image.save(output_path)
    print(f"Склеенное изображение сохранено как '{output_path}'")

def main_join_img(input_folder, output_file):
    if not os.path.exists(output_file):
        os.makedirs(output_file)
    stitch_images(input_folder, output_file)



if __name__ == "__main__":
    import config_postprocessing as cpost
    input_folder = cpost.path_image_features
    output_file = cpost.path_image_features_join
    main_join_img(input_folder, output_file)
