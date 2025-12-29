import os
import cv2
import numpy as np
from PIL import Image
import shapefile  # pyshp
import config_postprocessing as cpost


def process_image(input_path, output_image_path, output_shp_path, min_area=100, contour_thickness=3):
    # === ЗАГРУЗКА И ПРЕДОБРАБОТКА ===
    image = Image.open(input_path).convert("L")
    image_np = np.array(image)

    # Бинаризация (инверсия)
    _, binary = cv2.threshold(image_np, 127, 255, cv2.THRESH_BINARY_INV)

    # === НАХОЖДЕНИЕ КОНТУРОВ ===
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # === ФИЛЬТРАЦИЯ ШУМОВ ===
    filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) >= min_area]

    # === СОЗДАНИЕ ИЗОБРАЖЕНИЯ С КОНТУРАМИ ===
    h, w = binary.shape
    output_image = np.ones((h, w, 3), dtype=np.uint8) * 255  # белый фон

    # Заливка полигонов серым
    cv2.drawContours(output_image, filtered_contours, -1, (150, 150, 150), thickness=cv2.FILLED)
    # Обводка полигонов красным
    cv2.drawContours(output_image, filtered_contours, -1, (255, 0, 0), thickness=contour_thickness)

    # === СОХРАНЕНИЕ КАРТИНКИ ===
    Image.fromarray(output_image).save(output_image_path)

    # === СОХРАНЕНИЕ ПОЛИГОНОВ В SHAPEFILE ===
    shp_writer = shapefile.Writer(output_shp_path, shapeType=shapefile.POLYGON)
    shp_writer.field("ID", "N")

    for i, contour in enumerate(filtered_contours):
        points = contour.squeeze()
        if len(points.shape) == 1:  # если всего одна точка
            continue
        points_list = points.tolist()

        # замыкаем полигон
        if points_list[0] != points_list[-1]:
            points_list.append(points_list[0])

        shp_writer.poly([points_list])
        shp_writer.record(i)

    shp_writer.close()


def process_directory(input_dir, output_image_dir, output_shp_dir, min_area=100, contour_thickness=3):
    os.makedirs(output_image_dir, exist_ok=True)
    os.makedirs(output_shp_dir, exist_ok=True)

    for filename in os.listdir(input_dir):
        if filename.lower().endswith((".png", ".jpg", ".jpeg")):
            input_path = os.path.join(input_dir, filename)
            name_wo_ext = os.path.splitext(filename)[0]
            output_image_path = os.path.join(output_image_dir, name_wo_ext + ".png")
            output_shp_path = os.path.join(output_shp_dir, name_wo_ext)

            print(f"Обработка: {filename}")
            process_image(input_path, output_image_path, output_shp_path,
                          min_area=min_area, contour_thickness=contour_thickness)
            print(f" -> Контуры: {output_image_path}")
            print(f" -> Полигоны (SHP): {output_shp_path}.shp")


if __name__ == "__main__":
    # 📁 Путь к папкам из config
    input_dir = cpost.path_img_seg
    output_image_dir = cpost.path_img_counter
    output_shp_dir = cpost.path_polygons_shp

    # ⚙️ Запуск обработки
    process_directory(input_dir, output_image_dir, output_shp_dir,
                      min_area=cpost.min_area, contour_thickness=cpost.contour_thickness)
