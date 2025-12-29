import os
import cv2
import numpy as np
from PIL import Image
import shapefile  # pyshp
import polygon_generator.config_polygon_generator as cpg


def process_image(input_path, output_image_path, output_shp_path, min_area=100, contour_thickness=3):
    # === LOADING AND PREPROCESSING ===
    image = Image.open(input_path).convert("L")
    image_np = np.array(image)

    # Binarization (inversion)
    _, binary = cv2.threshold(image_np, 127, 255, cv2.THRESH_BINARY_INV)

    # === FINDING CONTOURS ===
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # === NOISE FILTERING ===
    filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) >= min_area]

    # === CREATING IMAGE WITH CONTOURS ===
    h, w = binary.shape
    output_image = np.ones((h, w, 3), dtype=np.uint8) * 255  # white background

    # Fill polygons with gray
    cv2.drawContours(output_image, filtered_contours, -1, (150, 150, 150), thickness=cv2.FILLED)
    # Outline polygons with red
    cv2.drawContours(output_image, filtered_contours, -1, (255, 0, 0), thickness=contour_thickness)

    # === SAVING IMAGE ===
    Image.fromarray(output_image).save(output_image_path)

    # === SAVING POLYGONS TO SHAPEFILE ===
    shp_writer = shapefile.Writer(output_shp_path, shapeType=shapefile.POLYGON)
    shp_writer.field("ID", "N")

    # It is assumed that you have a height variable — image height
    # For example: height, width = image.shape[:2]

    for i, contour in enumerate(filtered_contours):
        points = contour.squeeze()
        if len(points.shape) == 1:  # if only one point
            continue
        points_list = points.tolist()
        height = h
        # Reflect coordinates along Y axis (flip vertically)
        flipped_points = [(x, height - y) for (x, y) in points_list]

        # Close polygon
        if flipped_points[0] != flipped_points[-1]:
            flipped_points.append(flipped_points[0])

        shp_writer.poly([flipped_points])
        shp_writer.record(i)

    shp_writer.close()


def main_polygon_generator(input_dir, output_image_dir, output_shp_dir, min_area=100, contour_thickness=3):
    os.makedirs(output_image_dir, exist_ok=True)
    os.makedirs(output_shp_dir, exist_ok=True)

    for filename in os.listdir(input_dir):
        if filename.lower().endswith((".png", ".jpg", ".jpeg")):
            input_path = os.path.join(input_dir, filename)
            name_wo_ext = os.path.splitext(filename)[0]
            output_image_path = os.path.join(output_image_dir, name_wo_ext + ".png")
            output_shp_path = os.path.join(output_shp_dir, name_wo_ext)

            print(f"Processing: {filename}")
            process_image(input_path, output_image_path, output_shp_path,
                          min_area=min_area, contour_thickness=contour_thickness)
            print(f" -> Contours: {output_image_path}")
            print(f" -> Polygons (SHP): {output_shp_path}.shp")
            

if __name__ == "__main__":
    # Path to folders from config
    input_dir = cpg.path_image_prediction
    output_image_dir = cpg.path_image_contours
    output_shp_dir = cpg.path_polygons_shp

    # Run processing
    main_polygon_generator(input_dir, output_image_dir, output_shp_dir,
                      min_area=cpg.min_area, contour_thickness=cpg.contour_thickness)
