import laspy
import numpy as np
from PIL import Image
from scipy.interpolate import NearestNDInterpolator
import os

def mask_to_las_with_class_nn_rgb(las_file_path, image_file_path, output_las_path,
                                  class_colors,
                                  grid_size=500):
    # Step 1. Reading the source LAS file
    las = laspy.read(las_file_path)

    # Extracting X, Y, Z coordinates
    x, y, z = las.x, las.y, las.z

    # Step 2. Reading the mask image
    image = Image.open(image_file_path)
    image = np.array(image)

    # Assuming the image has size (grid_size, grid_size, 3)
    img_height, img_width, _ = image.shape

    # Step 3. Shift X, Y coordinates so they start from 0
    x_min, y_min = np.min(x), np.min(y)
    x_shifted = x - x_min
    y_shifted = y - y_min

    # Scale LAS file coordinates to range from 0 to 1 (normalization)
    x_scaled = x_shifted / np.max(x_shifted)
    y_scaled = y_shifted / np.max(y_shifted)

    # Step 4. Convert image pixel indices to coordinates from 0 to 1
    xi = np.linspace(0, 1, img_width)
    yi = np.linspace(0, 1, img_height)
    xi, yi = np.meshgrid(xi, yi)

    # Convert image grid coordinates and corresponding colors to 1D arrays for interpolation
    xi_flat = xi.ravel()
    yi_flat = yi.ravel()
    colors_flat = image.reshape(-1, 3)  # Convert image colors to flat array

    # Step 5. Create interpolator based on nearest neighbors
    interpolator = NearestNDInterpolator(np.column_stack((xi_flat, yi_flat)), colors_flat)

    # Step 6. Apply interpolation for each point from LAS file
    nearest_colors = interpolator(x_scaled, y_scaled)

    # Convert class color map to more convenient structure for lookup
    color_to_class = {tuple(v): k for k, v in class_colors.items()}

    # Step 8. Determine class and RGB for each point based on nearest color
    classifications = np.zeros(len(nearest_colors), dtype=np.uint8)
    rgb_values = np.zeros((len(nearest_colors), 3), dtype=np.uint16)  # for RGB values

    for i, color in enumerate(nearest_colors):
        # Convert colors to integers for matching
        color = tuple(np.round(color).astype(int))
        classifications[i] = color_to_class.get(color, 0)  # Default class 0 (Unclassified)

        # Write RGB values for current class
        if color in color_to_class:
            rgb = np.array(color) * 256  # Convert colors to 16-bit value for LAS
            rgb_values[i] = rgb.astype(np.uint16)
        else:
            rgb_values[i] = (0, 0, 0)  # If class not found, set black color

    # Step 9. Create new LAS file with required data (x, y, z, classification, rgb)
    new_las = laspy.create(point_format=las.point_format, file_version=las.header.version)

    # Transfer x, y, z, classification
    new_las.x = x
    new_las.y = y
    new_las.z = z
    new_las.classification = classifications

    # Check if source LAS file supports RGB saving
    if 'red' in new_las.point_format.dimension_names:
        new_las.red = rgb_values[:, 0]  # Write red channel
        new_las.green = rgb_values[:, 1]  # Write green channel
        new_las.blue = rgb_values[:, 2]  # Write blue channel
    else:
        # Add RGB channels if they are missing
        new_las.point_format.add_extra_dimension(name='red', dtype=np.uint16)
        new_las.point_format.add_extra_dimension(name='green', dtype=np.uint16)
        new_las.point_format.add_extra_dimension(name='blue', dtype=np.uint16)

        new_las.red = rgb_values[:, 0]
        new_las.green = rgb_values[:, 1]
        new_las.blue = rgb_values[:, 2]

    # Step 10. Save updated LAS file
    new_las.write(output_las_path)

    print(f'File {output_las_path} successfully created with point classes and RGB values.')

# === main ===
if __name__ == "__main__":
    # Example class color dictionary
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

    # Paths
    las_file_path = "temp/las/446_3972.las"
    image_file_path = "temp\img_features_join_multi_class\joined.png"
    output_las_path = "temp/output_file.las"

    # Function call
    mask_to_las_with_class_nn_rgb(
        las_file_path=las_file_path,
        image_file_path=image_file_path,
        output_las_path=output_las_path,
        class_colors=class_colors
    )
