# Configuration file for preprocessing pipeline

# ===== Slicing - slicing_las_python.py
path_las_before_cut = r"las"
path_las_after_cut = r"las_cut"
las_cut_size = 250
 
# ===== 4. TRANSFORMATION 2D - transformation.py
path_input_las_for_2d = r"las_cut"
path_out_tensors = r"tensors"
K_nn = 4
M_tensor_size = 512
num_points_lim = 30_000
feature_input_tensor = {
    "x": 0,
    "y": 1,
    "z": 2,
    "r": 3,
    "g": 4,
    "b": 5,
    "class": 6,
}
feature_output_tensor = {
                "z_mean": 0,
                "n_z": 1,
                "n_r": 2,
                "r": 3,
                "g": 4,
                "b": 5,
                "class": 6
                }

# ===== 6. Visualization - image_generator.py
path_tensor_to_visual = r"tensors"
path_image_features = r"img_features"
path_image_rgb = r"img_rgb"

channels_visualisation_rgb = {
    "r":0,
    "g":1,
    "b":2,
}

channels_visualisation = {
    "z_mean":0,
    "n_z":1,
    "n_r":2,
}

# ===== visual-classes - tensor_to_image_class.py
path_visual_tensor = r""
path_images = r""

class_to_color = {
    0: [255, 255, 255],
    1: [0, 0 ,0],
    2: [255, 255, 255],
    3: [255, 255, 255],
    4: [255, 255, 255],
    5: [255, 255, 255],
    6: [255, 255, 255],
    7: [255, 255, 255],
    8: [255, 255, 255],
    9: [255, 255, 255],
    10: [255, 255, 255],
    11: [255, 255, 255],
    12: [255, 255, 255],
    13: [255, 255, 255],
    14: [255, 255, 255],
    15: [255, 255, 255],
    16: [255, 255, 255],
    17: [255, 255, 255],
    18: [255, 255, 255],
    19: [255, 255, 255],
}

# ===== prediction preprocessing
class_to_color_multiclass = {
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