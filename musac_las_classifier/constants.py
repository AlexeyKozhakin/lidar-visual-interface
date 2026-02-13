"""
Single source of truth for all constants used across the pipeline.

Class color mappings, feature tensor definitions, class names, and default parameters.
"""

# STPLS3D 20-class semantic taxonomy
CLASS_NAMES = {
    0: "Ground",
    1: "Building",
    2: "Low Vegetation",
    3: "Medium Vegetation",
    4: "High Vegetation",
    5: "Vehicle",
    6: "Truck",
    7: "Aircraft",
    8: "Military Vehicle",
    9: "Bike",
    10: "Motorcycle",
    11: "Light Pole",
    12: "Street Sign",
    13: "Clutter",
    14: "Fence",
    15: "Road",
    16: "Sidewalk",
    17: "Parking Area",
    18: "Rail",
    19: "Grass",
}

# RGB color for each class (used in prediction masks and 3D back-projection)
CLASS_COLORS = {
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

# Binary building class colors (building=1 vs background=0)
CLASS_COLORS_BINARY = {
    0: [255, 255, 255],
    1: [0, 0, 0],
}

# Reverse mapping: RGB tuple -> class index
COLOR_TO_CLASS = {tuple(v): k for k, v in CLASS_COLORS.items()}
COLOR_TO_CLASS_BINARY = {tuple(v): k for k, v in CLASS_COLORS_BINARY.items()}

# Input tensor channel mapping (raw LAS data → numpy array columns)
FEATURE_INPUT_TENSOR = {
    "x": 0,
    "y": 1,
    "z": 2,
    "r": 3,
    "g": 4,
    "b": 5,
    "class": 6,
}

# Output tensor channel mapping (computed features → tensor channels)
FEATURE_OUTPUT_TENSOR = {
    "z_mean": 0,
    "n_z": 1,
    "n_r": 2,
    "r": 3,
    "g": 4,
    "b": 5,
    "class": 6,
}

# Channels used for model input visualization (z_mean, n_z, n_r → RGB)
CHANNELS_VISUALIZATION = {
    "z_mean": 0,
    "n_z": 1,
    "n_r": 2,
}

# Channels used for RGB visualization
CHANNELS_VISUALIZATION_RGB = {
    "r": 0,
    "g": 1,
    "b": 2,
}

# Default pipeline parameters
DEFAULT_TILE_SIZE = 250        # meters
DEFAULT_K_NN = 4               # KNN neighbors
DEFAULT_M_TENSOR_SIZE = 512    # grid resolution
DEFAULT_NUM_POINTS_LIM = 30000 # points per tile
DEFAULT_BATCH_SIZE = 8         # inference batch size
DEFAULT_MIN_POLYGON_AREA = 500 # minimum contour area in pixels
DEFAULT_CONTOUR_THICKNESS = 3  # contour line thickness

NUM_CLASSES_MULTICLASS = len(CLASS_COLORS)
NUM_CLASSES_BINARY = 2
