# LiDAR Visual Interface

A comprehensive system for processing, analyzing, and visualizing LiDAR data with machine learning-based building segmentation and polygon generation. This repository contains **two main applications**:

1. **Polygon Generation Application** - Processes multiple LAS files to generate building polygons and contours
2. **3D Multi-Class Segmentation Application** - Processes single LAS files for 3D multi-class segmentation and returns classified LAS files

## Overview

This project provides a complete pipeline for LiDAR data processing, from raw LAS files to building segmentation and polygon generation. It includes preprocessing, feature extraction, machine learning prediction, and postprocessing capabilities.

## Features

### Polygon Generation Application
- **LiDAR Data Processing**: Convert LAS files to tensor format with feature extraction
- **Image Generation**: Create visual representations of LiDAR data (RGB and feature images)
- **Building Segmentation**: Machine learning-based building detection using U-Net architecture
- **Polygon Generation**: Automatic contour detection and Shapefile generation
- **Parallel Processing**: Multi-threaded processing for large datasets
- **Web Interface**: Streamlit-based user interface for easy interaction

### 3D Multi-Class Segmentation Application
- **Single LAS Processing**: Process individual LAS files for 3D segmentation
- **Multi-Class Classification**: Support for up to 20 different object classes
- **3D Point Cloud Classification**: Assign classification labels to each 3D point
- **RGB Color Mapping**: Visual representation of classified points with RGB colors
- **LAS Output**: Return classified LAS files with embedded classification data
- **Nearest Neighbor Interpolation**: Accurate mapping of 2D segmentation to 3D points

## Project Structure

```
lidar-visual-interface/
├── main_poligon.py                 # Main polygon generation application
├── main_color_las.py              # Main 3D segmentation application
├── utils.py                        # Utility functions for file management
├── preprocessing/                  # Data preprocessing modules
│   ├── config_preprocessing.py     # Preprocessing configuration
│   ├── slicing_las.py             # LAS file tiling
│   ├── transformation_las2npy.py  # LAS to tensor conversion
│   └── image_generator.py         # Tensor to image conversion
├── postprocessing/                 # Post-processing modules
│   ├── config_postprocessing.py   # Post-processing configuration
│   └── join_img.py                # Image stitching
├── predictor_building_segmentation/ # ML prediction module
│   ├── config_prediction.py       # Prediction configuration
│   ├── predict_building_segmentation.py
│   └── model/                     # Trained models
├── predictor_multiclass_segmentation/ # Multi-class ML prediction module
│   ├── config_prediction.py       # Multi-class prediction configuration
│   ├── predict_multiclass_segmentation.py
│   └── model/                     # Multi-class trained models
├── polygon_generator/             # Polygon generation module
│   ├── config_polygon_generator.py
│   └── polygon_generator.py
└── generate_colored_las_3D/       # 3D colored LAS generation
    ├── config_colored_las.py      # 3D segmentation configuration
    └── generate_colored_las_3D.py # 3D point cloud classification
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/AlexeyKozhakin/lidar-visual-interface.git
cd lidar-visual-interface
```

2. You need to create 3 environments if you need some development and want to run application and work with them from side of development
- for app poligons
```bash
python -m venv venv_polygon_app
```
```bash
.\venv_polygon_app\Scripts\activate
```
```bash
pip install -r requirements_polygon_app.txt
```

- for app 3d_las_colored

2. Install dependencies:
```bash
python -m venv venv_3d_las_colored_app
```
```bash
.\venv_3d_las_colored_app\Scripts\activate
```
```bash
pip install -r requirements_3d_las_colored_app.txt
```

3. Install additional applications:

- **[LasTools](https://lastools.github.io/)**  
  If you have `.laz` files, use LasTools to convert them to `.las` format.

- **[CloudCompare](https://www.danielgm.net/cc/)**  
  Application for 3D point cloud visualization.

- **[Potree](https://github.com/potree/potree)**  
  Alternative tool for web-based 3D point cloud visualization.

- **[SAGA GIS](https://saga-gis.sourceforge.io/en/index.html)**  
  Application for visualizing polygon results and advanced geospatial analysis.

## Usage

### Application 1: Polygon Generation

#### Desktop Applications

You can download the following applications:

- **[LasFilter](https://drive.google.com/file/d/1xY-dF9Nef1U0kk-r9ruKcYNgmSQRbeLe/view?usp=drive_link)**  
  Application for noise cleaning in LAS files and reducing point density to create lighter files.

- **[Segmentation and Polygon Generator](https://drive.google.com/file/d/1tyRdCsoVPXIdrV4xQklq-VKvql8K8_W_/view?usp=sharing)**  
  Application for building segmentation and generating polygons.

- **[3D Dataset Classification App](https://drive.google.com/file/d/1oY6EJGXahKysqOX7wn4frd4vzzXkIZHh/view?usp=drive_link)**  
  Application for 3D classification of datasets.

- **[Download example LAS files from a free dataset STPLS3D](https://drive.google.com/drive/folders/16eT2g6jTkBbqpKfWcr9lT7S2xU6VNCag)**


## Class Definitions and Color Mapping

| Class ID | Class Name | Description | RGB Color | Hex Color |
|----------|------------|-------------|-----------|-----------|
| 0 | Ground | Including grass, paved road, dirt, etc. | `[0, 0, 0]` | `#000000` |
| 1 | Building | Including commercial, residential, educational buildings | `[180, 180, 180]` | `#B4B4B4` |
| 2 | LowVegetation | 0.5m < vegetation height < 2.0m | `[0, 255, 0]` | `#00FF00` |
| 3 | MediumVegetation | 2.0m < vegetation height < 5.0m | `[255, 255, 0]` | `#FFFF00` |
| 4 | HighVegetation | 5.0m < vegetation height | `[255, 0, 0]` | `#FF0000` |
| 5 | Vehicle | Including sedans and hatchback cars | `[135, 206, 250]` | `#87CEFA` |
| 6 | Truck | Including pickup trucks, cement trucks, flat-bed trailers, trailer trucks, etc. | `[135, 206, 251]` | `#87CEFB` |
| 7 | Aircraft | Including helicopters and airplanes | `[135, 206, 252]` | `#87CEFC` |
| 8 | MilitaryVehicle | Including tanks and Humvees | `[135, 206, 253]` | `#87CEFD` |
| 9 | Bike | Bicycles | `[135, 206, 254]` | `#87CEFE` |
| 10 | Motorcycle | Motorcycles | `[0, 0, 1]` | `#000001` |
| 11 | LightPole | Including light poles and traffic lights | `[0, 0, 2]` | `#000002` |
| 12 | StreetSign | Including road signs erected at the side of roads | `[0, 0, 3]` | `#000003` |
| 13 | Clutter | Including city furniture, construction equipment, barricades, and other 3D shapes | `[190, 153, 153]` | `#BE9999` |
| 14 | Fence | Including timber, brick, concrete, metal fences | `[190, 153, 154]` | `#BE999A` |
| 15 | Road | Including asphalt and concrete roads | `[0, 0, 4]` | `#000004` |
| 16 | Unassigned | Not defined in current classification | `[0, 0, 5]` | `#000005` |
| 17 | Windows | Glass windows | `[180, 180, 181]` | `#B4B4B5` |
| 18 | Dirt | Bare earth | `[0, 0, 6]` | `#000006` |
| 19 | Grass | Including grass lawn, wild grass, etc. | `[0, 254, 0]` | `#00FE00` |

### Color Legend

- **Ground (0)**: Black - Base terrain including roads and soil
- **Building (1)**: Light Gray - All types of buildings
- **Low Vegetation (2)**: Bright Green - Short vegetation (0.5-2m)
- **Medium Vegetation (3)**: Yellow - Medium height vegetation (2-5m)
- **High Vegetation (4)**: Red - Tall vegetation (>5m)
- **Vehicles (5-9)**: Light Blue variants - Different vehicle types
- **Infrastructure (11-12)**: Dark Blue variants - Poles and signs
- **Clutter (13)**: Light Brown - Miscellaneous objects
- **Fence (14)**: Brown variant - Various fence types
- **Road (15)**: Dark Blue - Road surfaces
- **Windows (17)**: Light Gray variant - Glass surfaces
- **Dirt (18)**: Dark Blue - Bare earth
- **Grass (19)**: Green variant - Grass areas

### Video tutorials
- **[Video demonstration](https://drive.google.com/file/d/1VoYphqrhmuAaDeUUslMY0tVvkt_gdLVN/view?usp=sharing)** — This video shows how to:
  - Convert LAS files to LAZ format using the LasTools app
  - Clean and optimize LAS files
  - Use app LasFilter
  - Polygon buiding generator app
  - 3D Mulitclass segmentation

- **[Video tutorial: Model Training](https://drive.google.com/file/d/1BFdtrAxS8K_JkaHx_xsvnYrY6lPw27uH/view?usp=sharing)** — This video demonstrates how to:
  - Preprocess data for model training
  - Configure model parameters
  - Launch the training process
  - Monitor training progress and results
  - Save and export the trained model

- **[Jupyter Notebook: Model Training](training/notebook/visual_lidar_code_training_model.ipynb)** — This notebook contains code and examples for:
  - Data preprocessing
  - Model configuration
  - Training workflow
  - Visualization of results

- **[Jupyter Notebook: Stretch LAS to Rectangle](training/notebook/stretch_las_to_rectangle.ipynb)** — This notebook demonstrates how to:
  - Transform and normalize LAS point cloud data
  - Stretch and fit point clouds to a rectangular area
  - Prepare data for further processing or model training




## Processing Pipelines

### Polygon Generation Pipeline

#### 1. Data Preprocessing
- **LAS Tiling**: Split large LAS files into 250m x 250m tiles
- **Feature Extraction**: Convert point clouds to tensor format with features:
  - Z-mean (elevation)
  - Normal vectors (n_z, n_r)
  - RGB values
  - Classification

#### 2. Visualization
- **Feature Images**: Generate grayscale images from extracted features
- **RGB Images**: Create color images from RGB data
- **Image Stitching**: Combine tiles into larger mosaics

#### 3. Machine Learning
- **Building Segmentation**: U-Net model with ResNet34 encoder
- **Binary Classification**: Building vs. non-building pixels
- **Model Architecture**: Segmentation Models PyTorch (smp)

#### 4. Post-processing
- **Contour Detection**: Find building boundaries
- **Polygon Generation**: Create vector polygons
- **Shapefile Export**: Save results in GIS-compatible format

### 3D Multi-Class Segmentation Pipeline

#### 1. Single LAS Processing
- **LAS File Input**: Process individual LAS files
- **Feature Extraction**: Convert to tensor format
- **Image Generation**: Create feature and RGB images

#### 2. Multi-Class Prediction
- **Multi-Class Model**: U-Net with ResNet34 encoder
- **20 Classes**: Support for various object types
- **Segmentation Maps**: Generate classification masks

#### 3. 3D Point Classification
- **Coordinate Mapping**: Map 2D segmentation to 3D coordinates
- **Nearest Neighbor Interpolation**: Accurate point classification
- **RGB Color Assignment**: Assign colors based on class labels

#### 4. LAS Output Generation
- **Classification Labels**: Embed classification data in LAS file
- **RGB Values**: Add color information to each point
- **Enhanced LAS**: Return classified point cloud with visual data

## Configuration

### Polygon Generation Configuration

#### Preprocessing Configuration (`preprocessing/config_preprocessing.py`)
- File paths for input/output
- Tensor size and KNN parameters
- Feature mapping configuration
- Visualization channel settings

#### Prediction Configuration (`predictor_building_segmentation/config_prediction.py`)
- Model checkpoint paths
- Output directory settings

#### Polygon Configuration (`polygon_generator/config_polygon_generator.py`)
- Minimum polygon area
- Contour thickness
- Output format settings

### 3D Segmentation Configuration

#### Multi-Class Prediction Configuration (`predictor_multiclass_segmentation/config_prediction.py`)
- Multi-class model checkpoint paths
- Output directory settings
- Class mapping configuration

#### 3D Classification Configuration (`generate_colored_las_3D/config_colored_las.py`)
- Class color mapping (20 classes)
- Output LAS file paths
- RGB color assignment settings

## Data Formats

### Input
- **LAS Files**: LiDAR point cloud data
- **Supported**: LAS 1.0-1.4 formats

### Output

#### Polygon Generation
- **Tensors**: NumPy arrays (.npy) with extracted features
- **Images**: PNG files for visualization
- **Shapefiles**: Vector polygons (.shp, .dbf, .shx)

#### 3D Segmentation
- **Segmentation Images**: PNG files with class masks
- **Classified LAS**: Enhanced LAS files with classification labels and RGB colors
- **Feature Images**: PNG files for visualization

## Dependencies

### Core Dependencies
- `numpy`: Numerical computing
- `pandas`: Data manipulation
- `laspy`: LAS file reading
- `PIL`: Image processing
- `opencv-python`: Computer vision
- `scipy`: Scientific computing

### Machine Learning
- `torch`: PyTorch framework
- `torchvision`: Computer vision models
- `segmentation-models-pytorch`: Segmentation models




