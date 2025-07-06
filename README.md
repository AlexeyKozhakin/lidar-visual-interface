# LiDAR Visual Interface

A comprehensive system for processing, analyzing, and visualizing LiDAR data with machine learning-based building segmentation and polygon generation.

## Overview

This project provides a complete pipeline for LiDAR data processing, from raw LAS files to building segmentation and polygon generation. It includes preprocessing, feature extraction, machine learning prediction, and postprocessing capabilities.

## Features

- **LiDAR Data Processing**: Convert LAS files to tensor format with feature extraction
- **Image Generation**: Create visual representations of LiDAR data (RGB and feature images)
- **Building Segmentation**: Machine learning-based building detection using U-Net architecture
- **Polygon Generation**: Automatic contour detection and Shapefile generation
- **Parallel Processing**: Multi-threaded processing for large datasets
- **Web Interface**: Streamlit-based user interface for easy interaction

## Project Structure

```
lidar-visual-interface/
├── main_poligon.py                 # Main application entry point
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
├── polygon_generator/             # Polygon generation module
│   ├── config_polygon_generator.py
│   └── polygon_generator.py
└── generate_colored_las_3D/       # 3D colored LAS generation
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd lidar-visual-interface
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Install additional system dependencies:
   - **lastile**: For LAS file tiling (LASlib)
   - **GDAL**: For geospatial data processing

## Usage

### Web Interface

Run the main application with Streamlit:

```bash
streamlit run main_poligon.py
```

The interface provides:
- LAS file upload
- Data processing pipeline execution
- Visualization of results
- File management

### Command Line

Individual modules can be run independently:

```bash
# LAS file tiling
python preprocessing/slicing_las.py

# Feature extraction
python preprocessing/transformation_las2npy.py

# Image generation
python preprocessing/image_generator.py

# Building prediction
python predictor_building_segmentation/predict_building_segmentation.py

# Polygon generation
python polygon_generator/polygon_generator.py
```

## Processing Pipeline

### 1. Data Preprocessing
- **LAS Tiling**: Split large LAS files into 250m x 250m tiles
- **Feature Extraction**: Convert point clouds to tensor format with features:
  - Z-mean (elevation)
  - Normal vectors (n_z, n_r)
  - RGB values
  - Classification

### 2. Visualization
- **Feature Images**: Generate grayscale images from extracted features
- **RGB Images**: Create color images from RGB data
- **Image Stitching**: Combine tiles into larger mosaics

### 3. Machine Learning
- **Building Segmentation**: U-Net model with ResNet34 encoder
- **Binary Classification**: Building vs. non-building pixels
- **Model Architecture**: Segmentation Models PyTorch (smp)

### 4. Post-processing
- **Contour Detection**: Find building boundaries
- **Polygon Generation**: Create vector polygons
- **Shapefile Export**: Save results in GIS-compatible format

## Configuration

### Preprocessing Configuration (`preprocessing/config_preprocessing.py`)
- File paths for input/output
- Tensor size and KNN parameters
- Feature mapping configuration
- Visualization channel settings

### Prediction Configuration (`predictor_building_segmentation/config_prediction.py`)
- Model checkpoint paths
- Output directory settings

### Polygon Configuration (`polygon_generator/config_polygon_generator.py`)
- Minimum polygon area
- Contour thickness
- Output format settings

## Data Formats

### Input
- **LAS Files**: LiDAR point cloud data
- **Supported**: LAS 1.0-1.4 formats

### Output
- **Tensors**: NumPy arrays (.npy) with extracted features
- **Images**: PNG files for visualization
- **Shapefiles**: Vector polygons (.shp, .dbf, .shx)

## Performance

- **Parallel Processing**: Multi-threaded operations for large datasets
- **Memory Optimization**: Efficient tensor operations
- **Scalable**: Handles large LiDAR datasets

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

### Geospatial
- `pyshp`: Shapefile handling
- `gdal`: Geospatial data processing

### Web Interface
- `streamlit`: Web application framework

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

[Add your license information here]

## Acknowledgments

- LASlib for LAS file processing
- Segmentation Models PyTorch for ML models
- Streamlit for web interface framework
