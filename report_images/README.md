# Report Images Directory

This directory contains images for the GENERAL_REPORT.md documentation.

## Directory Structure

### `/training/`
- **training_curves.png** - Training and validation loss curves
- **model_architecture.png** - U-Net architecture visualization
- **training_metrics.png** - Accuracy, precision, recall graphs

### `/features/`
- **feature_visualization.png** - Extracted features visualization
- **elevation_features.png** - Z-mean, Z-std features
- **surface_normal_features.png** - n_z, n_r features
- **distance_features.png** - KNN distance features

### `/rgb/`
- **rgb_visualization.png** - RGB color representation
- **color_channels.png** - Individual RGB channel visualization
- **normalized_rgb.png** - Normalized color channels

### `/predictions/`
- **building_segmentation_results.png** - Building detection results
- **multiclass_segmentation_results.png** - Multi-class segmentation
- **prediction_comparison.png** - Ground truth vs prediction
- **confidence_maps.png** - Model confidence visualization

### `/postprocessing/`
- **stitched_prediction.png** - Combined tile results
- **polygon_generation.png** - Contour detection results
- **shapefile_export.png** - GIS output visualization

### `/ui/`
- **polygon_generation_app.png** - Main application interface
- **3d_segmentation_app.png** - 3D segmentation interface
- **processing_progress.png** - Real-time progress display
- **result_preview.png** - Result browsing interface

### `/3d/`
- **colored_3d_points.png** - 3D point cloud with colors
- **3d_classification_visualization.png** - Classified 3D points
- **point_cloud_comparison.png** - Before/after classification

## Image Requirements

- **Format**: PNG or JPG
- **Resolution**: 800-1200px width recommended
- **Quality**: High resolution for clarity
- **File Size**: Optimize for web viewing (< 2MB per image)

## Usage

Images are referenced in GENERAL_REPORT.md using relative paths:
```markdown
![Description](report_images/category/filename.png)
```

## Naming Convention

- Use descriptive names with underscores
- Include category prefix if needed
- Example: `building_segmentation_results.png` 