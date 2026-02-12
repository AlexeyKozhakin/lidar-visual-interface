# Source Card: 2017_qi_pointnet_deep_learning_point_sets

- Citation key: qi2017pointnet
- Type: paper
- PDF file: 2017_qi_pointnet_deep_learning_point_sets.pdf

## Problem
- Learn directly from unordered 3D point sets for classification and segmentation without voxelization or handcrafted features.

## Method
- PointNet processes points independently with shared MLPs and aggregates with a symmetric max-pooling function.
- Uses input and feature transform networks (T-Net) for alignment/robustness.
- Produces global features for classification and combines global+local features for segmentation.

## Data / Dataset
- ModelNet40 (classification).
- ShapeNet part segmentation.
- Stanford 3D indoor semantic segmentation benchmark.

## Metrics
- Classification accuracy.
- Mean IoU for segmentation tasks.

## Main Results
- Shows strong performance versus prior 3D methods at publication time.
- Establishes a practical baseline for direct point-based learning.

## Limitations (from source)
- Limited local geometric context because points are first processed independently.
- Neighborhood relations are not explicitly modeled.

## Why We Cite It
- Foundational point-based deep learning architecture for 3D point clouds.
- Baseline reference when discussing later local-context methods.

## Notes / Quotes (optional)
- TODO: add exact page/section for architecture and key quantitative numbers.
