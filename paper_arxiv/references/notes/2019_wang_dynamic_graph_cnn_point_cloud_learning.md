# Source Card: 2019_wang_dynamic_graph_cnn_point_cloud_learning

- Citation key: wang2019dgcnn
- Type: paper
- PDF file: 2019_wang_dynamic_graph_cnn_point_cloud_learning.pdf

## Problem
- Improve point cloud learning by explicitly modeling local neighborhood relations.

## Method
- Dynamic graph CNN (DGCNN) with EdgeConv operation.
- Recomputes k-NN graph in feature space across layers to capture evolving local structure.
- Aggregates edge features for classification/segmentation.

## Data / Dataset
- ModelNet40.
- Part segmentation and scene segmentation benchmarks (reported in paper).

## Metrics
- Classification accuracy.
- Mean IoU for segmentation tasks.

## Main Results
- Strong performance compared to prior point-based models at publication time.
- Demonstrates benefit of dynamic neighborhood modeling.

## Limitations (from source)
- Graph construction/reconstruction increases computation and memory cost.
- Runtime can be a constraint for very large point clouds.

## Why We Cite It
- Key graph-based 3D method after PointNet/PointNet++.
- Useful for framing trade-offs between accuracy and computational complexity.

## Notes / Quotes (optional)
- TODO: add exact page/section for EdgeConv definition and benchmark numbers.
