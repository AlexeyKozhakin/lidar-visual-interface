# Source Card: 2019_milioto_rangenetpp_fast_accurate_lidar_semantic_segmentation

- Citation key: milioto2019rangenetpp
- Type: paper
- PDF file: 2019_milioto_rangenetpp_fast_accurate_lidar_semantic_segmentation.pdf

## Problem
- Need fast and accurate LiDAR semantic segmentation for real-time autonomous systems.

## Method
- RangeNet++: projects 3D LiDAR to range image and performs 2D CNN segmentation.
- Post-processing maps predictions back to 3D points.

## Data / Dataset
- Common LiDAR semantic segmentation benchmarks (e.g., autonomous driving datasets).

## Metrics
- IoU/mIoU and runtime-oriented measures.

## Main Results
- Strong speed-accuracy tradeoff, demonstrating practical projection-based segmentation.

## Limitations (from source)
- Projection can lose 3D detail; performance depends on sensor geometry and post-processing.

## Why We Cite It
- Directly relevant to our 3D->2D->3D pipeline design rationale.

## Notes / Quotes (optional)
- TODO: extract exact benchmark numbers and runtime figures.
