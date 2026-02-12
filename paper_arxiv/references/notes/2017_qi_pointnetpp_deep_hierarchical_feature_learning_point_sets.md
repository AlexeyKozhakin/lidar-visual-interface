# Source Card: 2017_qi_pointnetpp_deep_hierarchical_feature_learning_point_sets

- Citation key: qi2017pointnetpp
- Type: paper
- PDF file: 2017_qi_pointnetpp_deep_hierarchical_feature_learning_point_sets.pdf

## Problem
- Extend PointNet to better capture local structures and handle non-uniform point sampling.

## Method
- Hierarchical set abstraction: sampling + grouping + local PointNet.
- Multi-scale and multi-resolution grouping strategies for robustness to varying density.
- Learns local-to-global features in a recursive hierarchy.

## Data / Dataset
- ModelNet40.
- ShapeNet part segmentation.
- Indoor scene benchmarks (as reported in paper).

## Metrics
- Classification accuracy.
- Mean IoU for part/semantic segmentation.

## Main Results
- Improves over PointNet on tasks requiring local geometric reasoning.
- Better robustness under non-uniform sampling conditions.

## Limitations (from source)
- Still computationally heavier than PointNet.
- Grouping/sampling design choices can affect speed and performance.

## Why We Cite It
- Canonical hierarchical point-based model.
- Important comparison point for urban-scale 3D segmentation discussion.

## Notes / Quotes (optional)
- TODO: add exact page/section for multi-scale grouping and quantitative gains.
