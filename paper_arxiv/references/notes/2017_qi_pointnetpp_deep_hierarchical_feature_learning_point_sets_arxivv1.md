# Source Card: 2017_qi_pointnetpp_deep_hierarchical_feature_learning_point_sets_arxivv1

- Citation key: qi2017pointnetpp_arxivv1
- Type: paper
- PDF file: 2017_qi_pointnetpp_deep_hierarchical_feature_learning_point_sets_arxivv1.pdf

## Problem
- Learn hierarchical local-to-global features on unordered point sets and address PointNet local-structure limitations.

## Method
- PointNet++ with set abstraction (sampling, grouping, shared PointNet blocks).
- Multi-scale feature aggregation for non-uniform density robustness.

## Data / Dataset
- Standard benchmarks reported in PointNet++ paper (ModelNet40, ShapeNet, scene benchmarks).

## Metrics
- Classification accuracy and segmentation mIoU (task-dependent).

## Main Results
- Strong improvements over point-based baselines requiring local geometric reasoning.

## Limitations (from source)
- Higher compute cost than PointNet; sensitivity to sampling/grouping choices.

## Why We Cite It
- Canonical hierarchical point-based architecture; key reference in Related Work.

## Notes / Quotes (optional)
- TODO: this is arXiv v1 copy; use one canonical PointNet++ citation in final references.
