# Draft Paper Outline (arXiv-style)

## Title
- From 3D LiDAR to 2D Segmentation and Back: A Practical Pipeline for Building Footprints and Multiclass Point-Cloud Labeling

## Authors
- Saviour Formosa — PhD Professor (Associate), University of Malta
- Tram Nguyen — PhD student, University of Malta
- Dylan Seychell — PhD, Faculty Member, University of Malta
- Alexey Kozhakin — PhD student, University of Malta
- <span style="background: #fff3b0; padding: 0 4px;"><span style="color: #8b0000;">TODO: confirm author order</span></span>
- <span style="background: #fff3b0; padding: 0 4px;"><span style="color: #8b0000;">TODO: add emails/affiliations formatting</span></span>

## Abstract
- We present a practical end-to-end pipeline for LiDAR processing that converts 3D point clouds into 2D feature maps for segmentation and projects predictions back to 3D for GIS-ready outputs.
- The method uses KNN-based 3D→2D feature encoding and U-Net (ResNet-34) models for both binary building segmentation and 20-class labeling, followed by tile stitching and polygon extraction.
- We train on STPLS3D and demonstrate inference on Malta scenes (St Paul’s Bay, Xewkija, Gozo Rabat, Sliema), producing building footprints and classified LAS files.
- The pipeline achieves <span style="background: #fff3b0; padding: 0 4px;"><span style="color: #8b0000;">METRICS_TBD</span></span> on STPLS3D and yields high-quality qualitative results on Malta, enabling efficient building contour generation and 3D visualization.
- <span style="background: #fff3b0; padding: 0 4px;"><span style="color: #8b0000;">TODO: insert mIoU/IoU/F1 values for STPLS3D (binary + multiclass) and report inference/processing time per tile (and hardware).</span></span>

## 1. Introduction
- Motivation: large LiDAR datasets are heavy to process; manual workflows are slow; GIS-ready outputs are needed.
- Problem statement: automate building extraction and multiclass labeling from raw LAS with minimal manual steps.
- Contributions (draft):
  - End-to-end pipeline from LAS to building polygons and GIS-ready outputs.
  - 3D→2D feature encoding with KNN on a regular grid for efficient 2D segmentation.
  - Dual models: binary building segmentation and 20-class segmentation.
  - 3D back-projection to LAS with class + RGB for visualization and downstream use.
- <span style="background: #fff3b0; padding: 0 4px;"><span style="color: #8b0000;">TODO: refine contribution wording + novelty claim</span></span>

## 2. Related Work
- 3D point cloud segmentation (direct 3D networks, voxel/point approaches).
- 2D projection-based LiDAR processing for efficiency.
- Building footprint extraction and GIS polygonization.
- <span style="background: #fff3b0; padding: 0 4px;"><span style="color: #8b0000;">TODO: write this section as 1–3 short paragraphs (not bullets); compare approaches and position our pipeline.</span></span>
- <span style="background: #fff3b0; padding: 0 4px;"><span style="color: #8b0000;">TODO: add 6–10 sources (papers + datasets).</span></span>
- <span style="background: #fff3b0; padding: 0 4px;"><span style="color: #8b0000;">TODO: add citation to the “3D Data Science” book (PDF) and summarize current approaches from it.</span></span>
- <span style="background: #fff3b0; padding: 0 4px;"><span style="color: #8b0000;">TODO: identify comparison methods and cite them.</span></span>

## 3. Method
### 3.1 Overview
- Pipeline: tiling → feature encoding → 2D segmentation → stitching → polygons → 3D back-projection.
- Two applications:
  - Polygon generation app for multiple LAS files.
  - 3D multiclass segmentation app for a single LAS file.

### 3.2 Data Preprocessing and Tiling
- LAS input (LAS 1.0–1.4) with XYZ, RGB, class if present.
- Tile size: 250m × 250m.
- Point limit: 30,000 points per tile; if fewer, resample with replacement.
- Coordinate normalization: subtract min; scale to [0, 1] for 2D mapping.
- Malta inference naming: filename parsed as `x_km_y_km`, used to place 250m tiles within 1km blocks.
- <span style="background: #fff3b0; padding: 0 4px;"><span style="color: #8b0000;">TODO: exact dataset sizes and split details</span></span>

### 3.3 3D-to-2D Feature Encoding
- Grid size: 512 × 512.
- KNN: K=4 via cKDTree.
- Features:
  - z_mean with 2D smoothing and profile correction.
  - n_z and n_r from local std of XYZ (surface normal proxy).
  - RGB via mean over K neighbors.
  - class channel via neighbor class (optional).
- <span style="background: #fff3b0; padding: 0 4px;"><span style="color: #8b0000;">TODO: justify feature set and smoothing choice</span></span>

### 3.4 2D Segmentation Models
- U-Net with ResNet34 encoder, in_channels=3.
- Binary building segmentation: 2 classes.
- Multiclass segmentation: 20 classes with fixed color map.
- Inference batch size in code: 8.
- Loss: cross-entropy (per report).
- Optimizer: Adam (per report).
- <span style="background: #fff3b0; padding: 0 4px;"><span style="color: #8b0000;">TODO: training hyperparameters (epochs, LR, scheduler), augmentation</span></span>

### 3.5 Post-processing and Polygon Generation
- Tile stitching by coordinates from filename; missing tiles filled with black.
- Binary thresholding on predicted mask.
- Contour detection (OpenCV), external contours.
- Minimum area filtering: min_area=500.
- Export polygons to Shapefile (SHP + DBF + SHX).
- <span style="background: #fff3b0; padding: 0 4px;"><span style="color: #8b0000;">TODO: polygon accuracy evaluation (if any)</span></span>

### 3.6 3D Back-Projection
- Nearest-neighbor interpolation from 2D mask to 3D points (NearestNDInterpolator).
- Writes classification + RGB back to LAS; adds RGB dimensions if missing.

## 4. Experiments
### 4.1 Datasets
- STPLS3D used for training/validation (urban LiDAR).
- Malta inference areas: St Paul’s Bay, Xewkija, Gozo Rabat, Sliema.
- <span style="background: #fff3b0; padding: 0 4px;"><span style="color: #8b0000;">TODO: exact STPLS3D split, number of tiles/scenes, class distribution</span></span>
- <span style="background: #fff3b0; padding: 0 4px;"><span style="color: #8b0000;">TODO: Malta dataset size, coordinate system, availability of ground truth</span></span>

### 4.2 Training Protocol
- Optimizer: Adam (per report).
- Loss: cross-entropy (per report).
- <span style="background: #fff3b0; padding: 0 4px;"><span style="color: #8b0000;">TODO: epochs, batch size, LR, scheduler, augmentations, hardware</span></span>

### 4.3 Metrics
- Metrics computed in training notebook: precision, recall, IoU, F1.
- <span style="background: #fff3b0; padding: 0 4px;"><span style="color: #8b0000;">TODO: report values (overall + per-class)</span></span>

## 5. Results
### 5.1 Quantitative Results
- <span style="background: #fff3b0; padding: 0 4px;"><span style="color: #8b0000;">TODO: table with STPLS3D metrics for binary and multiclass models</span></span>

### 5.2 Qualitative Results
- Feature visualizations: elevation/surface normals/RGB (Gozo Rabat, Xewkija, St Paul’s Bay).
- Building segmentation outputs (same Malta areas).
- Multiclass segmentation outputs (same Malta areas).
- 3D colored/classified LAS (e.g., Sliema).
- <span style="background: #fff3b0; padding: 0 4px;"><span style="color: #8b0000;">TODO: pick final figures and captions for paper</span></span>

## 6. Applications
- GIS building footprints (Shapefile export).
- Urban planning / infrastructure mapping.
- Environmental monitoring via multiclass outputs.
- <span style="background: #fff3b0; padding: 0 4px;"><span style="color: #8b0000;">TODO: discuss additional applications enabled by the method.</span></span>

## 7. Limitations
- 3D→2D projection may lose vertical structure.
- Sensitivity to point density and tiling boundaries.
- Generalization to unseen sensors/regions not fully tested.
- <span style="background: #fff3b0; padding: 0 4px;"><span style="color: #8b0000;">TODO: check “3D Data Science” book for relevant limitations to cite or discuss.</span></span>

## 8. Conclusion and Future Work
- <span style="background: #fff3b0; padding: 0 4px;"><span style="color: #8b0000;">TODO: summary of results and practical impact.</span></span>
- <span style="background: #fff3b0; padding: 0 4px;"><span style="color: #8b0000;">TODO: future work (better 3D supervision, larger datasets, more classes, ablations).</span></span>

## 9. Acknowledgements
- <span style="background: #fff3b0; padding: 0 4px;"><span style="color: #8b0000;">TODO: funding, collaborators, dataset providers (STPLS3D, Malta sources, GlaDOS sumpercomputer)</span></span>

## 10. References
- <span style="background: #fff3b0; padding: 0 4px;"><span style="color: #8b0000;">TODO: bib entries for segmentation, LiDAR datasets, polygon extraction</span></span>


