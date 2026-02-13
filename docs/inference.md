# Inference Guide

## Installation

```bash
pip install -e .
```

## Python API

### Multiclass Classification (20 classes)

```python
from musac_las_classifier import LasClassificationPipeline, PipelineConfig

config = PipelineConfig.for_multiclass(
    checkpoint_path="models/model_epoch_31_multiclass.pth",
    encoder_weights_path="models/resnet34-333f7ec4.pth",
)

pipeline = LasClassificationPipeline(config, workdir="workdir")
pipeline.load_las("path/to/las_directory")
pipeline.run_classification("output/classified.las")
```

### Binary Building Segmentation + Polygon Extraction

```python
from musac_las_classifier import LasClassificationPipeline, PipelineConfig

config = PipelineConfig.for_binary(
    checkpoint_path="models/model_epoch_25_binary.pth",
    encoder_weights_path="models/resnet34-333f7ec4.pth",
)

pipeline = LasClassificationPipeline(config, workdir="workdir")
pipeline.load_las("path/to/las_directory")
pipeline.run_polygon_extraction("output/shapefiles")
```

### Step-by-Step Execution

For more control, run individual pipeline stages:

```python
pipeline.load_las("path/to/las_directory")
pipeline.slice()            # LAS -> 250m tiles
pipeline.encode()           # Tiles -> KNN feature tensors
pipeline.generate_images()  # Tensors -> PNG feature images
pipeline.predict()          # Images -> segmentation masks
pipeline.stitch()           # Masks -> joined mosaic
```

## CLI

### Classify LAS files

```bash
musac-classify path/to/las_dir \
    -o output/classified.las \
    --checkpoint models/model_epoch_31_multiclass.pth \
    --encoder-weights models/resnet34-333f7ec4.pth \
    --device auto \
    -v
```

### Extract building polygons

```bash
musac-polygons path/to/las_dir \
    -o output/shapefiles \
    --checkpoint models/model_epoch_25_binary.pth \
    --encoder-weights models/resnet34-333f7ec4.pth \
    --min-area 500 \
    -v
```

## Configuration

`PipelineConfig` accepts these parameters:

| Parameter | Default | Description |
|---|---|---|
| `tile_size` | 250 | Tile size in meters |
| `k_nn` | 4 | K nearest neighbors |
| `m_tensor_size` | 512 | Grid resolution (MxM) |
| `num_points_lim` | 30000 | Target points per tile |
| `batch_size` | 8 | Inference batch size |
| `device` | "auto" | "auto", "cuda", or "cpu" |
| `num_classes` | 20 | Number of segmentation classes |
| `min_polygon_area` | 500 | Min contour area for polygons |

## Model Weights

Place model files in `models/`:

```
models/
├── resnet34-333f7ec4.pth          # ResNet34 encoder (download: python scripts/download_models.py)
├── model_epoch_25_binary.pth      # Binary building segmentation
└── model_epoch_31_multiclass.pth  # 20-class multiclass segmentation
```

## Output Formats

- **Classified LAS**: Standard LAS file with `classification` field (0-19) and optional RGB per point
- **Shapefiles**: ESRI Shapefile (.shp, .shx, .dbf) with building footprint polygons
