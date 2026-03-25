# musac_las_classifier

A Python library for automatic classification of LAS point cloud files using a 3D-to-2D deep learning pipeline.

---

## Quick Start

### 1. Installation

Install directly from GitHub:

```bash
pip install git+https://github.com/AlexeyKozhakin/lidar-visual-interface.git@feature/model-pip-package
```
### 2. Download Sample Data


[Download example LAS data to get started](https://drive.google.com/drive/folders/1P4Ay2Q3a5al7yzbE41eqfVA_PrvPTZrb?usp=drive_link)


### 3. Multi-class segmentation: recommended package API

```python
from musac_las_classifier import (
    LasClassificationPipeline,
    LasPipelineConfig
)

from musac_las_classifier import config_preprocessing as cp
from musac_las_classifier import config_prediction as cpred
from musac_las_classifier import config_colored_las as ccl

config = LasPipelineConfig.for_multiclass(
    tile_size=250,
    feature_output_tensor=cp.feature_output_tensor,
    channels_visualisation=cp.channels_visualisation,
    checkpoint_path=cpred.checkpoint_path,
    encoder_weights_path=cpred.encoder_weights_path,
    class_colors=ccl.class_colors,
)

pipeline = LasClassificationPipeline(
    config=config,
    workdir="example_data/workdir",
)

pipeline.load_las("example_data/las")
pipeline.run()
pipeline.export_las("example_data/las_output")
```

### 4. Multi-class segmentation: legacy constructor still supported

The legacy constructor style still works:

```python
from musac_las_classifier import (
    LasClassificationPipeline,
    LasPipelineConfig
)

from musac_las_classifier import config_preprocessing as cp
from musac_las_classifier import config_prediction as cpred
from musac_las_classifier import config_colored_las as ccl

config = LasPipelineConfig(
    tile_size=250,
    feature_output_tensor=cp.feature_output_tensor,
    channels_visualisation=cp.channels_visualisation,
    checkpoint_path=cpred.checkpoint_path,
    encoder_weights_path=cpred.encoder_weights_path,
    class_colors=ccl.class_colors,
)

pipeline = LasClassificationPipeline(
    config=config,
    workdir="example_data/workdir_multiclass_legacy",
)

pipeline.load_las("example_data/las")
pipeline.run()
pipeline.export_las("example_data/las_output")
```

### 5. Polygon extraction workflow

```python
from musac_las_classifier import (
    LasClassificationPipeline,
    LasPipelineConfig
)

from musac_las_classifier import config_preprocessing as cp

config = LasPipelineConfig.for_polygon_extraction(
    tile_size=250,
    feature_output_tensor=cp.feature_output_tensor,
    channels_visualisation=cp.channels_visualisation,
    min_polygon_area=500,
    contour_thickness=3,
)

pipeline = LasClassificationPipeline(
    config=config,
    workdir="example_data/workdir_polygon",
)

pipeline.load_las("example_data/las")
pipeline.run_polygon_extraction()
```

Polygon outputs are written into the pipeline workdir:

- `img_predict_building/`
- `img_predict_building_join/`
- `img_contours/`
- `polygons_shp/`

### 6. Video turorial

[Whatch video](https://drive.google.com/file/d/1oi5J1lDz-6PCoYGv-mn5n2uQ_4vNptqj/view)
