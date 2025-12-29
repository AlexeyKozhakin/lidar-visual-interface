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


### 3. Simple code to start pipeline
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
    workdir="example_data/workdir",
)

pipeline.load_las("example_data/las")
pipeline.run()
pipeline.export_las("example_data/las_output")

```

### 4. Video turorial

[Whatch video](https://opentopography.org/data)
