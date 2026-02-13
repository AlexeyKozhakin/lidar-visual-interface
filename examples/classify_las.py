"""
Minimal example: classify a LAS point cloud using the pipeline.

Usage:
    python examples/classify_las.py path/to/las_dir output/classified.las
"""

import sys
from musac_las_classifier import LasClassificationPipeline, PipelineConfig

# Configure for 20-class multiclass segmentation
config = PipelineConfig.for_multiclass(
    checkpoint_path="models/model_epoch_31_multiclass.pth",
    encoder_weights_path="models/resnet34-333f7ec4.pth",
)

# Run pipeline
pipeline = LasClassificationPipeline(config, workdir="workdir")
pipeline.load_las(sys.argv[1] if len(sys.argv) > 1 else "data/las")
pipeline.run_classification(sys.argv[2] if len(sys.argv) > 2 else "output/classified.las")
