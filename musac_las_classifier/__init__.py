"""
musac_las_classifier — End-to-end LiDAR point cloud classification pipeline.

Converts raw LAS files to 2D feature images, runs U-Net segmentation,
and projects predictions back to 3D for GIS-ready outputs.
"""

from musac_las_classifier.config import PipelineConfig
from musac_las_classifier.pipeline import LasClassificationPipeline

__all__ = ["LasClassificationPipeline", "PipelineConfig"]
__version__ = "0.2.0"
