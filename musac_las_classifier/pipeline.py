"""
High-level pipeline orchestrator.

Provides LasClassificationPipeline — the main entry point for running
end-to-end LAS classification or polygon extraction.

Author: Alexey Kozhakin
Email: alexeykozhakin@gmail.com
"""

import json
import logging
import os
import time
from contextlib import contextmanager
from pathlib import Path

from musac_las_classifier.config import PipelineConfig
from musac_las_classifier.postprocessing.backprojection import backproject_mask_to_las
from musac_las_classifier.postprocessing.polygons import extract_polygons_from_directory
from musac_las_classifier.postprocessing.stitching import stitch_tiles, stitch_tiles_by_basename
from musac_las_classifier.prediction.segmentation import predict_tiles
from musac_las_classifier.preprocessing.encoding import encode_las_to_tensors
from musac_las_classifier.preprocessing.imaging import tensors_to_images
from musac_las_classifier.preprocessing.tiling import tile_las_files

logger = logging.getLogger(__name__)


class LasClassificationPipeline:
    """End-to-end pipeline for LAS point cloud classification.

    Supports two workflows:
    - ``run_classification()``: LAS -> tiles -> features -> multiclass segmentation -> 3D back-projection
    - ``run_polygon_extraction()``: LAS -> tiles -> features -> binary segmentation -> stitching -> polygons

    Example::

        from musac_las_classifier import LasClassificationPipeline, PipelineConfig

        config = PipelineConfig.for_multiclass(
            checkpoint_path="models/model_epoch_31_multiclass.pth",
            encoder_weights_path="models/resnet34-333f7ec4.pth",
        )
        pipeline = LasClassificationPipeline(config, workdir="workdir")
        pipeline.load_las("path/to/las_files")
        pipeline.run_classification("output/classified.las")
    """

    def __init__(self, config: PipelineConfig, workdir: str):
        self.config = config
        self.workdir = Path(workdir)

        # Internal working directories
        self.las_dir = None
        self.las_file = None
        self.las_file_name = None

        self.las_cut_dir = self.workdir / "las_cut"
        self.tensor_dir = self.workdir / "tensors"
        self.img_features_dir = self.workdir / "img_features"
        self.img_pred_dir = self.workdir / "img_predictions"
        self.img_join_dir = self.workdir / "img_joined"

        # Timing
        self._stage_times = {}
        self._tile_count = 0

        self._create_directories()

    def _create_directories(self):
        for d in [
            self.las_cut_dir,
            self.tensor_dir,
            self.img_features_dir,
            self.img_pred_dir,
            self.img_join_dir,
        ]:
            d.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def _timed(self, stage_name):
        """Context manager to time a pipeline stage."""
        start = time.time()
        yield
        elapsed = time.time() - start
        self._stage_times[stage_name] = elapsed
        logger.info("Stage '%s' completed in %.2f seconds", stage_name, elapsed)

    def load_las(self, las_directory: str):
        """Register an input LAS directory.

        Args:
            las_directory: Path to directory containing one or more .las files.
        """
        self.las_dir = Path(las_directory)
        if not self.las_dir.exists():
            raise FileNotFoundError(f"LAS directory not found: {las_directory}")

        filenames = [f for f in os.listdir(self.las_dir) if f.lower().endswith(".las")]
        if not filenames:
            raise FileNotFoundError(f"No .las files found in: {las_directory}")

        self.las_file_name = filenames[0]
        self.las_file = str(self.las_dir / self.las_file_name)
        logger.info("Loaded LAS directory: %s (%d files)", las_directory, len(filenames))

    def _ensure_las_loaded(self):
        if self.las_dir is None:
            raise RuntimeError("Call load_las() before running the pipeline")

    def slice(self):
        """Tile LAS files into spatial blocks."""
        self._ensure_las_loaded()
        with self._timed("1_slice"):
            logger.info("Step 1/5: Tiling LAS files...")
            tile_las_files(
                str(self.las_dir),
                str(self.las_cut_dir),
                tile_size=self.config.tile_size,
                train_mode=self.config.train_mode,
            )
        self._tile_count = len(list(self.las_cut_dir.glob("*.las")))
        logger.info("Produced %d tiles", self._tile_count)

    def encode(self):
        """Encode tiled LAS files into feature tensors."""
        with self._timed("2_encode"):
            logger.info("Step 2/5: Encoding LAS to feature tensors...")
            encode_las_to_tensors(
                str(self.las_cut_dir),
                str(self.tensor_dir),
                feature_input_tensor=self.config.feature_input_tensor,
                feature_output_tensor=self.config.feature_output_tensor,
                num_points_lim=self.config.num_points_lim,
                M=self.config.m_tensor_size,
                K=self.config.k_nn,
            )

    def generate_images(self):
        """Convert feature tensors to PNG images for model input."""
        with self._timed("3_imaging"):
            logger.info("Step 3/5: Generating feature images...")
            tensors_to_images(
                str(self.tensor_dir),
                str(self.img_features_dir),
                feature_output_tensor=self.config.feature_output_tensor,
                channels_visualization=self.config.channels_visualization,
            )

    def predict(self):
        """Run segmentation model on feature images."""
        if self.config.checkpoint_path is None:
            raise RuntimeError("checkpoint_path not set in config")
        if self.config.encoder_weights_path is None:
            raise RuntimeError("encoder_weights_path not set in config")

        with self._timed("4_predict"):
            logger.info("Step 4/5: Running segmentation inference...")
            predict_tiles(
                str(self.img_features_dir),
                str(self.img_pred_dir),
                checkpoint_path=self.config.checkpoint_path,
                encoder_weights_path=self.config.encoder_weights_path,
                num_classes=self.config.num_classes,
                class_colors=self.config.class_colors,
                batch_size=self.config.batch_size,
                device=self.config.resolve_device(),
            )

    def stitch(self):
        """Stitch predicted tiles into a full scene mosaic."""
        with self._timed("5_stitch"):
            logger.info("Step 5/5: Stitching prediction tiles...")
            stitch_tiles(
                str(self.img_pred_dir),
                str(self.img_join_dir),
            )

    # ---- Timing report ----

    def get_timing_report(self) -> dict:
        """Get timing report for all pipeline stages.

        Returns:
            Dict with per-stage times, total time, tile count, and per-tile averages.
        """
        total_time = sum(self._stage_times.values())
        tile_count = max(self._tile_count, 1)

        return {
            "stage_times_sec": dict(self._stage_times),
            "total_time_sec": round(total_time, 2),
            "tile_count": self._tile_count,
            "per_tile_total_sec": round(total_time / tile_count, 2),
            "per_tile_inference_sec": round(
                self._stage_times.get("4_predict", 0) / tile_count, 2
            ),
        }

    def save_timing_report(self, output_path: str = None):
        """Save timing report as JSON and log to console.

        Args:
            output_path: Path to save JSON report. Defaults to workdir/timing_report.json.
        """
        report = self.get_timing_report()

        if output_path is None:
            output_path = str(self.workdir / "timing_report.json")

        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)

        logger.info("=== Pipeline Timing Report ===")
        for stage, t in report["stage_times_sec"].items():
            logger.info("  %-20s: %8.2f s", stage, t)
        logger.info("  %-20s: %8.2f s", "TOTAL", report["total_time_sec"])
        logger.info("  Tiles processed: %d", report["tile_count"])
        logger.info("  Per-tile total: %.2f s", report["per_tile_total_sec"])
        logger.info("  Per-tile inference: %.2f s", report["per_tile_inference_sec"])
        logger.info("Report saved to %s", output_path)

        return report

    # ---- High-level workflows ----

    def run_classification(self, output_las_path: str,
                           joined_image_name: str = "joined.png"):
        """Run full classification pipeline and export a classified LAS file.

        Args:
            output_las_path: Path for the output classified LAS file.
            joined_image_name: Name of the stitched prediction image.
        """
        self._ensure_las_loaded()
        self.slice()
        self.encode()
        self.generate_images()
        self.predict()
        self.stitch()

        joined_image_path = self.img_join_dir / joined_image_name
        if not joined_image_path.exists():
            raise FileNotFoundError(
                f"Joined prediction image not found: {joined_image_path}"
            )

        with self._timed("6_backproject"):
            os.makedirs(os.path.dirname(output_las_path) or ".", exist_ok=True)
            logger.info("Exporting classified LAS: %s", output_las_path)
            backproject_mask_to_las(
                las_file_path=self.las_file,
                image_file_path=str(joined_image_path),
                output_las_path=output_las_path,
                class_colors=self.config.class_colors,
            )

        self.save_timing_report()

    def run_polygon_extraction(self, output_shp_dir: str):
        """Run polygon extraction pipeline for building footprints.

        Uses binary segmentation to detect buildings and exports Shapefiles.

        Args:
            output_shp_dir: Directory to save Shapefile outputs.
        """
        self._ensure_las_loaded()
        self.slice()
        self.encode()
        self.generate_images()
        self.predict()

        with self._timed("5_stitch"):
            stitch_tiles_by_basename(
                str(self.img_pred_dir),
                str(self.img_join_dir),
            )

        with self._timed("6_polygons"):
            output_image_dir = str(self.workdir / "contour_images")
            logger.info("Extracting polygons to %s", output_shp_dir)
            extract_polygons_from_directory(
                str(self.img_join_dir),
                output_image_dir,
                output_shp_dir,
                min_area=self.config.min_polygon_area,
                contour_thickness=self.config.contour_thickness,
            )

        self.save_timing_report()

    def run(self):
        """Run preprocessing + prediction + stitching (without export).

        Useful when you want to inspect intermediate results before exporting.
        """
        self._ensure_las_loaded()
        self.slice()
        self.encode()
        self.generate_images()
        self.predict()
        self.stitch()
        self.save_timing_report()
