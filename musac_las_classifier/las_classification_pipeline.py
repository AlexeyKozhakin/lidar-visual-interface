"""
LAS Classification Pipeline

End-to-end pipeline for:
- slicing LAS files
- feature generation
- multiclass segmentation prediction
- postprocessing
- exporting classified LAS files

Author: Alexey Kozhakin
email: alexeykozhakin@gmail.com
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional
import os

# =========================
# Imports of your pipeline
# =========================

from musac_las_classifier.preprocessing.slicing_las_python import (
    main_not_parallel_cut_tiles
)

from musac_las_classifier.preprocessing.transformation_las2npy import (
    main_not_parallel_transform_to_tensor
)

from musac_las_classifier.preprocessing.image_generator import (
    main_not_parallel_tensor_to_image
)

from musac_las_classifier.predictor_multiclass_segmentation.predict_multiclass_segmentation import (
    main_prediction
)

from musac_las_classifier.predictor_building_segmentation.predict_building_segmentation import (
    DEFAULT_CHECKPOINT_PATH as DEFAULT_BUILDING_CHECKPOINT_PATH,
    DEFAULT_ENCODER_WEIGHTS_PATH as DEFAULT_BUILDING_ENCODER_WEIGHTS_PATH,
    main_prediction as main_building_prediction
)

from musac_las_classifier.postprocessing.join_img import (
    main_join_img
)

from musac_las_classifier.polygon_generator.polygon_generator import (
    main_polygon_generator
)

from musac_las_classifier.generate_colored_las_3D.generate_class_las_3D import (
    mask_to_las_with_class_only
)

# =========================
# Configuration
# =========================

@dataclass
class LasPipelineConfig:
    """Configuration for LAS classification pipeline"""

    # preprocessing
    # =====================
    # Preprocessing
    # =====================

    tile_size: int = 250
    K_nn: int = 4
    M_tensor_size: int = 512
    num_points_lim: int = 30_000

    feature_input_tensor: Dict[str, int] = field(
        default_factory=lambda: {
            "x": 0,
            "y": 1,
            "z": 2,
            "r": 3,
            "g": 4,
            "b": 5,
            "class": 6,
        }
    )

    feature_output_tensor: Dict[str, int] = field(
        default_factory=lambda: {
            "z_mean": 0,
            "n_z": 1,
            "n_r": 2,
            "r": 3,
            "g": 4,
            "b": 5,
            "class": 6,
        }
    )



    channels_visualisation: Optional[list[int]] = None

    # =====================
    # Model
    # =====================

    checkpoint_path: Optional[str] = None
    encoder_weights_path: Optional[str] = None
    building_checkpoint_path: Optional[str] = str(DEFAULT_BUILDING_CHECKPOINT_PATH)
    building_encoder_weights_path: Optional[str] = str(DEFAULT_BUILDING_ENCODER_WEIGHTS_PATH)
    min_polygon_area: int = 100
    contour_thickness: int = 3

    # =====================
    # Output
    # =====================

    class_colors: Optional[dict] = None

    @classmethod
    def for_multiclass(cls, **kwargs):
        return cls(**kwargs)

    @classmethod
    def for_polygon_extraction(cls, **kwargs):
        return cls(**kwargs)


# =========================
# Pipeline Class
# =========================

class LasClassificationPipeline:
    """
    High-level orchestrator for LAS classification pipeline.
    """

    def __init__(self, config: LasPipelineConfig, workdir: str):
        self.config = config
        self.workdir = Path(workdir)

        # internal paths
        self.las_dir: Path | None = None

        self.las_cut_dir = self.workdir / "las_cut"
        self.tensor_dir = self.workdir / "tensor"
        self.img_features_dir = self.workdir / "img_features"
        self.img_pred_dir = self.workdir / "img_predict_multiclass"
        self.img_join_dir = self.workdir / "img_predict_multiclass_join"
        self.img_pred_building_dir = self.workdir / "img_predict_building"
        self.img_join_building_dir = self.workdir / "img_predict_building_join"
        self.img_contours_dir = self.workdir / "img_contours"
        self.polygons_shp_dir = self.workdir / "polygons_shp"

        self._create_directories()

    # ---------------------
    # Internal utils
    # ---------------------

    def _create_directories(self):
        """Create working directories"""
        for d in [
            self.las_cut_dir,
            self.tensor_dir,
            self.img_features_dir,
            self.img_pred_dir,
            self.img_join_dir,
            self.img_pred_building_dir,
            self.img_join_building_dir,
            self.img_contours_dir,
            self.polygons_shp_dir,
        ]:
            d.mkdir(parents=True, exist_ok=True)

    # ---------------------
    # Public API
    # ---------------------

    def load_las(self, las_directory: str):
        """Register input LAS directory"""
        self.las_dir = Path(las_directory)
        filenames = [f for f in os.listdir(self.las_dir) if f.endswith('.las')]
        self.las_file_name = filenames[0]
        self.las_file = os.path.join(self.las_dir,filenames[0])

        if not self.las_dir.exists():
            raise FileNotFoundError(f"LAS directory not found: {las_directory}")

    def slice_las(self):
        """Cut LAS files into tiles"""
        if self.las_dir is None:
            raise RuntimeError("LAS directory not loaded")

        main_not_parallel_cut_tiles(
            str(self.las_dir),
            str(self.las_cut_dir),
            tile_size=self.config.tile_size,
        )

    def transform_to_tensor(self):
        main_not_parallel_transform_to_tensor(
    str(self.las_cut_dir), 
    str(self.tensor_dir),
    self.config.feature_input_tensor,
    self.config.feature_output_tensor,
    self.config.num_points_lim, 
    self.config.M_tensor_size,
    self.config.K_nn
)

    def prepare_features(self):
        """Generate feature images from tensors"""
        main_not_parallel_tensor_to_image(
            str(self.tensor_dir),
            str(self.img_features_dir),
            self.config.feature_output_tensor,
            self.config.channels_visualisation,
        )

    def predict(self):
        """Run multiclass segmentation model"""
        main_prediction(
            str(self.img_features_dir),
            str(self.img_pred_dir),
            self.config.checkpoint_path,
            self.config.encoder_weights_path
        )

    def postprocess(self):
        """Join predicted image tiles"""
        main_join_img(
            str(self.img_pred_dir),
            str(self.img_join_dir),
        )

    def predict_buildings(self):
        """Run binary building segmentation model"""
        main_building_prediction(
            str(self.img_features_dir),
            str(self.img_pred_building_dir),
            self.config.building_checkpoint_path,
            encoder_weights_path=self.config.building_encoder_weights_path,
        )

    def postprocess_buildings(self):
        """Join predicted building image tiles"""
        main_join_img(
            str(self.img_pred_building_dir),
            str(self.img_join_building_dir),
        )

    def export_polygons(
        self,
        output_shp_dir: Optional[str] = None,
        output_image_dir: Optional[str] = None,
    ):
        """Generate polygons from joined building prediction images"""
        shp_dir = Path(output_shp_dir) if output_shp_dir else self.polygons_shp_dir
        image_dir = Path(output_image_dir) if output_image_dir else self.img_contours_dir

        main_polygon_generator(
            str(self.img_join_building_dir),
            str(image_dir),
            str(shp_dir),
            min_area=self.config.min_polygon_area,
            contour_thickness=self.config.contour_thickness,
        )

    def export_las(
        self,
        output_las_path: str,
        joined_image_name: str = "joined.png",
    ):
        """Export classified LAS file"""

        joined_image_path = self.img_join_dir / joined_image_name
        output_las_file_path = os.path.join(output_las_path, self.las_file_name)

        if not joined_image_path.exists():
            raise FileNotFoundError(
                f"Joined prediction image not found: {joined_image_path}"
            )
        os.makedirs(output_las_path, exist_ok=True)
        mask_to_las_with_class_only(
            las_file_path=self.las_file,
            image_file_path=str(joined_image_path),
            output_las_path=output_las_file_path,
            class_colors=self.config.class_colors,
        )

    def run(self):
        """Run full pipeline"""
        self.slice_las()
        self.transform_to_tensor()
        self.prepare_features()
        self.predict()
        self.postprocess()

    def run_polygon_extraction(
        self,
        output_shp_dir: Optional[str] = None,
        output_image_dir: Optional[str] = None,
    ):
        """Run full polygon extraction workflow"""
        self.slice_las()
        self.transform_to_tensor()
        self.prepare_features()
        self.predict_buildings()
        self.postprocess_buildings()
        self.export_polygons(output_shp_dir=output_shp_dir, output_image_dir=output_image_dir)
