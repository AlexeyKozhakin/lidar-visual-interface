import sys
import os
import time
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QPushButton, QFileDialog,
    QLabel, QTextEdit, QProgressBar
)
from PyQt5.QtCore import Qt

from preprocessing.transformation_las2npy import main_not_parallel_transform_to_tensor
from preprocessing.image_generator import main_not_parallel_tensor_to_image
from preprocessing.slicing_las_python import main_not_parallel_cut_tiles
import preprocessing.config_preprocessing as cp
import postprocessing.config_postprocessing as cpost
import predictor_building_segmentation.config_prediction as cpred
import polygon_generator.config_polygon_generator as cpg
from predictor_building_segmentation.predict_building_segmentation import main_prediction
from polygon_generator.polygon_generator import main_polygon_generator
from postprocessing.join_img import main_join_img

class MainApp(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("LAS File Processing - PyQt Version")
        self.resize(600, 800)
        
        layout = QVBoxLayout()

        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)

        self.upload_btn = QPushButton("Upload LAS Files")
        self.upload_btn.clicked.connect(self.upload_files)

        self.pipeline_btn = QPushButton("Run Full Pipeline")
        self.pipeline_btn.clicked.connect(self.run_full_pipeline)

        layout.addWidget(self.upload_btn)
        layout.addWidget(self.pipeline_btn)
        layout.addWidget(QLabel("Progress:"))
        layout.addWidget(self.progress_bar)
        layout.addWidget(QLabel("Logs:"))
        layout.addWidget(self.log_box)

        self.setLayout(layout)
        self.setup_directories()

    def setup_directories(self):
        dirs = [
            cp.path_las_before_cut, cp.path_out_tensors, cp.path_image_features,
            cp.path_image_rgb, cpost.path_image_features_join, cpost.path_image_rgb_join,
            cpred.output_img_segment_buildings_predict, cpost.path_image_prediction_join,
            cpg.path_image_contours, cpg.path_polygons_shp
        ]
        for d in dirs:
            os.makedirs(d, exist_ok=True)

    def log(self, message):
        self.log_box.append(message)
        QApplication.processEvents()

    def set_progress(self, percent):
        self.progress_bar.setValue(percent)
        QApplication.processEvents()

    def upload_files(self):
        files, _ = QFileDialog.getOpenFileNames(self, "Upload LAS Files", "", "LAS files (*.las)")
        if files:
            for file_path in files:
                dest_path = os.path.join(cp.path_las_before_cut, os.path.basename(file_path))
                os.makedirs(cp.path_las_before_cut, exist_ok=True)
                with open(file_path, "rb") as fsrc, open(dest_path, "wb") as fdst:
                    fdst.write(fsrc.read())
            self.log(f"Uploaded and saved {len(files)} LAS files.")

    def split_tiles(self):
        self.log("1. Splitting LAS files into tiles...")
        main_not_parallel_cut_tiles(cp.path_las_before_cut, cp.path_las_after_cut, tile_size=250)
        self.log("1. Splitting completed!")
        self.set_progress(15)

    def generate_features(self):
        self.log("2. Starting feature generation...")
        start = time.time()
        main_not_parallel_transform_to_tensor(
            cp.path_input_las_for_2d, cp.path_out_tensors,
            cp.feature_input_tensor, cp.feature_output_tensor, 
            cp.num_points_lim, cp.M_tensor_size, cp.K_nn
        )
        end = time.time()
        self.log(f"2. Feature generation completed in {round(end - start, 2)} seconds.")
        self.set_progress(35)

    def generate_feature_images(self):
        self.log("3. Generating feature images...")
        start = time.time()
        main_not_parallel_tensor_to_image(cp.path_tensor_to_visual, cp.path_image_features,
                                          cp.feature_output_tensor, cp.channels_visualisation)
        main_join_img(cp.path_image_features, cpost.path_image_features_join)
        end = time.time()
        self.log(f"3. Feature images generated in {round(end - start, 2)} seconds.")
        self.set_progress(50)

    def generate_rgb_images(self):
        self.log("4. Generating RGB images...")
        start = time.time()
        main_not_parallel_tensor_to_image(cp.path_tensor_to_visual, cp.path_image_rgb,
                                          cp.feature_output_tensor, cp.channels_visualisation_rgb)
        main_join_img(cp.path_image_rgb, cpost.path_image_rgb_join)
        end = time.time()
        self.log(f"4. RGB images generated in {round(end - start, 2)} seconds.")
        self.set_progress(65)

    def predict(self):
        self.log("5. Starting prediction...")
        main_prediction(
            cp.path_image_features,
            cpred.output_img_segment_buildings_predict,
            cpred.checkpoint_path_features
        )
        main_join_img(cpred.output_img_segment_buildings_predict, cpost.path_image_prediction_join)
        self.log("5. Prediction completed!")
        self.set_progress(85)

    def generate_polygons(self):
        self.log("6. Generating polygons...")
        main_polygon_generator(
            cpost.path_image_prediction_join,
            cpg.path_image_contours,
            cpg.path_polygons_shp,
            min_area=cpg.min_area,
            contour_thickness=cpg.contour_thickness
        )
        self.log("6. Polygons generated!")
        self.set_progress(100)

    def run_full_pipeline(self):
        self.pipeline_btn.setEnabled(False)
        self.set_progress(0)
        try:
            self.split_tiles()
            self.generate_features()
            self.generate_feature_images()
            self.generate_rgb_images()
            self.predict()
            self.generate_polygons()
            self.log("✅ Pipeline finished successfully!")
        except Exception as e:
            self.log(f"❌ Error during pipeline: {e}")
        finally:
            self.pipeline_btn.setEnabled(True)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainApp()
    window.show()
    sys.exit(app.exec_())
