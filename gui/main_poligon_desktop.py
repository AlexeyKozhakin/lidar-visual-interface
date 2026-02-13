import sys
import os
import time
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QPushButton, QFileDialog, QLabel, QTextEdit
)
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

        self.upload_btn = QPushButton("Upload LAS Files")
        self.upload_btn.clicked.connect(self.upload_files)

        self.split_btn = QPushButton("Split on 250m x 250m")
        self.split_btn.clicked.connect(self.split_tiles)

        self.generate_features_btn = QPushButton("Generate Features")
        self.generate_features_btn.clicked.connect(self.generate_features)

        self.generate_feature_images_btn = QPushButton("Generate Feature Images")
        self.generate_feature_images_btn.clicked.connect(self.generate_feature_images)

        self.generate_rgb_images_btn = QPushButton("Generate RGB Images")
        self.generate_rgb_images_btn.clicked.connect(self.generate_rgb_images)

        self.predict_btn = QPushButton("Predict")
        self.predict_btn.clicked.connect(self.predict)

        self.generate_polygons_btn = QPushButton("Generate Polygons")
        self.generate_polygons_btn.clicked.connect(self.generate_polygons)

        layout.addWidget(self.upload_btn)
        layout.addWidget(self.split_btn)
        layout.addWidget(self.generate_features_btn)
        layout.addWidget(self.generate_feature_images_btn)
        layout.addWidget(self.generate_rgb_images_btn)
        layout.addWidget(self.predict_btn)
        layout.addWidget(self.generate_polygons_btn)
        layout.addWidget(QLabel("Logs:"))
        layout.addWidget(self.log_box)

        self.setLayout(layout)

        # Create needed folders
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
        self.log("Splitting LAS files into tiles...")
        main_not_parallel_cut_tiles(cp.path_las_before_cut, cp.path_las_after_cut, tile_size=250)
        self.log("Splitting completed!")

    def generate_features(self):
        self.log("Starting feature generation...")
        start = time.time()
        main_not_parallel_transform_to_tensor(
            cp.path_input_las_for_2d, cp.path_out_tensors,
            cp.feature_input_tensor, cp.feature_output_tensor, 
            cp.num_points_lim, cp.M_tensor_size, cp.K_nn
        )
        end = time.time()
        self.log(f"Feature generation completed in {round(end - start, 2)} seconds.")

    def generate_feature_images(self):
        self.log("Generating feature images...")
        start = time.time()
        main_not_parallel_tensor_to_image(cp.path_tensor_to_visual, cp.path_image_features,
                                      cp.feature_output_tensor, cp.channels_visualisation)
        main_join_img(cp.path_image_features, cpost.path_image_features_join)
        end = time.time()
        self.log(f"Feature images generated in {round(end - start, 2)} seconds.")

    def generate_rgb_images(self):
        self.log("Generating RGB images...")
        start = time.time()
        main_not_parallel_tensor_to_image(cp.path_tensor_to_visual, cp.path_image_rgb,
                                      cp.feature_output_tensor, cp.channels_visualisation_rgb)
        main_join_img(cp.path_image_rgb, cpost.path_image_rgb_join)
        end = time.time()
        self.log(f"RGB images generated in {round(end - start, 2)} seconds.")

    def predict(self):
        self.log("Starting prediction...")
        main_prediction(
            cp.path_image_features,
            cpred.output_img_segment_buildings_predict,
            cpred.checkpoint_path_features
        )
        main_join_img(cpred.output_img_segment_buildings_predict, cpost.path_image_prediction_join)
        self.log("Prediction completed!")

    def generate_polygons(self):
        self.log("Generating polygons...")
        main_polygon_generator(
            cpost.path_image_prediction_join,
            cpg.path_image_contours,
            cpg.path_polygons_shp,
            min_area=cpg.min_area,
            contour_thickness=cpg.contour_thickness
        )
        self.log("Polygons generated!")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainApp()
    window.show()
    sys.exit(app.exec_())
