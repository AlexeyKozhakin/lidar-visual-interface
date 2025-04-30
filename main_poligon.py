#import streamlit as st
import os
import numpy as np
from PIL import Image
from preprocessing.transformation_las2npy import main_parallel_transform_to_tensor
from preprocessing.image_generator import main_parallel_tensor_to_image
from preprocessing.slicing_las import main_parallel_cut_tiles
import time
import preprocessing.config_preprocessing as cp
import postprocessing.config_postprocessing as cpost
import predictor_building_segmentation.config_prediction as cpred
import polygon_generator.config_polygon_generator as cpg
from predictor_building_segmentation.predict_building_segmentation import main_prediction
from polygon_generator.polygon_generator import main_polygon_generator
from utils import show_existing_files
from postprocessing.join_img import main_join_img

# ============= Set up all temp folders for processing =======================
save_dir = cp.path_las_before_cut
os.makedirs(save_dir, exist_ok=True) # folder for las
os.makedirs(cp.path_out_tensors, exist_ok=True) # folder for tensors
os.makedirs(cp.path_image_features, exist_ok=True) # folder for images features
os.makedirs(cp.path_image_rgb, exist_ok=True) # folder for images rgb
os.makedirs(cpost.path_image_features_join, exist_ok=True) # folder for join image features
os.makedirs(cpost.path_image_rgb_join, exist_ok=True) # folder for join image rgb

os.makedirs(cpred.output_img_segment_buildings_predict, exist_ok=True) # folder for join image rgb
os.makedirs(cpost.path_image_prediction_join, exist_ok=True) # folder for join image rgb

os.makedirs(cpg.path_image_contours, exist_ok=True) # folder for join image rgb
os.makedirs(cpg.path_polygons_shp, exist_ok=True) # folder for join image rgb

# =================== Main functions (Needs to move to special module) =========
def load_las_files(uploaded_files):
    """Save uploaded LAS files to the temp/las directory."""
    os.makedirs(save_dir, exist_ok=True)  # Create folder if it doesn't exist
    
    saved_files = []
    
    for uploaded_file in uploaded_files:
        file_path = os.path.join(save_dir, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())  # Write file contents
        saved_files.append(file_path)
    
    st.session_state.las_files = saved_files  # Save file paths to session
    st.success(f"Uploaded and saved {len(uploaded_files)} files to {save_dir}")

def process_pipeline():
    """Run data processing with information output in Streamlit."""
    st.info("Starting data processing...")
    
    input_directory = cp.path_input_las_for_2d  # Path to directory with LAS files
    output_directory = cp.path_out_tensors
    M = cp.M_tensor_size
    K = cp.K_nn
    
    start = time.time()
    main_parallel_transform_to_tensor(
        input_directory, output_directory,
        cp.feature_input_tensor, cp.feature_output_tensor, 
        cp.num_points_lim, M, K
    )
    end = time.time()
    
    elapsed_time = round(end - start, 2)
    st.success(f"Processing completed! Execution time: {elapsed_time} seconds.")    

def process_images(input_dir, output_dir, channels_visualisation):
    start = time.time()
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    main_parallel_tensor_to_image(input_dir, output_dir,
                                cp.feature_output_tensor, channels_visualisation)
    
    st.success("Image generation completed!")

# ============================== Start Interface =============================================
st.title("LAS File Processing Prototype Application")

uploaded_files = st.file_uploader("Upload LAS files", accept_multiple_files=True, type=["las"])
if uploaded_files:
    load_las_files(uploaded_files)

# ====================================== Check for existing LAS files ===============================
show_existing_files(save_dir, name='LAS files', ext='.las')

if st.button("Split on 250m x 250m"):
    input_directory = cp.path_las_before_cut
    output_directory = cp.path_las_after_cut
    main_parallel_cut_tiles(input_directory, output_directory, tile_size=250)


if st.button("Generate Features"):
    process_pipeline()

# ====================================== Check for existing tensor files ===============================
#show_existing_files(cp.path_out_tensors, name='Tensors', ext='.npy')

# if "processed_images" in st.session_state and st.session_state.processed_images:
#     st.subheader("Processed Images")
#     for img in st.session_state.processed_images:
#         st.image(img, caption="Processed Image", use_column_width=True)

# ======================================= Generate Features Images ============================================
if st.button("Generate Feature Images"):
    process_images(cp.path_tensor_to_visual, cp.path_image_features, cp.channels_visualisation)
    input_folder = cp.path_image_features
    output_file = cpost.path_image_features_join
    main_join_img(input_folder, output_file)
    # ====================================== Check for existing feature image files ===============================
show_existing_files(cpost.path_image_features_join, name='Feature Images', ext='.png')

# ======================================= Generate RGB Images ============================================
if st.button("Generate RGB Data Images"):
    process_images(cp.path_tensor_to_visual, cp.path_image_rgb, cp.channels_visualisation_rgb)
    input_folder = cp.path_image_rgb
    output_file = cpost.path_image_rgb_join
    main_join_img(input_folder, output_file)

# ====================================== Check for existing RGB image files ===============================
show_existing_files(cpost.path_image_rgb_join, name='RGB Data Images', ext='.png')

# ======================================= Predict Images ============================================
if st.button("Predict"):
    main_prediction(cp.path_image_features, 
                    cpred.output_img_segment_buildings_predict, 
                    cpred.checkpoint_path_features)

    input_folder = cpred.output_img_segment_buildings_predict
    output_file = cpost.path_image_prediction_join
    main_join_img(input_folder, output_file)

# ====================================== Check predict images files ===============================
show_existing_files(cpost.path_image_prediction_join, name='Predict Data Images', ext='.png')

# =======================================  Generate polygons ============================================
if st.button("Generate Polygons"):
    # 📁 Путь к папкам из config
    input_dir = cpost.path_image_prediction_join
    output_image_dir = cpg.path_image_contours
    output_shp_dir = cpg.path_polygons_shp

    # ⚙️ Запуск обработки
    main_polygon_generator(input_dir, output_image_dir, output_shp_dir,
                      min_area=cpg.min_area, contour_thickness=cpg.contour_thickness)

# ====================================== Check predict images with polygons ===============================
show_existing_files(cpg.path_image_contours, name='Polygon Data Images', ext='.png')
