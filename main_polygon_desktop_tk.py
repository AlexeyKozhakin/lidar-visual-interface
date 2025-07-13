import os
import sys
import time
import threading
import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext, messagebox

# Import pipeline functions
from preprocessing.transformation_las2npy import main_not_parallel_transform_to_tensor
from preprocessing.image_generator import main_not_parallel_tensor_to_image
from preprocessing.slicing_las_python import main_not_parallel_cut_tiles
import preprocessing.config_preprocessing as cp
import postprocessing.config_postprocessing as cpost
import predictor_building_segmentation.config_prediction as cpred
import predictor_multiclass_segmentation.config_prediction as cmulticlass
import polygon_generator.config_polygon_generator as cpg
from predictor_building_segmentation.predict_building_segmentation import main_prediction
from predictor_multiclass_segmentation.predict_multiclass_segmentation import main_prediction as main_multiclass_prediction
from polygon_generator.polygon_generator import main_polygon_generator
from postprocessing.join_img import main_join_img
import ctypes

try:
    ctypes.windll.shcore.SetProcessDpiAwareness(1)  # For Windows 8.1+
except:
    try:
        ctypes.windll.user32.SetProcessDPIAware()  # For Windows 7
    except:
        pass

class LASApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("LAS File Processing - Tkinter Version")
        self.state('zoomed')  # For Windows
        self.resizable(False, False)

        # 📁 Create unique project directory by time (up to seconds)
        self.project_root = self.generate_project_path(os.getcwd())
        os.makedirs(self.project_root, exist_ok=True)

        self.create_widgets()
        self.setup_directories()


    def open_folder(self, path):
        path = os.path.abspath(path)
        if not os.path.exists(path):
            os.makedirs(path)
        if sys.platform == "win32":
            os.startfile(path)
        elif sys.platform == "darwin":
            os.system(f"open '{path}'")
        else:
            os.system(f"xdg-open '{path}'")

    def create_widgets(self):

        button_width = 30  # Number of characters in width



        self.upload_btn = ttk.Button(self, text="Upload LAS Files", command=self.upload_files, width=button_width)
        self.upload_btn.pack(pady=(15, 5), anchor='center')

        # Checkbox for multi-class segmentation
        self.multiclass_var = tk.BooleanVar()
        self.multiclass_checkbox = ttk.Checkbutton(self, text="Enable Multi-class Segmentation", variable=self.multiclass_var)
        self.multiclass_checkbox.pack(pady=(5, 5), anchor='center')

        self.pipeline_btn = ttk.Button(self, text="Run Full Pipeline", command=self.start_pipeline_thread, width=button_width)
        self.pipeline_btn.pack(pady=(15, 5), anchor='center')

        ttk.Label(self, text="Progress:").pack(pady=5)

        self.progress = ttk.Progressbar(self, mode='determinate')
        self.progress.pack(padx=20, pady=5, fill=tk.X, expand=True)


        ttk.Label(self, text="Logs:").pack(pady=5)
        self.log_box = scrolledtext.ScrolledText(self, wrap=tk.WORD, height=25)
        self.log_box.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

        # Directory buttons under the log
        dir_frame = ttk.LabelFrame(self, text="Open Output Directories")
        dir_frame.pack(pady=10, padx=10, fill=tk.X)

        # Grid settings
        for i in range(2):
            dir_frame.columnconfigure(i, weight=1)

        # Set required width for all buttons
        button_width = 35

        # Directory buttons (in 2 rows by 3 columns)
        dir_frame = ttk.Frame(self)
        dir_frame.pack(pady=10, anchor="center")

        ttk.Button(dir_frame, text="📂 Open Project Root",
                command=lambda: self.open_folder(self.project_root),
                width=button_width).grid(row=0, column=0, padx=5, pady=5, sticky="ew")

        ttk.Button(dir_frame, text="📂 Features Input Folder",
                command=lambda: self.open_folder(os.path.join(self.project_root, cpost.path_image_features_join)),
                width=button_width).grid(row=0, column=1, padx=5, pady=5, sticky="ew")

        ttk.Button(dir_frame, text="📂 RGB Folder",
                command=lambda: self.open_folder(os.path.join(self.project_root, cpost.path_image_rgb_join)),
                width=button_width).grid(row=0, column=2, padx=5, pady=5, sticky="ew")

        ttk.Button(dir_frame, text="📂 Segment Output Folder",
                command=lambda: self.open_folder(os.path.join(self.project_root, cpost.path_image_prediction_join)),
                width=button_width).grid(row=1, column=0, padx=5, pady=5, sticky="ew")

        ttk.Button(dir_frame, text="📂 Contours Building Picture",
                command=lambda: self.open_folder(os.path.join(self.project_root, cpg.path_image_contours)),
                width=button_width).grid(row=1, column=1, padx=5, pady=5, sticky="ew")

        ttk.Button(dir_frame, text="📂 Shape Contours (SHP)",
                command=lambda: self.open_folder(os.path.join(self.project_root, cpg.path_polygons_shp)),
                width=button_width).grid(row=1, column=2, padx=5, pady=5, sticky="ew")

        ttk.Button(dir_frame, text="📂 Multi-class Results",
                command=lambda: self.open_folder(os.path.join(self.project_root, cmulticlass.output_directory_join)),
                width=button_width).grid(row=2, column=0, padx=5, pady=5, sticky="ew")

        # Footer
        footer = ttk.Label(self, text="MUSAC Project", anchor="center", font=("Arial", 10, "italic"))
        footer.pack(side=tk.BOTTOM, pady=5)

    def generate_project_path(self, base_dir):
        from datetime import datetime

        folder_name = datetime.now().strftime("project_%d_%m_%Y_%H_%M_%S")
        return os.path.join(base_dir, folder_name)






    def setup_directories(self):
        dirs = [
            cp.path_las_before_cut, cp.path_out_tensors, cp.path_image_features,
            cp.path_image_rgb, cpost.path_image_features_join, cpost.path_image_rgb_join,
            cpred.output_img_segment_buildings_predict, cpost.path_image_prediction_join,
            cpg.path_image_contours, cpg.path_polygons_shp
        ]
        for d in dirs:
            os.makedirs(os.path.join(self.project_root, d), exist_ok=True)
        
        # Create directories for multi-class segmentation
        multiclass_dirs = [
            cmulticlass.output_directory,
            cmulticlass.output_directory_join
        ]
        for d in multiclass_dirs:
            os.makedirs(os.path.join(self.project_root, d), exist_ok=True)

    def log(self, message):
        self.log_box.insert(tk.END, message + "\n")
        self.log_box.see(tk.END)
        self.update()

    def set_progress(self, value):
        self.progress["value"] = value
        self.update()

    def upload_files(self):
        files = filedialog.askopenfilenames(filetypes=[("LAS files", "*.las")])
        if files:
            # Automatically create new project directory when uploading files
            new_project_path = self.generate_project_path(os.getcwd())
            try:
                os.makedirs(new_project_path)
                self.project_root = new_project_path
                self.log(f"📂 Automatically created new project folder: {new_project_path}")
                self.setup_directories()
                
                # Copy files to new project directory
                for file_path in files:
                    dest_path = os.path.join(self.project_root, cp.path_las_before_cut, os.path.basename(file_path))
                    os.makedirs(os.path.join(self.project_root, cp.path_las_before_cut), exist_ok=True)
                    with open(file_path, "rb") as fsrc, open(dest_path, "wb") as fdst:
                        fdst.write(fsrc.read())
                
                self.log(f"✅ Uploaded and saved {len(files)} LAS files to new project.")
                
            except Exception as e:
                self.log(f"❌ Failed to create project folder: {e}")
                messagebox.showerror("Error", str(e))

    def start_pipeline_thread(self):
        threading.Thread(target=self.run_full_pipeline, daemon=True).start()

    def run_full_pipeline(self):
        self.pipeline_btn.config(state=tk.DISABLED)
        self.set_progress(0)
        try:
            self.log("1. Splitting LAS files into tiles...")
            main_not_parallel_cut_tiles(
                        os.path.join(self.project_root, cp.path_las_before_cut), 
                        os.path.join(self.project_root, cp.path_las_after_cut), tile_size=250
                                        )
            self.log("1. Splitting completed!")
            self.set_progress(15)

            self.log("2. Starting feature generation...")
            start = time.time()
            print()
            main_not_parallel_transform_to_tensor(
                os.path.join(self.project_root, cp.path_input_las_for_2d), os.path.join(self.project_root, cp.path_out_tensors),
                cp.feature_input_tensor, cp.feature_output_tensor,
                cp.num_points_lim, cp.M_tensor_size, cp.K_nn
            )
            self.log(f"2. Feature generation completed in {round(time.time() - start, 2)} sec.")
            self.set_progress(35)

            self.log("3. Generating feature images...")
            start = time.time()
            main_not_parallel_tensor_to_image(
                os.path.join(self.project_root, cp.path_tensor_to_visual), 
                os.path.join(self.project_root, cp.path_image_features),
                cp.feature_output_tensor, 
                cp.channels_visualisation
                                              )
            main_join_img(
                os.path.join(
                        self.project_root, cp.path_image_features), 
                        os.path.join(self.project_root, cpost.path_image_features_join)
                          )
            self.log(f"3. Feature images generated in {round(time.time() - start, 2)} sec.")
            self.set_progress(50)

            self.log("4. Generating RGB images...")
            start = time.time()
            main_not_parallel_tensor_to_image(
                os.path.join(self.project_root, cp.path_tensor_to_visual), 
                os.path.join(self.project_root, cp.path_image_rgb),
                cp.feature_output_tensor,
                cp.channels_visualisation_rgb
                                              )
            main_join_img(
                os.path.join(self.project_root, cp.path_image_rgb), 
                os.path.join(self.project_root, cpost.path_image_rgb_join)
                             )
            self.log(f"4. RGB images generated in {round(time.time() - start, 2)} sec.")
            self.set_progress(65)

            self.log("5. Starting building prediction...")
            main_prediction(os.path.join(self.project_root, cp.path_image_features),
                            os.path.join(self.project_root, cpred.output_img_segment_buildings_predict),
                            cpred.checkpoint_path_features)
            main_join_img(
                os.path.join(self.project_root, cpred.output_img_segment_buildings_predict), 
                os.path.join(self.project_root, cpost.path_image_prediction_join)
                             )
            self.log("5. Building prediction completed!")
            
            # Multi-class segmentation (if enabled)
            if self.multiclass_var.get():
                self.log("5a. Starting multi-class prediction...")
                main_multiclass_prediction(
                    os.path.join(self.project_root, cp.path_image_features),
                    os.path.join(self.project_root, cmulticlass.output_directory),
                    cmulticlass.checkpoint_path
                )
                main_join_img(
                    os.path.join(self.project_root, cmulticlass.output_directory),
                    os.path.join(self.project_root, cmulticlass.output_directory_join)
                )
                self.log("5a. Multi-class prediction completed!")
            
            self.set_progress(85)

            self.log("6. Generating polygons...")
            main_polygon_generator(
                os.path.join(self.project_root, cpost.path_image_prediction_join),
                os.path.join(self.project_root, cpg.path_image_contours),
                os.path.join(self.project_root, cpg.path_polygons_shp),
                min_area=cpg.min_area,
                contour_thickness=cpg.contour_thickness
            )
            self.log("6. Polygons generated!")
            self.set_progress(100)

            self.log("✅ Pipeline finished successfully!")

        except Exception as e:
            self.log(f"❌ Error during pipeline: {e}")
            messagebox.showerror("Error", str(e))

        finally:
            self.pipeline_btn.config(state=tk.NORMAL)

if __name__ == "__main__":
    app = LASApp()
    app.mainloop()
