import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext
import threading
import sys
import os
import time
import ctypes
import platform
import subprocess
import webbrowser

# DPI awareness for Windows to prevent blurriness
try:
    ctypes.windll.shcore.SetProcessDpiAwareness(1)
except:
    try:
        ctypes.windll.user32.SetProcessDPIAware()
    except:
        pass

# Import your pipeline modules here
from preprocessing.transformation_las2npy import main_not_parallel_transform_to_tensor
from preprocessing.image_generator import main_not_parallel_tensor_to_image
from preprocessing.slicing_las_python import main_not_parallel_cut_tiles
import preprocessing.config_preprocessing as cp
import postprocessing.config_postprocessing as cpost
import predictor_multiclass_segmentation.config_prediction as cpred
import polygon_generator.config_polygon_generator as cpg
from predictor_multiclass_segmentation.predict_multiclass_segmentation import main_prediction
from polygon_generator.polygon_generator import main_polygon_generator
from postprocessing.join_img import main_join_img
from generate_colored_las_3D.generate_colored_las_3D import mask_to_las_with_class_nn_rgb
from generate_colored_las_3D.generate_class_las_3D import mask_to_las_with_class_only
import generate_colored_las_3D.config_colored_las as ccl

class MainApp:
    def __init__(self, root):
        self.root = root
        self.root.title("LAS Multi-file Independent Pipeline")
        self.root.state('zoomed')  # Fullscreen on Windows
        
        # Configure grid weights for proper layout
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)

        style = ttk.Style()
        style.theme_use('clam')

        self.current_project_dir = None
        self.init_ui()

    def init_ui(self):
        # Main frame
        main_frame = ttk.Frame(self.root)
        main_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        main_frame.grid_rowconfigure(1, weight=1)
        main_frame.grid_columnconfigure(0, weight=1)

        # Top buttons frame
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        button_frame.grid_columnconfigure(1, weight=1)

        self.upload_btn = ttk.Button(button_frame, text="Upload LAS Files", command=self.upload_files)
        self.upload_btn.grid(row=0, column=0, padx=(0, 10))

        self.run_btn = ttk.Button(button_frame, text="Run Full Pipeline", command=self.run_full_pipeline)
        self.run_btn.grid(row=0, column=1, padx=(0, 10))

        self.project_dir_btn = ttk.Button(button_frame, text="Open Project Directory", command=self.open_project_directory, state=tk.DISABLED)
        self.project_dir_btn.grid(row=0, column=2)

        # Log frame
        log_frame = ttk.Frame(main_frame)
        log_frame.grid(row=1, column=0, sticky="nsew")
        log_frame.grid_rowconfigure(0, weight=1)
        log_frame.grid_columnconfigure(0, weight=1)

        self.log_text = scrolledtext.ScrolledText(log_frame, height=30, state=tk.DISABLED)
        self.log_text.grid(row=0, column=0, sticky="nsew")

        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        self.status_bar = ttk.Label(main_frame, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.grid(row=2, column=0, sticky="ew", pady=(10, 0))

        # Progress bar
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(main_frame, variable=self.progress_var, maximum=100)
        self.progress_bar.grid(row=3, column=0, sticky="ew", pady=(5, 0))

    def log(self, message):
        timestamp = time.strftime("%H:%M:%S")
        msg = f"[{timestamp}] {message}\n"
        self.log_text.config(state=tk.NORMAL)
        self.log_text.insert(tk.END, msg)
        self.log_text.see(tk.END)
        self.log_text.config(state=tk.DISABLED)
        self.root.update_idletasks()

    def update_status(self, message):
        self.status_var.set(message)
        self.root.update_idletasks()

    def update_progress(self, value):
        self.progress_var.set(value)
        self.root.update_idletasks()

    def open_project_directory(self):
        if self.current_project_dir and os.path.exists(self.current_project_dir):
            if platform.system() == "Windows":
                os.startfile(self.current_project_dir)
            elif platform.system() == "Darwin":  # macOS
                subprocess.run(["open", self.current_project_dir])
            else:  # Linux
                subprocess.run(["xdg-open", self.current_project_dir])
        else:
            self.log("No project directory available.")

    def upload_files(self):
        files = filedialog.askopenfilenames(title="Select LAS Files", filetypes=[("LAS files", "*.las")])
        if files:
            self.files = files
            self.log(f"Selected {len(files)} LAS files for processing.")
            self.update_status(f"Ready to process {len(files)} files")

    def run_full_pipeline(self):
        if not hasattr(self, 'files') or not self.files:
            self.log("No files selected.")
            return
        self.run_btn.config(state=tk.DISABLED)
        self.upload_btn.config(state=tk.DISABLED)
        self.project_dir_btn.config(state=tk.DISABLED)
        threading.Thread(target=self.run_pipeline_for_all_files, daemon=True).start()

    def run_pipeline_for_all_files(self):
        # Create one main project directory
        project_timestamp = time.strftime('%d_%m_%Y_%H_%M_%S')
        main_project_dir = f"project_batch_{project_timestamp}"
        os.makedirs(main_project_dir, exist_ok=True)
        
        # Store the main project directory for the button
        self.current_project_dir = os.path.abspath(main_project_dir)
        
        self.log(f"Created main project directory: {main_project_dir}")
        self.update_status(f"Processing {len(self.files)} files in project: {main_project_dir}")
        
        total_files = len(self.files)
        for i, file_path in enumerate(self.files):
            try:
                progress = (i / total_files) * 100
                self.update_progress(progress)
                self.update_status(f"Processing file {i+1} of {total_files}")
                self.process_single_file(file_path, main_project_dir)
            except Exception as e:
                self.log(f"Error processing {os.path.basename(file_path)}: {e}")
        
        self.update_progress(100)
        self.update_status("All pipelines completed")
        self.log("✅ All pipelines completed.")
        self.log(f"All results saved in: {main_project_dir}")
        self.run_btn.config(state=tk.NORMAL)
        self.upload_btn.config(state=tk.NORMAL)
        self.project_dir_btn.config(state=tk.NORMAL)

    def process_single_file(self, file_path, main_project_dir):
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        # Create subdirectory for this file within the main project directory
        file_project_dir = os.path.join(main_project_dir, base_name)
        os.makedirs(file_project_dir, exist_ok=True)

        folders = ["las", "las_cut", "tensors", "img_features", "img_features_join",
                   "img_rgb", "img_rgb_join", "img_predict_multi_class",
                   "img_predict_multi_class_join", "las_colored"]
        dirs = {}
        for folder in folders:
            # Create folder directory first
            folder_dir = os.path.join(file_project_dir, folder)
            os.makedirs(folder_dir, exist_ok=True)
            # Then create subdirectory with file name inside each folder
            dir_path = os.path.join(folder_dir, base_name)
            os.makedirs(dir_path, exist_ok=True)
            dirs[folder] = dir_path

        dest_path = os.path.join(dirs["las"], os.path.basename(file_path))
        with open(file_path, "rb") as src, open(dest_path, "wb") as dst:
            dst.write(src.read())
        self.log(f"Uploaded and prepared {file_path}.")

        try:
            self.log(f"Processing {base_name}: Splitting tiles...")
            self.update_status(f"Processing {base_name}: Splitting tiles...")
            main_not_parallel_cut_tiles(dirs["las"], dirs["las_cut"], tile_size=250)
            self.log(f"{base_name}: Tile splitting completed.")

            self.log(f"{base_name}: Generating features...")
            self.update_status(f"Processing {base_name}: Generating features...")
            main_not_parallel_transform_to_tensor(
                dirs["las_cut"], dirs["tensors"],
                cp.feature_input_tensor, cp.feature_output_tensor,
                cp.num_points_lim, cp.M_tensor_size, cp.K_nn
            )
            self.log(f"{base_name}: Feature generation completed.")

            self.log(f"{base_name}: Generating feature images...")
            self.update_status(f"Processing {base_name}: Generating feature images...")
            main_not_parallel_tensor_to_image(
                dirs["tensors"], dirs["img_features"],
                cp.feature_output_tensor, cp.channels_visualisation
            )
            main_join_img(dirs["img_features"], dirs["img_features_join"])
            self.log(f"{base_name}: Feature images generated.")

            self.log(f"{base_name}: Generating RGB images...")
            self.update_status(f"Processing {base_name}: Generating RGB images...")
            main_not_parallel_tensor_to_image(
                dirs["tensors"], dirs["img_rgb"],
                cp.feature_output_tensor, cp.channels_visualisation_rgb
            )
            main_join_img(dirs["img_rgb"], dirs["img_rgb_join"])
            self.log(f"{base_name}: RGB images generated.")

            self.log(f"{base_name}: Running prediction...")
            self.update_status(f"Processing {base_name}: Running prediction...")
            main_prediction(dirs["img_features"], dirs["img_predict_multi_class"], cpred.checkpoint_path)
            main_join_img(dirs["img_predict_multi_class"], dirs["img_predict_multi_class_join"])
            self.log(f"{base_name}: Prediction completed.")

            self.log(f"{base_name}: Generating colored LAS...")
            self.update_status(f"Processing {base_name}: Generating colored LAS...")
            file_las = os.path.join(dirs["las"], os.path.basename(file_path))
            joined_images = [f for f in os.listdir(dirs["img_predict_multi_class_join"]) if f.endswith('.png')]
            if not joined_images:
                raise Exception("No joined prediction images found.")
            file_img_colored = os.path.join(dirs["img_predict_multi_class_join"], joined_images[0])
            output_las_colored = os.path.join(dirs["las_colored"], f"{base_name}_colored.las")

            mask_to_las_with_class_nn_rgb(
                las_file_path=file_las,
                image_file_path=file_img_colored,
                output_las_path=output_las_colored,
                class_colors=ccl.class_colors
            )
            
            output_las_with_class = os.path.join(dirs["las_colored"], f"{base_name}_with_class.las")
            mask_to_las_with_class_only(
                las_file_path=file_las,
                image_file_path=file_img_colored,
                output_las_path=output_las_with_class,
                class_colors=ccl.class_colors
            )      


            self.log(f"{base_name}: Colored LAS generated at {output_las_colored}.")
            self.update_status(f"Completed processing {base_name}")
        except Exception as e:
            self.log(f"❌ Error processing {base_name}: {e}")
            self.update_status(f"Error processing {base_name}")
            raise

def main():
    root = tk.Tk()
    app = MainApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
