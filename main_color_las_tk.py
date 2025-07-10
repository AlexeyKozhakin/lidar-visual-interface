import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext
import threading
import sys
import os
import time
import ctypes

# DPI awareness for Windows to prevent blurriness
try:
    ctypes.windll.shcore.SetProcessDpiAwareness(1)  # For Windows 8.1+
except:
    try:
        ctypes.windll.user32.SetProcessDPIAware()  # For Windows 7
    except:
        pass

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
        self.root.title("LAS File Processing - Tkinter Version")
        
        # Make application fullscreen
        self.root.state('zoomed')  # For Windows
        # Alternative for other platforms:
        # self.root.attributes('-zoomed', True)  # For Linux
        # self.root.attributes('-fullscreen', True)  # For macOS
        
        self.root.resizable(True, True)
        
        # Configure style
        style = ttk.Style()
        style.theme_use('clam')
        
        # Project directory - will be set when files are uploaded
        self.project_dir = None
        # Lists to store uploaded files
        self.las_files = []
        self.las_basenames = []
        
        self.init_ui()

    def init_ui(self):
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky="nsew")
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(4, weight=1)
        
        # Title
        title_label = ttk.Label(main_frame, text="LiDAR 3D Classification & Colored LAS System", 
                               font=('Arial', 16, 'bold'))
        title_label.grid(row=0, column=0, columnspan=2, pady=(0, 20))
        
        # Upload button
        self.upload_btn = ttk.Button(main_frame, text="Upload LAS Files", 
                                    command=self.upload_files)
        self.upload_btn.grid(row=1, column=0, columnspan=2, pady=(0, 10), sticky="ew")
        
        # Pipeline button
        self.pipeline_btn = ttk.Button(main_frame, text="Run Full Pipeline", 
                                      command=self.run_full_pipeline)
        self.pipeline_btn.grid(row=2, column=0, columnspan=2, pady=(0, 10), sticky="ew")
        
        # Open project folder button
        self.open_folder_btn = ttk.Button(main_frame, text="Open Project Folder", 
                                         command=self.open_project_folder, state=tk.DISABLED)
        self.open_folder_btn.grid(row=3, column=0, columnspan=2, pady=(0, 10), sticky="ew")
        
        # Progress section
        progress_frame = ttk.LabelFrame(main_frame, text="Progress", padding="5")
        progress_frame.grid(row=4, column=0, columnspan=2, pady=(0, 10), sticky="ew")
        progress_frame.columnconfigure(0, weight=1)
        
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(progress_frame, variable=self.progress_var, 
                                           maximum=100, length=400)
        self.progress_bar.grid(row=0, column=0, sticky="ew", pady=5)
        
        self.progress_label = ttk.Label(progress_frame, text="Ready")
        self.progress_label.grid(row=1, column=0, pady=(0, 5))
        
        # Log section
        log_frame = ttk.LabelFrame(main_frame, text="Processing Log", padding="5")
        log_frame.grid(row=5, column=0, columnspan=2, sticky="nsew")
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)
        
        # Create text widget with scrollbar
        text_frame = ttk.Frame(log_frame)
        text_frame.grid(row=0, column=0, sticky="nsew")
        text_frame.columnconfigure(0, weight=1)
        text_frame.rowconfigure(0, weight=1)
        
        self.log_text = scrolledtext.ScrolledText(text_frame, height=20, width=80, 
                                                 wrap=tk.WORD, state=tk.DISABLED)
        self.log_text.grid(row=0, column=0, sticky="nsew")
        
        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var, 
                              relief=tk.SUNKEN, anchor=tk.W)
        status_bar.grid(row=6, column=0, columnspan=2, sticky="ew", pady=(10, 0))

    def normpath(self, *args):
        """Return normalized absolute path (Windows/Unix safe)"""
        return os.path.abspath(os.path.join(*args))

    def create_project_directory(self):
        """Create project directory with timestamp"""
        timestamp = time.strftime("%d_%m_%Y_%H_%M_%S")
        self.project_dir = f"project_3Dcolored_{timestamp}"
        os.makedirs(self.project_dir, exist_ok=True)
        self.log(f"Created project directory: {self.project_dir}")
        return self.project_dir

    def setup_directories(self):
        """Create necessary directories for processing within project directory"""
        if not self.project_dir:
            raise Exception("Project directory not created. Please upload files first.")
            
        # Create all subdirectories within project directory
        dirs = [
            self.normpath(self.project_dir, "las"),
            self.normpath(self.project_dir, "las_cut"),
            self.normpath(self.project_dir, "tensors"),
            self.normpath(self.project_dir, "img_features"),
            self.normpath(self.project_dir, "img_features_join"),
            self.normpath(self.project_dir, "img_features_join_all"),
            self.normpath(self.project_dir, "img_rgb"),
            self.normpath(self.project_dir, "img_rgb_join"),
            self.normpath(self.project_dir, "img_rgb_join_all"),
            self.normpath(self.project_dir, "img_predict_multi_class"),
            self.normpath(self.project_dir, "img_predict_multi_class_join"),
            self.normpath(self.project_dir, "img_predict_multi_class_join_all"),
            self.normpath(self.project_dir, "las_colored"),
            self.normpath(self.project_dir, "las_with_class")
        ]
        for d in dirs:
            os.makedirs(d, exist_ok=True)
        
        self.log(f"Created {len(dirs)} subdirectories in project folder")

    def log(self, message):
        """Add message to log with timestamp"""
        timestamp = time.strftime("%H:%M:%S")
        log_message = f"[{timestamp}] {message}\n"
        
        self.log_text.config(state=tk.NORMAL)
        self.log_text.insert(tk.END, log_message)
        self.log_text.see(tk.END)
        self.log_text.config(state=tk.DISABLED)
        
        # Update status
        self.status_var.set(message)
        self.root.update_idletasks()

    def set_progress(self, percent):
        """Update progress bar"""
        self.progress_var.set(percent)
        self.progress_label.config(text=f"Progress: {percent}%")
        self.root.update_idletasks()

    def upload_files(self):
        """Upload LAS files using file dialog"""
        files = filedialog.askopenfilenames(
            title="Select LAS Files",
            filetypes=[("LAS files", "*.las"), ("All files", "*.*")]
        )
        
        if files:
            # Create new project directory
            self.create_project_directory()
            self.setup_directories()
            
            # Clear previous files
            self.las_files.clear()
            self.las_basenames.clear()
            uploaded_count = 0
            for file_path in files:
                try:
                    if self.project_dir is None:
                        raise Exception("Project directory not created")
                    dest_path = self.normpath(self.project_dir, "las", os.path.basename(file_path))
                    with open(file_path, "rb") as fsrc, open(dest_path, "wb") as fdst:
                        fdst.write(fsrc.read())
                    self.las_files.append(dest_path)
                    self.las_basenames.append(os.path.splitext(os.path.basename(file_path))[0])
                    uploaded_count += 1
                except Exception as e:
                    self.log(f"Error uploading {os.path.basename(file_path)}: {e}")
            self.log(f"Successfully uploaded {uploaded_count} LAS files to {self.project_dir}")
            self.status_var.set(f"Uploaded {uploaded_count} files to {self.project_dir}")
            # Enable the open folder button
            self.open_folder_btn.config(state=tk.NORMAL)

    def open_project_folder(self):
        """Open the project folder in file explorer"""
        if self.project_dir and os.path.exists(self.project_dir):
            try:
                import subprocess
                import platform
                
                system = platform.system()
                if system == "Windows":
                    subprocess.run(["explorer", self.project_dir], check=True)
                elif system == "Darwin":  # macOS
                    subprocess.run(["open", self.project_dir], check=True)
                else:  # Linux
                    subprocess.run(["xdg-open", self.project_dir], check=True)
                
                self.log(f"Opened project folder: {self.project_dir}")
                self.status_var.set(f"Opened folder: {self.project_dir}")
            except Exception as e:
                self.log(f"Error opening folder: {e}")
                self.status_var.set("Error opening folder")
        else:
            self.log("No project folder available")
            self.status_var.set("No project folder")

    def split_tiles(self):
        """Split all LAS files into tiles (аналогично main_polygon_desktop_tk.py)"""
        self.log("1. Splitting all LAS files into tiles...")
        try:
            if not self.project_dir:
                raise Exception("Project directory not created. Please upload files first.")
            input_dir = self.normpath(self.project_dir, "las")
            output_dir = self.normpath(self.project_dir, "las_cut")
            main_not_parallel_cut_tiles(input_dir, output_dir, tile_size=250)
            self.log("1. Splitting completed for all files!")
            self.set_progress(15)
        except Exception as e:
            self.log(f"Error in splitting: {e}")
            raise

    def generate_features(self):
        """Generate features from all LAS tiles in las_cut"""
        self.log("2. Starting feature generation for all tiles...")
        try:
            if not self.project_dir:
                raise Exception("Project directory not created. Please upload files first.")
            start = time.time()
            las_cut_dir = self.normpath(self.project_dir, "las_cut")
            tensors_dir = self.normpath(self.project_dir, "tensors")
            main_not_parallel_transform_to_tensor(
                las_cut_dir, tensors_dir,
                cp.feature_input_tensor, cp.feature_output_tensor,
                cp.num_points_lim, cp.M_tensor_size, cp.K_nn
            )
            end = time.time()
            self.log(f"2. Feature generation for all tiles completed in {round(end - start, 2)} seconds.")
            self.set_progress(35)
        except Exception as e:
            self.log(f"Error in feature generation: {e}")
            raise

    def generate_feature_images(self):
        """Generate feature images for all tensors"""
        self.log("3. Generating feature images for all tiles...")
        try:
            if not self.project_dir:
                raise Exception("Project directory not created. Please upload files first.")
            start = time.time()
            tensors_dir = self.normpath(self.project_dir, "tensors")
            features_output_dir = self.normpath(self.project_dir, "img_features")
            os.makedirs(features_output_dir, exist_ok=True)
            main_not_parallel_tensor_to_image(
                tensors_dir, features_output_dir,
                cp.feature_output_tensor, cp.channels_visualisation
            )
            end = time.time()
            self.log(f"3. Feature images for all tiles generated in {round(end - start, 2)} seconds.")
            self.set_progress(50)
        except Exception as e:
            self.log(f"Error in feature image generation: {e}")
            raise

    def generate_rgb_images(self):
        """Generate RGB images for all tensors"""
        self.log("4. Generating RGB images for all tiles...")
        try:
            if not self.project_dir:
                raise Exception("Project directory not created. Please upload files first.")
            start = time.time()
            tensors_dir = self.normpath(self.project_dir, "tensors")
            rgb_output_dir = self.normpath(self.project_dir, "img_rgb")
            os.makedirs(rgb_output_dir, exist_ok=True)
            main_not_parallel_tensor_to_image(
                tensors_dir, rgb_output_dir,
                cp.feature_output_tensor, cp.channels_visualisation_rgb
            )
            end = time.time()
            self.log(f"4. RGB images for all tiles generated in {round(end - start, 2)} seconds.")
            self.set_progress(65)
        except Exception as e:
            self.log(f"Error in RGB image generation: {e}")
            raise

    def predict(self):
        """Run prediction for all feature images"""
        self.log("5. Starting prediction for all tiles...")
        try:
            if not self.project_dir:
                raise Exception("Project directory not created. Please upload files first.")
            features_dir = self.normpath(self.project_dir, "img_features")
            feature_imgs = [f for f in os.listdir(features_dir) if f.endswith('.png')]
            predict_output_dir = self.normpath(self.project_dir, "img_predict_multi_class")
            os.makedirs(predict_output_dir, exist_ok=True)
            for feature_img in feature_imgs:
                feature_img_path = self.normpath(features_dir, feature_img)
                predict_img_output = self.normpath(predict_output_dir, feature_img)
                main_prediction(feature_img_path, predict_img_output, cpred.checkpoint_path)
                self.log(f"5. Prediction completed for {feature_img}!")
            self.log("5. Prediction for all tiles completed!")
            self.set_progress(85)
        except Exception as e:
            self.log(f"Error in prediction: {e}")
            raise

    def generate_color_las(self):
        """Generate colored LAS and LAS with classes for all files"""
        self.log("6. Generating colored LAS and LAS with classes for all files...")
        try:
            for las_file, basename in zip(self.las_files, self.las_basenames):
                las_input_dir = self.normpath(self.project_dir, "las")
                predict_join_dir = self.normpath(self.project_dir, "img_predict_multi_class_join")
                las_colored_dir = self.normpath(self.project_dir, "las_colored")
                las_with_class_dir = self.normpath(self.project_dir, "las_with_class")
                
                file_las = las_file  # las_file уже содержит полный путь к файлу в проекте
                file_img_colored = self.normpath(predict_join_dir, f"{basename}_join.png")
                
                # Generate colored LAS with RGB values
                file_las_colored = self.normpath(las_colored_dir, f"{basename}_colored.las")
                mask_to_las_with_class_nn_rgb(
                    las_file_path=file_las,
                    image_file_path=file_img_colored,
                    output_las_path=file_las_colored,
                    class_colors=ccl.class_colors
                )
                self.log(f"6a. Colored LAS generated for {basename}!")
                
                # Generate LAS with classes only (preserving original RGB)
                file_las_with_class = self.normpath(las_with_class_dir, f"{basename}_with_class.las")
                mask_to_las_with_class_only(
                    las_file_path=file_las,
                    image_file_path=file_img_colored,
                    output_las_path=file_las_with_class,
                    class_colors=ccl.class_colors
                )
                self.log(f"6b. LAS with classes generated for {basename}!")
            
            self.set_progress(95)
        except Exception as e:
            self.log(f"Error in LAS generation: {e}")
            raise

    def generate_overall_join_and_las(self):
        """Generate overall join images and LAS files from all source files"""
        self.log("7. Generating overall join images and LAS files...")
        try:
            # Create overall join directories
            overall_dirs = [
                self.normpath(self.project_dir, "img_features_join_all"),
                self.normpath(self.project_dir, "img_rgb_join_all"),
                self.normpath(self.project_dir, "img_predict_multi_class_join_all")
            ]
            for dir_path in overall_dirs:
                os.makedirs(dir_path, exist_ok=True)
            # Собираем все реальные файлы из папок
            all_tensors_files = [self.normpath(self.project_dir, "tensors", f) for f in os.listdir(self.normpath(self.project_dir, "tensors")) if f.endswith('.npy')]
            all_features_files = [self.normpath(self.project_dir, "img_features", f) for f in os.listdir(self.normpath(self.project_dir, "img_features")) if f.endswith('.png')]
            all_rgb_files = [self.normpath(self.project_dir, "img_rgb", f) for f in os.listdir(self.normpath(self.project_dir, "img_rgb")) if f.endswith('.png')]
            all_predict_files = [self.normpath(self.project_dir, "img_predict_multi_class", f) for f in os.listdir(self.normpath(self.project_dir, "img_predict_multi_class")) if f.endswith('.png')]
            # Generate overall feature images join
            overall_features_output = self.normpath(self.project_dir, "img_features_join_all", "all_join.png")
            main_not_parallel_tensor_to_image(all_tensors_files, overall_features_output,
                                              cp.feature_output_tensor, cp.channels_visualisation)
            main_join_img(all_features_files, overall_features_output)
            self.log("7a. Overall feature images join generated!")
            # Generate overall RGB images join
            overall_rgb_output = self.normpath(self.project_dir, "img_rgb_join_all", "all_join.png")
            main_not_parallel_tensor_to_image(all_tensors_files, overall_rgb_output,
                                              cp.feature_output_tensor, cp.channels_visualisation_rgb)
            main_join_img(all_rgb_files, overall_rgb_output)
            self.log("7b. Overall RGB images join generated!")
            # Generate overall prediction join
            overall_predict_output = self.normpath(self.project_dir, "img_predict_multi_class_join_all", "all_join.png")
            main_join_img(all_predict_files, overall_predict_output)
            self.log("7c. Overall prediction join generated!")
            # Generate overall LAS files (combine all source LAS files)
            las_colored_dir = self.normpath(self.project_dir, "las_colored")
            las_with_class_dir = self.normpath(self.project_dir, "las_with_class")
            # For overall LAS, we'll use the first LAS file as base and the overall prediction
            if self.las_files:
                first_las_file = self.las_files[0]
                file_las = first_las_file  # first_las_file уже содержит полный путь к файлу в проекте
                file_img_colored = overall_predict_output
                # Generate overall colored LAS
                file_las_colored = self.normpath(las_colored_dir, "all_colored.las")
                mask_to_las_with_class_nn_rgb(
                    las_file_path=file_las,
                    image_file_path=file_img_colored,
                    output_las_path=file_las_colored,
                    class_colors=ccl.class_colors
                )
                self.log("7d. Overall colored LAS generated!")
                # Generate overall LAS with classes
                file_las_with_class = self.normpath(las_with_class_dir, "all_with_class.las")
                mask_to_las_with_class_only(
                    las_file_path=file_las,
                    image_file_path=file_img_colored,
                    output_las_path=file_las_with_class,
                    class_colors=ccl.class_colors
                )
                self.log("7e. Overall LAS with classes generated!")
            self.log("7. Overall join and LAS generation completed!")
            self.set_progress(100)
        except Exception as e:
            self.log(f"Error in overall join and LAS generation: {e}")
            raise

    def run_pipeline_thread(self):
        """Run the full pipeline in a separate thread"""
        try:
            self.split_tiles()
            self.generate_features()
            self.generate_feature_images()
            self.generate_rgb_images()
            self.predict()
            self.generate_color_las()
            self.generate_overall_join_and_las() # Added this line
            self.log("✅ Pipeline finished successfully!")
            self.status_var.set("Pipeline completed successfully")
        except Exception as e:
            self.log(f"❌ Error during pipeline: {e}")
            self.status_var.set("Pipeline failed")
        finally:
            # Re-enable buttons in main thread
            self.root.after(0, self.enable_buttons)

    def enable_buttons(self):
        """Re-enable buttons after pipeline completion"""
        self.pipeline_btn.config(state=tk.NORMAL)
        self.upload_btn.config(state=tk.NORMAL)
        if self.project_dir:
            self.open_folder_btn.config(state=tk.NORMAL)

    def run_full_pipeline(self):
        """Start the full processing pipeline"""
        # Disable buttons during processing
        self.pipeline_btn.config(state=tk.DISABLED)
        self.upload_btn.config(state=tk.DISABLED)
        
        # Reset progress
        self.set_progress(0)
        self.log("Starting full processing pipeline...")
        
        # Run pipeline in separate thread to prevent UI freezing
        pipeline_thread = threading.Thread(target=self.run_pipeline_thread)
        pipeline_thread.daemon = True
        pipeline_thread.start()


def main():
    """Main function to run the application"""
    root = tk.Tk()
    app = MainApp(root)
    
    # Center the window
    root.update_idletasks()
    width = root.winfo_width()
    height = root.winfo_height()
    x = (root.winfo_screenwidth() // 2) - (width // 2)
    y = (root.winfo_screenheight() // 2) - (height // 2)
    root.geometry(f"{width}x{height}+{x}+{y}")
    
    try:
        root.mainloop()
    except KeyboardInterrupt:
        print("Application interrupted by user")
    except Exception as e:
        print(f"Application error: {e}")


if __name__ == "__main__":
    main() 