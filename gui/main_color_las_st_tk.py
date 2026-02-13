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
        
        self.init_ui()

    def init_ui(self):
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(4, weight=1)
        
        # Title
        title_label = ttk.Label(main_frame, text="LiDAR 3D Classification System", 
                               font=('Arial', 16, 'bold'))
        title_label.grid(row=0, column=0, columnspan=2, pady=(0, 20))
        
        # Upload button
        self.upload_btn = ttk.Button(main_frame, text="Upload LAS Files", 
                                    command=self.upload_files)
        self.upload_btn.grid(row=1, column=0, columnspan=2, pady=(0, 10), sticky=(tk.W, tk.E))
        
        # Pipeline button
        self.pipeline_btn = ttk.Button(main_frame, text="Run Full Pipeline", 
                                      command=self.run_full_pipeline)
        self.pipeline_btn.grid(row=2, column=0, columnspan=2, pady=(0, 10), sticky=(tk.W, tk.E))
        
        # Open project folder button
        self.open_folder_btn = ttk.Button(main_frame, text="Open Project Folder", 
                                         command=self.open_project_folder, state=tk.DISABLED)
        self.open_folder_btn.grid(row=3, column=0, columnspan=2, pady=(0, 10), sticky=(tk.W, tk.E))
        
        # Progress section
        progress_frame = ttk.LabelFrame(main_frame, text="Progress", padding="5")
        progress_frame.grid(row=4, column=0, columnspan=2, pady=(0, 10), sticky=(tk.W, tk.E))
        progress_frame.columnconfigure(0, weight=1)
        
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(progress_frame, variable=self.progress_var, 
                                           maximum=100, length=400)
        self.progress_bar.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=5)
        
        self.progress_label = ttk.Label(progress_frame, text="Ready")
        self.progress_label.grid(row=1, column=0, pady=(0, 5))
        
        # Log section
        log_frame = ttk.LabelFrame(main_frame, text="Processing Log", padding="5")
        log_frame.grid(row=5, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S))
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)
        
        # Create text widget with scrollbar
        text_frame = ttk.Frame(log_frame)
        text_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        text_frame.columnconfigure(0, weight=1)
        text_frame.rowconfigure(0, weight=1)
        
        self.log_text = scrolledtext.ScrolledText(text_frame, height=20, width=80, 
                                                 wrap=tk.WORD, state=tk.DISABLED)
        self.log_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var, 
                              relief=tk.SUNKEN, anchor=tk.W)
        status_bar.grid(row=6, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(10, 0))

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
            os.path.join(self.project_dir, "las"),
            os.path.join(self.project_dir, "las_cut"),
            os.path.join(self.project_dir, "tensors"),
            os.path.join(self.project_dir, "img_features"),
            os.path.join(self.project_dir, "img_features_join"),
            os.path.join(self.project_dir, "img_rgb"),
            os.path.join(self.project_dir, "img_rgb_join"),
            os.path.join(self.project_dir, "img_predict_multi_class"),
            os.path.join(self.project_dir, "img_predict_multi_class_join"),
            os.path.join(self.project_dir, "las_colored")
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
            
            uploaded_count = 0
            for file_path in files:
                try:
                    # Save files to project directory
                    dest_path = os.path.join(self.project_dir, "las", os.path.basename(file_path))
                    
                    with open(file_path, "rb") as fsrc, open(dest_path, "wb") as fdst:
                        fdst.write(fsrc.read())
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
        """Split LAS files into tiles"""
        self.log("1. Splitting LAS files into tiles...")
        try:
            las_input = os.path.join(self.project_dir, "las")
            las_cut_output = os.path.join(self.project_dir, "las_cut")
            main_not_parallel_cut_tiles(las_input, las_cut_output, tile_size=250)
            self.log("1. Splitting completed!")
            self.set_progress(15)
        except Exception as e:
            self.log(f"Error in splitting: {e}")
            raise

    def generate_features(self):
        """Generate features from LAS data"""
        self.log("2. Starting feature generation...")
        try:
            start = time.time()
            las_input = os.path.join(self.project_dir, "las_cut")
            tensors_output = os.path.join(self.project_dir, "tensors")
            main_not_parallel_transform_to_tensor(
                las_input, tensors_output,
                cp.feature_input_tensor, cp.feature_output_tensor, 
                cp.num_points_lim, cp.M_tensor_size, cp.K_nn
            )
            end = time.time()
            self.log(f"2. Feature generation completed in {round(end - start, 2)} seconds.")
            self.set_progress(35)
        except Exception as e:
            self.log(f"Error in feature generation: {e}")
            raise

    def generate_feature_images(self):
        """Generate feature images"""
        self.log("3. Generating feature images...")
        try:
            start = time.time()
            tensors_input = os.path.join(self.project_dir, "tensors")
            features_output = os.path.join(self.project_dir, "img_features")
            features_join_output = os.path.join(self.project_dir, "img_features_join")
            
            main_not_parallel_tensor_to_image(tensors_input, features_output,
                                              cp.feature_output_tensor, cp.channels_visualisation)
            main_join_img(features_output, features_join_output)
            end = time.time()
            self.log(f"3. Feature images generated in {round(end - start, 2)} seconds.")
            self.set_progress(50)
        except Exception as e:
            self.log(f"Error in feature image generation: {e}")
            raise

    def generate_rgb_images(self):
        """Generate RGB images"""
        self.log("4. Generating RGB images...")
        try:
            start = time.time()
            tensors_input = os.path.join(self.project_dir, "tensors")
            rgb_output = os.path.join(self.project_dir, "img_rgb")
            rgb_join_output = os.path.join(self.project_dir, "img_rgb_join")
            
            main_not_parallel_tensor_to_image(tensors_input, rgb_output,
                                              cp.feature_output_tensor, cp.channels_visualisation_rgb)
            main_join_img(rgb_output, rgb_join_output)
            end = time.time()
            self.log(f"4. RGB images generated in {round(end - start, 2)} seconds.")
            self.set_progress(65)
        except Exception as e:
            self.log(f"Error in RGB image generation: {e}")
            raise

    def predict(self):
        """Run prediction"""
        self.log("5. Starting prediction...")
        try:
            features_input = os.path.join(self.project_dir, "img_features")
            predict_output = os.path.join(self.project_dir, "img_predict_multi_class")
            predict_join_output = os.path.join(self.project_dir, "img_predict_multi_class_join")
            
            main_prediction(features_input, predict_output, cpred.checkpoint_path)
            main_join_img(predict_output, predict_join_output)
            self.log("5. Prediction completed!")
            self.set_progress(85)
        except Exception as e:
            self.log(f"Error in prediction: {e}")
            raise

    def generate_color_las(self):
        """Generate colored LAS file"""
        self.log("6. Generating colored LAS...")
        try:
            las_input_dir = os.path.join(self.project_dir, "las")
            predict_join_dir = os.path.join(self.project_dir, "img_predict_multi_class_join")
            las_colored_dir = os.path.join(self.project_dir, "las_colored")
            
            filenames_las = [f for f in os.listdir(las_input_dir) if f.endswith('.las')]
            if not filenames_las:
                raise Exception("No LAS files found in input directory")
            
            file_las = os.path.join(las_input_dir, filenames_las[0])
            filenames_img_colored = [f for f in os.listdir(predict_join_dir) if f.endswith('.png')]
            
            if not filenames_img_colored:
                raise Exception("No prediction images found")
            
            file_img_colored = os.path.join(predict_join_dir, filenames_img_colored[0])
            file_las_colored = os.path.join(las_colored_dir, filenames_las[0])
            
            mask_to_las_with_class_nn_rgb(
                las_file_path=file_las,
                image_file_path=file_img_colored,
                output_las_path=file_las_colored,
                class_colors=ccl.class_colors
            )
            self.log("6. Colored LAS generated!")
            self.log(f"Final result saved to: {file_las_colored}")
            self.set_progress(100)
        except Exception as e:
            self.log(f"Error in colored LAS generation: {e}")
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