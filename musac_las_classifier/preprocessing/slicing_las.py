import os
import subprocess
from multiprocessing import Pool

def process_file_cut_tiles(filename, input_directory, output_directory, tile_size=64):
    """
    Cuts one LAS file into tiles using lastile.

    :param filename: Name of the LAS file to process
    :param input_directory: Directory with source files
    :param output_directory: Directory to save cut files
    :param tile_size: Tile size (in meters)
    """
    input_file = os.path.join(input_directory, filename)
    name, _ = os.path.splitext(filename)
    output_subdir = os.path.join(output_directory, name)
    # Create subdirectory for current file if it doesn't exist
    os.makedirs(output_subdir, exist_ok=True)

    # Form lastile command
    command = [
        'lastile',
        '-i', input_file,        # Input file
        '-tile_size', str(tile_size),  # Tile size
        '-o', output_subdir       # Save directory
    ]

    # Execute command
    subprocess.run(command)
    print(f"[✔] {filename} successfully cut and saved to {output_subdir}")
    os.rmdir(output_subdir)

def main_parallel_cut_tiles(input_directory, output_directory, tile_size=64):
    """
    Parallel cutting of all LAS files in directory.

    :param input_directory: Directory with source LAS files
    :param output_directory: Directory to save cut files
    :param tile_size: Tile size (in meters)
    :param num_processes: Number of processes for parallel processing
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_directory, exist_ok=True)

    # Get list of .las files
    filenames = [f for f in os.listdir(input_directory) if f.endswith('.las')]

    num_processes = min(os.cpu_count(), len(filenames))
    with Pool(processes=num_processes) as pool:
        pool.starmap(process_file_cut_tiles, [(filename, input_directory, output_directory, tile_size) for filename in filenames])
                                               

if __name__ == "__main__":
    import config_preprocessing as cp
    import time
    input_directory = cp.path_las_before_cut  # Path to directory with LAS files
    output_directory = cp.path_las_after_cut  # Path to directory with LAS files
    # Create output directory if it doesn't exist
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    start = time.time()
    # Run cutting in 4 threads
    main_parallel_cut_tiles(input_directory, output_directory, tile_size=cp.las_cut_size)
    end = time.time()
    print(round((end-start)/60,1))