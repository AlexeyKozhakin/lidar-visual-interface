import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from multiprocessing import Pool

def visual_tensor(input_dir, filename, feature_output_tensor, channels_visualisation, output_dir):
    """
    Function for visualizing and saving selected tensor channels as images.

    Arguments:
    - input_dir: directory with input data (not used in this function, but can be used for logging).
    - filename: filename for saving images.
    - data: tensor of dimension (M, M, C), where C is number of channels.
    - feature_output_tensor: dictionary containing mapping between channel names and their indices.
    - channels_visualisation: dictionary with channel names for visualization and their indices in tensor.
    - output_dir: directory for saving images.
    """
    file_path = os.path.join(input_dir, filename)
    data = np.load(file_path)  # Loading
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # First create empty image for 3 channels (RGB)
    image_data = np.zeros((data.shape[0], data.shape[1], 3), dtype=np.uint8)

    # Iterate through channels that need to be visualized
    for channel_name, channel_idx in channels_visualisation.items():
        if channel_name in feature_output_tensor:
            # Extract required channel
            channel_data = data[:, :, feature_output_tensor[channel_name]]
            print('channel feature', feature_output_tensor[channel_name])

            # Channel normalization
            channel_data_normalized = channel_data / np.max(channel_data, axis=(0, 1), keepdims=True)*255
            #channel_data_normalized = channel_data
            print(channel_data_normalized.max())
            print(channel_data_normalized.min())

            # Limit values in range from 0 to 255 and convert to integers
            channel_data_normalized = np.clip(channel_data_normalized, 0, 255).astype(np.uint8)

            # Write data to corresponding image channel (e.g., r, g, or b)
            if channel_name == channel_name:
                image_data[:, :, channels_visualisation[channel_name]] = channel_data_normalized  # Channel R

                # Save image as PNG
    name, _ = os.path.splitext(filename)            
    output_path = os.path.join(output_dir, f"{name}.png")
    img = Image.fromarray(image_data)
    img.save(output_path)

def main_parallel_tensor_to_image(input_dir, output_dir,
                                          feature_output_tensor, channels_visualisation):
    
    """
    Parallel processing of all tensor files in directory.

    :param input_dir: Directory with source tensor files
    :param output_dir: Directory to save generated images
    :param feature_output_tensor: Dictionary with feature tensor configuration
    :param channels_visualisation: Dictionary with visualization channel configuration
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Get list of .npy files
    filenames = [f for f in os.listdir(input_dir) if f.endswith('.npy')]

    num_processes = 1  # min(os.cpu_count(), len(filenames))
    print(num_processes)
    print(os.cpu_count())
    print(len(filenames))
    with Pool(processes=num_processes) as pool:
        pool.starmap(visual_tensor, [(input_dir, filename,
                                      feature_output_tensor, channels_visualisation, output_dir) for filename in filenames])
        
def main_not_parallel_tensor_to_image(input_dir, output_dir,
                                      feature_output_tensor, channels_visualisation):
    
    """
    Sequential processing of all tensor files in directory.

    :param input_dir: Directory with source tensor files
    :param output_dir: Directory to save generated images
    :param feature_output_tensor: Dictionary with feature tensor configuration
    :param channels_visualisation: Dictionary with visualization channel configuration
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Get list of .npy files
    filenames = [f for f in os.listdir(input_dir) if f.endswith('.npy')]

    for filename in filenames:
        visual_tensor(input_dir, filename, feature_output_tensor, channels_visualisation, output_dir)

if __name__ == "__main__":
    import config_preprocessing as cp
    import time
    
    start = time.time()
    # Example function call
    input_dir = cp.path_tensor_to_visual
    output_dir = cp.path_image
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Get list of .npy files
    filenames = [f for f in os.listdir(input_dir) if f.endswith('.npy')]
    print(filenames)
    for filename in filenames:
        file_path = os.path.join(input_dir, filename)
        data = np.load(file_path)  # Loading
        # Call function for visualization
        visual_tensor(input_dir, filename,
                      cp.feature_input_tensor, 
                      cp.feature_output_tensor, cp.channels_visualisation, output_dir)
    end = time.time()
    print(round((end-start)))