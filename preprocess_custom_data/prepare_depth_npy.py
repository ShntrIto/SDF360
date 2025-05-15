import os
from glob import glob
import numpy as np
from PIL import Image

def prepare_depth_npy(directory_path):
    image_files = sorted(glob(os.path.join(directory_path, 'image/*.png')))
    save_dir = os.path.join(directory_path, 'depths')
    
    first_image_path = image_files[0]
    with Image.open(first_image_path) as img:
        width, height = img.size
    
    # Create zero-filled ndarray for each image and save as .npy
    for idx, image_file in enumerate(image_files):
        zero_array = np.zeros((height, width), dtype=np.float32) # [H, W] で保存
        npy_filename = os.path.join(save_dir, f"{idx:03d}.npy")
        np.save(npy_filename, zero_array)

# Example usage
directory_path = '/home/jaxa/shintaro/SDF360/dataset/OmniBlender/barbershop/colmap/preprocessed'
prepare_depth_npy(directory_path)