"""
    Script for loading and processing the FIVES (https://www.nature.com/articles/s41597-022-01564-3)
    dataset that contains 800 2kx2k retian images from human eyes along with paired vein masks.

    Since the images are 2048x2048 with a circular image surounded by a black border, this script
    contains code for cropping the images to a square inscribed within the circular image of the
    retina. This is done to remove the portion of the images that contains no information, which
    we believed may complicate/hamper model training. From this crop random patches are created.
"""

import os 
import re
import random 
import numpy as np 
from PIL import Image 
from tqdm import tqdm

# def load_data(data_dir):
#     """
#     Load images into npy array for easier management and loading.
#     """
    

def inscribed_crop(data_dir, processed_dir):
    """
    Crop images to remove outter black portions of image outside of retina.
    Parameters for doing this where found by examination of the images by hand.
    """
    print("Cropping images to inscribed square...")
    # Load image dataset from data_dir 
    # Assuming 'dataset' is a 3D array with shape (height, width, channels)
    img_files = [f for f in os.listdir(os.path.join(data_dir,"images")) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    mask_files = [f for f in os.listdir(os.path.join(data_dir,"masks")) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    num_images = len(img_files)

    for file in tqdm(img_files):
        img = Image.open(os.path.join(data_dir+"images/",file))
        cropped_img = img.crop((326,326,1720,1720))
        cropped_img.save(os.path.join(os.path.join(processed_dir,"images"),file))
    for file in tqdm(mask_files):
        mask = Image.open(os.path.join(data_dir+"masks/",file))
        cropped_mask = mask.crop((326,326,1720,1720))
        cropped_mask.save(os.path.join(os.path.join(processed_dir,"masks"),file))

    print("Finished cropping images.")


def random_crop(data_dir, processed_dir, resolution=64, aug_mul=1):
    print("Performing random crop and data augmentation...")
    # Load image dataset from data_dir 
    # Assuming 'dataset' is a 3D array with shape (height, width, channels)
    img_files = [f for f in os.listdir(os.path.join(data_dir,"images")) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    # mask_files = [f for f in os.listdir(os.path.join(data_dir,"masks")) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    num_images = len(img_files)

    for file in tqdm(img_files):
        # Generate names for files 
        tokens = re.split("_", file)
        index = int(tokens[0])
        label = tokens[1]

        image = Image.open(data_dir+"images/"+file)
        mask = Image.open(data_dir+"masks/"+file)

        for j in range(0,aug_mul):
            # Calculate cropping parameters to randomly crop image
            width, height = image.size
            top_x = width-resolution
            top_y = height-resolution
            left = np.random.randint(0, top_x)
            top = np.random.randint(0, top_y)
            right = left+resolution
            bottom = top+resolution

            # Perform crop
            img_cropped = image.crop((left, top, right, bottom))
            mask_cropped = mask.crop((left, top, right, bottom))

            # Glue images together 
            combined_image_L = Image.new("RGB", (2*resolution,resolution))
            combined_image_L.paste(img_cropped, (resolution,0))
            combined_image_L.paste(mask_cropped, (0,0))
            combined_image_R = Image.new("RGB", (2*resolution,resolution))
            combined_image_R.paste(mask_cropped, (resolution,0))
            combined_image_R.paste(img_cropped, (0,0))

            # Save the result to the output directory
            file_name = str(index+j)+"_"+label
            combined_image_L.save(os.path.join(processed_dir+"_retina/"+"images", f"{file_name}"))
            combined_image_R.save(os.path.join(processed_dir+"_masks/"+"images", f"{file_name}"))
    print("Finished.")


def main():
    # data paramters
    data_dir = "/home/sszabados/datasets/fives/train/"
    temp_dir = data_dir+"temp/"
    processed_dir = "/home/sszabados/datasets/fives32_random_crop_ddbm"

    # load_data(data_dir, processed_dir)
    # inscribed_crop(data_dir, temp_dir)
    random_crop(temp_dir, processed_dir, resolution=64, aug_mul=10)


if __name__ == "__main__":
    main()