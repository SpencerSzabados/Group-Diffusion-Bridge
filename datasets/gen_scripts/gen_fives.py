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
import cv2
from tqdm import tqdm

    
def convert_grey_scale(data_dir, processed_dir):
    """
    Preprocess data and convert it to grey scale images using weighted average of colour channels.
        Y = 0.3*R + 0.59*G + 0.11*B
    This is the default formula used within the PIL image conversion code.
    """
    print("Converting images to grey scale...")
    # Load image dataset from data_dir 
    # Assuming 'dataset' is a 3D array with shape (height, width, channels)
    img_files = [f for f in os.listdir(os.path.join(data_dir,"images")) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    for file in tqdm(img_files):
        img_grey = Image.open(data_dir+"images/"+file).convert('L')
        img_grey.save(os.path.join(processed_dir+"images/", f"{file}"))


def CLAHE(data_dir, processed_dir, threshold=0, grid_size=8):
    """
    Apply constrast limited adaptive histogram equilization to images.
    """
    print("Contrast normalizing images...")
    # Load image dataset from data_dir 
    # Assuming 'dataset' is a 3D array with shape (height, width, channels)
    img_files = [f for f in os.listdir(os.path.join(data_dir,"images")) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    tranform = cv2.createCLAHE(clipLimit=threshold, tileGridSize=(grid_size, grid_size))

    for file in tqdm(img_files):
        img_grey = cv2.imread(os.path.join(os.path.join(data_dir,"images"), file), cv2.COLOR_BGR2GRAY)
        img_grey = tranform.apply(img_grey)
        cv2.imwrite(os.path.join(processed_dir+"images/", f"{file}"), img_grey)


def gamma_correction(data_dir, processed_dir, gamma=1):
    """
    Apply gamma correction to batch of images
    """
    print("Gamma correcting images...")
    # Load image dataset from data_dir 
    # Assuming 'dataset' is a 3D array with shape (height, width, channels)
    img_files = [f for f in os.listdir(os.path.join(data_dir,"images")) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    for file in tqdm(img_files):
        img_grey = cv2.imread(os.path.join(os.path.join(data_dir,"images"), file))
        img_grey = (img_grey/255.).astype(np.float32)
        img_grey = np.power(img_grey, 1./gamma)
        img_grey = (img_grey*255).astype(np.uint8)
        cv2.imwrite(os.path.join(processed_dir+"images/", f"{file}"), img_grey)


def scale_data(data_dir, processed_dir, resolution=64):
    """
    Scale images to selected resolution.
    """
    print("Scaling images to set resolution...")
    # Load image dataset from data_dir 
    # Assuming 'dataset' is a 3D array with shape (height, width, channels)
    img_files = [f for f in os.listdir(os.path.join(data_dir,"images")) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    mask_files = [f for f in os.listdir(os.path.join(data_dir,"masks")) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    num_images = len(img_files)

    for file in tqdm(img_files):
        # Generate names for files 
        tokens = re.split("_", file)
        index = int(tokens[0])
        label = tokens[1]
        # Load images
        image = Image.open(data_dir+"images/"+file)
        mask = Image.open(data_dir+"masks/"+file)
        # Scale images 
        image = image.resize((resolution, resolution), Image.LANCZOS)
        mask = mask.resize((resolution, resolution), Image.LANCZOS)
        # Glue images together 
        combined_image = Image.new("RGB", (2*resolution,resolution))
        combined_image.paste(image, (0,0))
        combined_image.paste(mask, (resolution,0))

        # Save the result to the output directory
        file_name = str(index)+"_"+label
        combined_image.save(os.path.join(processed_dir, f"{file_name}"))
    print("Finished.")


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


def grid_crop(data_dir, processed_dir, resolution=64):
    """
    Script slices image into set of patches of the specifed resolution. The resulting patches 
    are stiched to their corresponding patch from the image mask.
    """
    print("Slicing images into grids...")
    img_files = [f for f in os.listdir(os.path.join(data_dir,"images")) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    num_images = len(img_files)
    w, h = Image.open(data_dir+"images/"+img_files[0]).size
    assert (w%resolution == 0) and (h%resolution == 0)

    for file in tqdm(img_files):
        # Generate names for files 
        tokens = re.split("_", file)
        index = int(tokens[0])
        label = tokens[1]

        image = Image.open(data_dir+"images/"+file)
        mask = Image.open(data_dir+"masks/"+file)

        w, h = image.size
        for i in range(w//resolution):
            # Calculate cropping parameters to randomly crop image
            for j in range(h//resolution):
                left = i*resolution
                top = j*resolution
                right = left+resolution
                bottom = top+resolution

                # Perform crop
                img_cropped = image.crop((left, top, right, bottom))
                mask_cropped = mask.crop((left, top, right, bottom))

                # Glue images together 
                combined_image = Image.new("L", (2*resolution,resolution))
                combined_image.paste(mask_cropped, (resolution,0))
                combined_image.paste(img_cropped, (0,0))
                
                # Save the result to the output directory
                file_name = str(index)+"_"+str(i)+str(j)+"_"+label
                combined_image.save(os.path.join(processed_dir, f"{file_name}"))
    print("Finished.")


def main():
    # data paramters
    data_dir = "/home/sszabados/datasets/fives/test/"
    temp_dir = "/home/sszabados/datasets/fives/temp/"
    processed_dir = "/home/sszabados/datasets/fives_patches128/test/"

    # convert_grey_scale(data_dir, temp_dir)
    # CLAHE(temp_dir, temp_dir, threshold=1.8, grid_size=8)
    # gamma_correction(temp_dir, processed_dir, gamma=1.2)
    # scale_data(data_dir, processed_dir, resolution=64)
    grid_crop(data_dir, processed_dir, resolution=128)

if __name__ == "__main__":
    main()