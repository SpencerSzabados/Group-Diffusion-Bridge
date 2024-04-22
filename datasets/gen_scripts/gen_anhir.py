"""
    Script for loading and processing the ANHIR dataset (https://anhir.grand-challenge.org/)
    
    File mostly contains script for preprocessing lung-lesion images into lower resolution image
    patches following the procedure outlined in Group Eqivariant GANs by Dey 
    (https://arxiv.org/abs/2005.01683). Along with scripts for rocessing data into a format
    that the DDBM can utilize.
"""

import os 
import re
import random 
import numpy as np 
from PIL import Image 
import cv2
from tqdm import tqdm

    
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


def random_crop_noise_png(data_dir, processed_dir, resolution=64, aug_mul=1, scale=0.5, sigma=10):
    """
    Script for generating paired {A,B} images used in training DDBM 
    (https://github.com/kelvins/awesome-mlops?tab=readme-ov-file#data-management).

    :param resolution: integer value between 1<resolution<128 to scale images.
    :param aug_mul: integer vaue, the number of random (crop) samples to create from each image.
    :param sigma: float value, amount of blurring to do to an image for the low resolution pair {A,B}.
    """
    print("Performing random crop and data augmentation...")
    # Load image dataset from data_dir 
    # Assuming 'dataset' is a 3D array with shape (height, width, channels)
    img_files = [f for f in os.listdir(os.path.join(data_dir)) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    # mask_files = [f for f in os.listdir(os.path.join(data_dir,"masks")) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    img_dict = {}
    aug_indices = len(img_files)*aug_mul
    
    print("Indexing files...")
    for file in tqdm(img_files):
        # Generate names for files 
        tokens = re.split("_", file)
        label = int(tokens[0])
        if label not in img_dict:
            img_dict[label] = []
        img_dict[label].append(file)

    print("Processing images...")
    for key in img_dict.keys():
        label_img_files = img_dict[key]
        index = 0

        for file in tqdm(label_img_files):
            # Generate names for files 
            image = Image.open(os.path.join(data_dir,file))

            for _ in range(0,aug_mul):
                index += 1

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

                # Generate low resolution (blurry) image pair
                blur_resolution = int(resolution*scale)
                blur_image = img_cropped.copy()

                # add noise to image before scaling 
                if sigma > 0:
                    blur_image = (np.array(blur_image).astype(np.float32)/127.5)-1.
                    noise = np.random.normal(0, sigma, blur_image.shape)
                    blur_image = ((np.clip(blur_image+noise, a_min=-1., a_max=1.)+1)*127.5).astype(np.uint8)
                    blur_image = Image.fromarray(blur_image)

                blur_image = blur_image.resize((blur_resolution,blur_resolution), Image.LANCZOS)
                blur_image = blur_image.resize((resolution,resolution), Image.LANCZOS)

                # Glue images together 
                combined_image = Image.new("RGB", (2*resolution,resolution))
                combined_image.paste(img_cropped, (resolution,0))
                combined_image.paste(blur_image, (0,0))

                combined_image.save(os.path.join(processed_dir, f"{key}_{index}.png"), format='PNG')
    print("Finished.")


def right_crop_png(data_dir, processed_dir, resolution=64):
    # Load image dataset from data_dir 
    # Assuming 'dataset' is a 3D array with shape (height, width, channels)
    files = [f for f in os.listdir(data_dir) if f.lower().endswith(('.jpg', '.jpeg','.png'))]
    num_images = len(files)
    print(str(num_images)+", "+str(np.array(Image.open(data_dir+files[0])).shape))

    for file in files:
        image = Image.open(data_dir+file)
        cropped_img = image.crop((resolution, 0, 2*resolution, resolution))
        # Save the result to the output directory
        cropped_img.save(os.path.join(processed_dir, f"{file}"), format='PNG')


def main():
    # data paramters
    data_dir = "/home/sszabados/datasets/anhir128/test_images/"
    temp_dir = "/home/sszabados/datasets/anhir/temp/"
    processed_dir = "/home/sszabados/datasets/anhir64_random_crop_ddbm/test/"
    fid_dir = "/home/sszabados/datasets/anhir64_random_crop_ddbm/fid_images/"

    random_crop_noise_png(data_dir, processed_dir, aug_mul=4, scale=0.25, sigma=0)
    # right_crop_png(data_dir, fid_dir)

if __name__ == "__main__":
    main()