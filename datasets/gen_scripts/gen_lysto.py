"""
    File contains scripts for generating LYSTO dataset (https://zenodo.org/records/3513571) 
    which contains various (native resolution 299x299 px) images of human cancer cells.
"""

import os
import re
import h5py
import random
import numpy as np
import torch as th
import torchvision
from torchvision import transforms
from PIL import Image
from tqdm import tqdm


# Data paramters 
h5_data_dir = "/home/datasets/lysto/"
h5_dataset_name = "test.h5"
label_dir = "/home/datasets/lysto/"
data_dir = "/home/datasets/lysto64_random_crop_ddbm/val/"
labels_name = "training_labels.csv"
npy_dataset_name = "test_images.npy" 
npy_labels_name = "test_labels.npy"
npy_organ_name = "test_organs.npy"

 
def convert_h5_npy():
    """
    Open lysto h5 file archive and convert it to a npy dataset file in native resolution.
    """
    # Open the HDF5 file
    h5_dataset = h5py.File(h5_data_dir+h5_dataset_name, 'r')
    images = h5_dataset['x']
    labels = h5_dataset['y']
    organs = h5_dataset['organ']
    npy_dataset = np.array(h5_dataset)
    npy_images = np.array(images)
    npy_labels = np.array(labels)
    # Preprocess organ labels to remove sub-identifiers and convert strings to number
    # labels as follow:
    # [colon, prostate, breast] --> [0,1,2]
    npy_orangs = np.array(organs)
    num_npy_organs = []
    for i in range(len(npy_orangs)):
        # Regex pattern splits on substrings "'" and "_"
        label = re.split("_", npy_orangs[i].decode("utf-8"))[0]
        # label = re.split("_", npy_orangs[i])[0]
        if label == "colon":
            num_npy_organs.append(0)
        elif label == "prostate":
            num_npy_organs.append(1)
        elif label == "breast":
            num_npy_organs.append(2)
        else:
            RuntimeWarning(f'Encountered unknown LYSTO label.')
    # Save data
    np.save(data_dir+npy_dataset_name, npy_images)
    np.save(data_dir+npy_labels_name, npy_labels)
    np.save(data_dir+npy_organ_name, num_npy_organs)
    

def gen_lysto_npy(resolution=299):
    """
    Open lysto267.npy dataset and downscale images to 64x64px from 267x267px.
    """
    # Load the .npy dataset
    # Assuming 'dataset' is a 4D array with shape (num_images, height, width, channels)
    org_dataset = np.load(str(h5_data_dir)+str(npy_dataset_name))
    org_labels = np.load(str(h5_data_dir)+str(npy_organ_name))
    # Create an empty array for downscaled images
    scaled_dataset = np.empty((org_dataset.shape[0], resolution, resolution, org_dataset.shape[3]), dtype=org_dataset.dtype)

    for i in tqdm(range(org_dataset.shape[0])):
        image = Image.fromarray(org_dataset[i])
        scaled_image = image.resize((resolution, resolution), Image.LANCZOS)
        scaled_dataset[i] = np.array(scaled_image)

    np.save(data_dir+npy_dataset_name, scaled_dataset)  
    np.save(data_dir+npy_organ_name, org_labels)

def gen_lysto_center_crop_npy(resolution=299):
    """
    Open lysto267.npy dataset and downscale images to 64x64px from 267x267px.
    """
    # Load the .npy dataset
    # Assuming 'dataset' is a 4D array with shape (num_images, height, width, channels)
    org_dataset = np.load(str(data_dir)+str(npy_dataset_name))
    # Create an empty array for downscaled images
    scaled_dataset = np.empty((org_dataset.shape[0], resolution, resolution, org_dataset.shape[3]), dtype=org_dataset.dtype)

    for i in range(org_dataset.shape[0]):
        image = Image.fromarray(org_dataset[i])

        # Calculate cropping parameters to center crop
        width, height = image.size
        left = (width - resolution) / 2
        top = (height - resolution) / 2
        right = (width + resolution) / 2
        bottom = (height + resolution) / 2

        # Perform center crop
        img_cropped = image.crop((left, top, right, bottom))

        scaled_dataset[i] = np.array(img_cropped)

    np.save(data_dir+npy_dataset_name, scaled_dataset)  


def gen_lysto_random_crop_npy(resolution=299, aug_mul=1):
    # Ensure the output directory exists
    os.makedirs(data_dir, exist_ok=True)

    # Load the .npy dataset
    # Assuming 'dataset' is a 4D array with shape (num_images, height, width, channels)
    org_dataset = np.load(data_dir+npy_dataset_name)
    org_labels = np.load(label_dir+npy_labels_name)
    # Create an empty array for downscaled images
    scaled_dataset = np.empty((org_dataset.shape[0]*aug_mul, resolution, resolution, org_dataset.shape[3]), dtype=org_dataset.dtype)
    scaled_labels = np.repeat(org_labels, aug_mul)
    for i in range(org_dataset.shape[0]):
        image = Image.fromarray(org_dataset[i])
        for _ in range(0,aug_mul):
            # Calculate cropping parameters to randomly crop image
            width, height = image.size
            top_x = width-resolution
            top_y = height-resolution
            left = np.random.randint(0, top_x)
            top = np.random.randint(0, top_y)
            right = left+resolution
            bottom = top+resolution

            # Perform center crop
            img_cropped = image.crop((left, top, right, bottom))

            # Resize the cropped image to the target size
            # img_resized = img_cropped.resize((resolution, resolution), Image.LANCZOS)
            scaled_dataset[i] = np.array(img_cropped)

    np.save(data_dir+npy_labels_name, scaled_labels)
    np.save(data_dir+npy_dataset_name, scaled_dataset) 


def gen_ddbm_lysto_random_crop_npy(resolution=64, aug_mul=1, scale=0.5, sigma=10):
    """
    Script for generating paired {A,B} images used in training DDBM 
    (https://github.com/kelvins/awesome-mlops?tab=readme-ov-file#data-management),
    for the LYSTO dataset (https://zenodo.org/records/3513571).

    :param resolution: integer value between 1<resolution<299 to scale images.
    :param aug_mul: integer vaue, the number of random (crop) samples to create from each image.
    :param sigma: float value, amount of blurring to do to an image for the low resolution pair {A,B}.
    """
    # Ensure the output directory exists
    os.makedirs(data_dir, exist_ok=True)

    # Load the .npy dataset
    # Assuming 'dataset' is a 4D array with shape (num_images, height, width, channels)
    org_dataset = np.load(str(data_dir)+str(npy_dataset_name))
    org_labels = np.load(str(data_dir)+str(npy_organ_name))
    # Create an empty array for downscaled images
    scaled_dataset = np.empty((org_dataset.shape[0]*aug_mul, resolution, 2*resolution, org_dataset.shape[3]), dtype=org_dataset.dtype)
    scaled_labels = np.repeat(org_labels, aug_mul)
    for i in tqdm(range(org_dataset.shape[0])):
        for _ in range(0,aug_mul):
            image = Image.fromarray(org_dataset[i])
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
                noise = np.random.normal(0,sigma,blur_image.shape)
                blur_image = ((np.clip(blur_image+noise, a_min=0, a_max=1)+1)*255).round().dtype(np.uint8)
                blur_image = Image.fromarray(blur_image)
            blur_image = blur_image.resize((blur_resolution,blur_resolution), Image.LANCZOS)
            blur_image = blur_image.resize((resolution,resolution), Image.LANCZOS)

            # Glue images together 
            combined_image = Image.new("RGB", (2*resolution,resolution))
            combined_image.paste(img_cropped, (resolution,0))
            combined_image.paste(blur_image, (0,0))

            # Resize the cropped image to the target size
            # img_resized = img_cropped.resize((resolution, resolution), Image.LANCZOS)
            scaled_dataset[i] = np.array(combined_image)

    np.save(data_dir+npy_organ_name, scaled_labels)
    np.save(data_dir+npy_dataset_name, scaled_dataset) 


def convert_npy_JPG():
    # Load the .npy dataset
    # Assuming 'dataset' is a 4D array with shape (num_images, height, width, channels)
    org_dataset = np.load(str(data_dir)+str(npy_dataset_name))
    labels = np.load(str(data_dir)+str(npy_organ_name))
    # Create an empty array for downscaled images
    for i in range(org_dataset.shape[0]):
        image = Image.fromarray(org_dataset[i])
        image.save(os.path.join(data_dir+"train_images", f"{labels[i]}_{i}.JPEG"))


def convert_npy_PNG():
    # Load the .npy dataset
    # Assuming 'dataset' is a 4D array with shape (num_images, height, width, channels)
    org_dataset = np.load(str(data_dir)+str(npy_dataset_name))
    labels = np.load(str(data_dir)+str(npy_organ_name))
    # Create an empty array for downscaled images
    for i in tqdm(range(org_dataset.shape[0])):
        image = Image.fromarray(org_dataset[i])
        image.save(os.path.join(data_dir, f"{labels[i]}_{i}.PNG"), format='PNG')


def gen_lysto_JPG(resolution=299):
    """
    Open lysto.npy dataset and downscale images to <resolution>x<resolution>px from 299x299px.
    Images are saved using the following naming convention <label_index.JPEG>
    as used for mnist data.
    """
    # Load the .npy dataset
    # Assuming 'dataset' is a 4D array with shape (num_images, height, width, channels)
    org_dataset = np.load(str(data_dir)+str(npy_dataset_name))
    labels = np.load(str(data_dir)+str(npy_organ_name))
   
    for i in range(org_dataset.shape[0]):
        image = Image.fromarray(org_dataset[i])
        scaled_image = image.resize((resolution, resolution), Image.LANCZOS)

        image.save(os.path.join(data_dir+"train_images", f"{labels[i]}_{i}.JPEG"))


def gen_lysto_center_crop_JPG(resolution=299):
    # Ensure the output directory exists
    os.makedirs(data_dir, exist_ok=True)

    # Load the .npy dataset
    # Assuming 'dataset' is a 4D array with shape (num_images, height, width, channels)
    org_dataset = np.load(str(data_dir)+str(npy_dataset_name))
    labels = np.load(str(data_dir)+str(npy_organ_name))
    # Create an empty array for downscaled images
    for i in range(org_dataset.shape[0]):
        image = Image.fromarray(org_dataset[i])

        # Calculate cropping parameters to center crop
        width, height = image.size
        left = (width - resolution) / 2
        top = (height - resolution) / 2
        right = (width + resolution) / 2
        bottom = (height + resolution) / 2

        # Perform center crop
        img_cropped = image.crop((left, top, right, bottom))

        # Resize the cropped image to the target size
        # img_resized = img_cropped.resize((resolution, resolution), Image.LANCZOS)

        # Save the result to the output directory
        img_cropped.save(os.path.join(data_dir+"train_images", f"{labels[i]}_{i}.JPEG"))


def gen_lysto_random_crop_JPG(resolution=299, aug_mul=0):
    # Ensure the output directory exists
    os.makedirs(data_dir, exist_ok=True)

    # Load the .npy dataset
    # Assuming 'dataset' is a 4D array with shape (num_images, height, width, channels)
    org_dataset = np.load(str(data_dir)+str(npy_dataset_name))
    labels = np.load(str(data_dir)+str(npy_organ_name))
    
    for i in range(org_dataset.shape[0]):
        for j in range(0,aug_mul):
            image = Image.fromarray(org_dataset[i])

            # Calculate cropping parameters to randomly crop image
            width, height = image.size
            top_x = width-resolution
            top_y = height-resolution
            left = np.random.randint(0, top_x)
            top = np.random.randint(0, top_y)
            right = left+resolution
            bottom = top+resolution

            # Perform center crop
            img_cropped = image.crop((left, top, right, bottom))

            # Resize the cropped image to the target size
            # img_resized = img_cropped.resize((resolution, resolution), Image.LANCZOS)

            # Save the result to the output directory
            img_cropped.save(os.path.join(data_dir+"train_images", f"{labels[i]}_{i+j*org_dataset.shape[0]}.JPEG"))


def gen_ddbm_lysto_right_crop(resolution=64):
    # Load image dataset from data_dir 
    # Assuming 'dataset' is a 3D array with shape (height, width, channels)
    files = [f for f in os.listdir(data_dir) if f.lower().endswith(('.jpg', '.jpeg','.png'))]
    num_images = len(files)

    for file in tqdm(files):
        image = Image.open(data_dir+file)
        cropped_img = image.crop((resolution, 0, 2*resolution, resolution))
        # Save the result to the output directory
        cropped_img.save(os.path.join(data_dir+"B", f"{file}"), format='PNG')


def gen_ddbm_lysto_left_crop(resolution=64):
    # Load image dataset from data_dir 
    # Assuming 'dataset' is a 3D array with shape (height, width, channels)
    files = [f for f in os.listdir(data_dir) if f.lower().endswith(('.jpg', '.jpeg','.png'))]
    num_images = len(files)

    for file in tqdm(files):
        image = Image.open(data_dir+file)
        cropped_img = image.crop((0, 0, resolution, resolution))
        # Save the result to the output directory
        cropped_img.save(os.path.join(data_dir+"A", f"{file}"), format='PNG')


def sample_lysto(num_samples=10, num_classes=3):
    class_images = {}
    # Load image dataset from data_dir 
    # Assuming 'dataset' is a 3D array with shape (height, width, channels)
    images = [f for f in os.listdir(data_dir+"train_images/") if f.lower().endswith(('.jpg', '.jpeg'))]
    num_images = len(images)
    resolution = Image.open(data_dir+"train_images/"+images[0]).size[0]
    print(str(num_images)+", "+str(np.array(Image.open(data_dir+"train_images/"+images[0])).shape))
    sample_images = th.zeros((num_samples*num_classes, 3, resolution, resolution))

    i = 0
    for image in images:
        if i < num_samples*100:
            label = int(image.split('_')[0])
            if label not in class_images:
                class_images[label] = []
            class_images[label].append(image)
            i += 1
        else:
            break

    transform = transforms.Compose([transforms.PILToTensor()])
    for label, images in class_images.items():
        selected_images = random.sample(images, num_samples*num_classes)
        for i in range(len(selected_images)):
            sample_images[i] = transform(Image.open(data_dir+"train_images/"+selected_images[i]))
        
    sample_images = sample_images/256

    grid_img = torchvision.utils.make_grid(sample_images, nrow=num_samples*num_classes, normalize=True)
    torchvision.utils.save_image(grid_img, f'tmp_imgs/lysto_sample_1.pdf')


if __name__=="__main__":
    convert_h5_npy()
    gen_lysto_npy(resolution=128)
    gen_ddbm_lysto_random_crop_npy(resolution=64, aug_mul=1, scale=0.25, sigma=0)
    convert_npy_PNG()
    gen_ddbm_lysto_right_crop(resolution=64)
    gen_ddbm_lysto_left_crop(resolution=64)

