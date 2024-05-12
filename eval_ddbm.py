"""
Train a diffusion model on images.
"""

import os
from pathlib import Path
from glob import glob
import argparse
import math
import random
import numpy as np
from PIL import Image
import torch as th
from torcheval.metrics.functional import multiclass_f1_score as F1
import torch.distributed as dist
from torch.cuda.amp import autocast

import torchvision
import torchvision.transforms as transforms
from torch.utils.data.sampler import Sampler
from datasets.image_folder import make_dataset
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from datasets.augment import AugmentPipe

from ddbm import dist_util, logger
from ddbm.resample import create_named_schedule_sampler
from ddbm.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    sample_defaults,
    args_to_dict,
    add_dict_to_argparser,
    get_workdir
)
from ddbm.train_util import TrainLoop
from ddbm.karras_diffusion import karras_sample
from ddbm.nn import mean_flat, append_dims, append_zero
from tqdm import tqdm

from diffusers.models import AutoencoderKL


def create_argparser():
    defaults = dict(
        data_dir="",
        work_dir="",
        dataset='edges2handbags',
        schedule_sampler="uniform",
        sigma_min=0.0001,
        sigma_max=1.0,
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        dice_weight=0.0,
        dice_tol=0.0,
        data_image_size=-1,
        data_image_channels=-1,
        global_batch_size=2048,
        batch_size=-1,
        microbatch=-1,  # -1 disables microbatches
        ema_rate="0.9999",  # comma-separated list of EMA values
        log_interval=50,
        test_interval=50,
        save_interval=10000,
        save_interval_for_preemption=50000,
        resume_checkpoint="",
        exp='',
        use_fp16=False,
        fp16_scale_growth=1e-3,
        debug=False,
        num_workers=2,
        use_augment=False,
        multi_gpu_sampling=False,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


def get_data_scaler(config):
  """Data normalizer. Assume data are always in [0, 1]."""
  if config.data.centered:
    # Rescale to [-1, 1]
    return lambda x: x * 2. - 1.
  else:
    return lambda x: x


def get_data_inverse_scaler(config):
  """Inverse data normalizer."""
  if config.data.centered:
    # Rescale [-1, 1] to [0, 1]
    return lambda x: (x + 1.) / 2.
  else:
    return lambda x: x


class UniformDequant(object):
  def __call__(self, x):
    return x + th.rand_like(x) / 256


class RASampler(Sampler):
    """Sampler that restricts data loading to a subset of the dataset for distributed,
    with repeated augmentation.
    It ensures that different each augmented version of a sample will be visible to a
    different process (GPU).
    Heavily based on 'th.utils.data.DistributedSampler'.
    This is borrowed from the DeiT Repo:
    https://github.com/facebookresearch/deit/blob/main/samplers.py
    """
    
    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True, seed=0, repetitions=3):
        if num_replicas is None:
            num_replicas = dist.get_world_size()
        if rank is None:
            rank = dist.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.num_samples = int(math.ceil(len(self.dataset) * float(repetitions) / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas
        self.num_selected_samples = int(math.floor(len(self.dataset) // 256 * 256 / self.num_replicas))
        self.shuffle = shuffle
        self.seed = seed
        self.repetitions = repetitions
    
    def __iter__(self):
        if self.shuffle:
            # Deterministically shuffle based on epoch
            g = th.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = th.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))
        
        # Add extra samples to make it evenly divisible
        indices = [ele for ele in indices for i in range(self.repetitions)]
        indices += indices[: (self.total_size - len(indices))]
        assert len(indices) == self.total_size
        
        # Subsample
        indices = indices[self.rank : self.total_size : self.num_replicas]
        assert len(indices) == self.num_samples
        
        return iter(indices[: self.num_selected_samples])
    
    def __len__(self):
        return self.num_selected_samples
    
    def set_epoch(self, epoch):
        self.epoch = epoch


class InfiniteBatchSampler(Sampler):
    def __init__(self, dataset_len, batch_size, seed=0, filling=False, shuffle=True, drop_last=False):
        self.dataset_len = dataset_len
        self.batch_size = batch_size
        self.iters_per_ep = dataset_len // batch_size if drop_last else (dataset_len + batch_size - 1) // batch_size
        self.max_p = self.iters_per_ep * batch_size
        self.filling = filling
        self.shuffle = shuffle
        self.epoch = 0
        self.seed = seed
        self.indices = self.gener_indices()
    
    def gener_indices(self):
        if self.shuffle:
            g = th.Generator()
            g.manual_seed(self.epoch + self.seed)
            indices = th.randperm(self.dataset_len, generator=g).numpy()
        else:
            indices = th.arange(self.dataset_len).numpy()
        
        tails = self.batch_size - (self.dataset_len % self.batch_size)
        if tails != self.batch_size and self.filling:
            tails = indices[:tails]
            np.random.shuffle(indices)
            indices = np.concatenate((indices, tails))
        
        # built-in list/tuple is faster than np.ndarray (when collating the data via a for-loop)
        # noinspection PyTypeChecker
        return tuple(indices.tolist())
    
    def __iter__(self):
        self.epoch = 0
        while True:
            self.epoch += 1
            p, q = 0, 0
            while p < self.max_p:
                q = p + self.batch_size
                yield self.indices[p:q]
                p = q
            if self.shuffle:
                self.indices = self.gener_indices()
    
    def __len__(self):
        return self.iters_per_ep


class DistInfiniteBatchSampler(InfiniteBatchSampler):
    def __init__(self, world_size, rank, dataset_len, glb_batch_size, seed=0, repeated_aug=0, filling=False, shuffle=True):
        # from torchvision.models import ResNet50_Weights
        # RA sampler: https://github.com/pyth/vision/blob/5521e9d01ee7033b9ee9d421c1ef6fb211ed3782/references/classification/sampler.py
        
        assert glb_batch_size % world_size == 0
        self.world_size, self.rank = world_size, rank
        self.dataset_len = dataset_len
        self.glb_batch_size = glb_batch_size
        self.batch_size = glb_batch_size // world_size
        
        self.iters_per_ep = (dataset_len + glb_batch_size - 1) // glb_batch_size
        self.filling = filling
        self.shuffle = shuffle
        self.repeated_aug = repeated_aug
        self.epoch = 0
        self.seed = seed
        self.indices = self.gener_indices()
    
    def gener_indices(self):
        global_max_p = self.iters_per_ep * self.glb_batch_size  # global_max_p % world_size must be 0 cuz glb_batch_size % world_size == 0
        if self.shuffle:
            g = th.Generator()
            g.manual_seed(self.epoch + self.seed)
            global_indices = th.randperm(self.dataset_len, generator=g)
            if self.repeated_aug > 1:
                global_indices = global_indices[:(self.dataset_len + self.repeated_aug - 1) // self.repeated_aug].repeat_interleave(self.repeated_aug, dim=0)[:global_max_p]
        else:
            global_indices = th.arange(self.dataset_len)
        filling = global_max_p - global_indices.shape[0]
        if filling > 0 and self.filling:
            global_indices = th.cat((global_indices, global_indices[:filling]))
        global_indices = tuple(global_indices.numpy().tolist())
        
        seps = th.linspace(0, len(global_indices), self.world_size + 1, dtype=th.int)
        local_indices = global_indices[seps[self.rank]:seps[self.rank + 1]]
        self.max_p = len(local_indices)
        return local_indices
    

def get_params(size, resize_size, crop_size, angle):
    w, h = size
    new_h = h
    new_w = w

    ss, ls = min(w, h), max(w, h)  # shortside and longside
    width_is_shorter = w == ss
    ls = int(resize_size * ls / ss)
    ss = resize_size
    new_w, new_h = (ss, ls) if width_is_shorter else (ls, ss)

    x = random.randint(0, np.maximum(0, new_w - crop_size))
    y = random.randint(0, np.maximum(0, new_h - crop_size))

    flip = random.random() > 0.5
    rotate = random.random() > 0.5
    if angle > 0:
        angle = random.randrange(0,365,angle) # step corresponds to angles of angle-deg increments

    return {'crop_pos': (x, y), 'flip': flip, 'rotate':rotate, 'angle':angle}


def get_transform(params, flip=True, totensor=True):
    transform_list = []

    if flip:
        transform_list.append(transforms.Lambda(lambda img: __flip(img, params['flip'])))
    if totensor:
        transform_list.append(transforms.ToTensor())

    return transforms.Compose(transform_list)


def get_tensor(normalize=True, toTensor=True, num_channels=3):
    transform_list = []
    if toTensor:
        transform_list += [transforms.ToTensor()]

    if normalize:
        transform_list += [normalize(num_channels=num_channels)]
    return transforms.Compose(transform_list)


def normalize(num_channels=3):
    if num_channels == 3:
        return transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    elif num_channels == 1:
        return transforms.Normalize((0.5), (0.5))
    else:
        raise NotImplementedError(f"Only num_channels == 1 or 3 currently supported.")


def __flip(img, flip):
    if flip:
        if isinstance(img, th.Tensor):
            return img.flip(-1)
        else:
            return img.transpose(Image.FLIP_LEFT_RIGHT)
    return img


def get_flip(img, flip):
    return __flip(img, flip)


class GridEdgesDataset(th.utils.data.Dataset):
    """A dataset class for paired image dataset.
    It assumes that the directory '/path/to/data/train' contains image pairs in the form of {A,B}.
    During test time, you need to prepare a directory '/path/to/data/test'.
    """

    def __init__(self, dataroot, train=True, img_size=2048, patch_size=512, num_channels=3, random_crop=False, random_flip=False, rotate=False, angle=0):
        """Initialize this dataset class.
        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        super().__init__()
        if train:
            self.train_dir = os.path.join(dataroot, 'train')  # get the image directory
            self.train_paths = make_dataset(self.train_dir) # get image paths
            self.AB_paths = sorted(self.train_paths)
        else:
            self.test_dir = os.path.join(dataroot, 'val')  # get the image directory
            self.AB_paths = make_dataset(self.test_dir) # get image paths
            
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.random_crop = random_crop
        self.random_flip = random_flip
        self.rotate = rotate
        self.angle = angle
        self.train = train
        self.mask = -1


    def __getitem__(self, index):
        """Return a data point and its metadata information.
        Parameters:
            index - - a random integer for data indexing
        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor) - - an image in the input domain
            B (tensor) - - its corresponding image in the target domain
            A_paths (str) - - image paths
            B_paths (str) - - image paths (same as A_paths)
        """
        # Read an image given a random integer index
        AB_path = self.AB_paths[index]
        if self.num_channels == 3:
            AB = Image.open(AB_path).convert('RGB')
        elif self.num_channels == 1:
            AB = Image.open(AB_path).convert('L')
        else:
            raise NotImplementedError(f"Only num_channels == 1 or 3 supported.")
        # Split AB image into A and B
        w, h = AB.size
        w2 = int(w / 2)
        A = AB.crop((0, 0, w2, h))
        B = AB.crop((w2, 0, w, h))

        # apply the same transform to both A and B
        params =  get_params(A.size, self.img_size, self.patch_size, self.angle)
        transform_image = get_transform(params, flip=self.random_flip)

        # Initialize empty tensors for batches
        B_batch = th.empty((0, 3, self.patch_size, self.patch_size), dtype=th.float32)
        A_batch = th.empty((0, 3, self.patch_size, self.patch_size), dtype=th.float32)

        # Split images into 4x4 grid of 512x512px and return batch
        for i in range(w2 // self.patch_size):
            # Calculate cropping parameters to randomly crop image
            for j in range(h // self.patch_size):
                left = i * self.patch_size
                top = j * self.patch_size
                right = left + self.patch_size
                bottom = top + self.patch_size

                # Perform crop
                B_cropped = B.crop((left, top, right, bottom))
                A_cropped = A.crop((left, top, right, bottom))

                # Transform and concatenate to batches
                B_batch = th.cat((B_batch, transform_image(B_cropped).unsqueeze(0)), dim=0)
                A_batch = th.cat((A_batch, transform_image(A_cropped).unsqueeze(0)), dim=0)

        return B_batch, A_batch, index, self.mask

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.AB_paths)


def load_data(
    data_dir,
    dataset,
    batch_size,
    image_size,
    num_channels=3,
    deterministic=False,
    include_test=False,
    seed=42,
    num_workers=1,
):
    # Compute batch size for this worker.
    root = data_dir

    if dataset == 'fives':
        valset = GridEdgesDataset(dataroot=root, train=True, img_size=image_size, num_channels=num_channels,
                                random_crop=False, random_flip=False, rotate=True, angle=90)

    num_tasks = dist.get_world_size()
    global_rank = dist.get_rank()
    sampler = th.utils.data.DistributedSampler(
            valset, num_replicas=num_tasks, rank=global_rank, shuffle=False, drop_last=False
        )
    val_loader = th.utils.data.DataLoader(
            valset, batch_size=1, sampler=sampler, num_workers=num_workers, drop_last=False
        )

    return val_loader


def preprocess(x):
    """
    Preprocessing function taken from train_util.py
    """    
    if x.shape[1] == 3:
        x =  2.*x - 1.
    return x


def training_sample(diffusion, model, vae, data, num_samples, args):
    """
    Generates a small selection of samples after pausing training of ddbm model. 
    Intended to be used for visual confirmation of training progress.
    Images are generated using conditional values from the test dataset (not from
    the training dataset).

    Samples are formatted into a grid of the form:
    +-----------+   
    |   Cond    |   These images are saved to the temp folder tmp_imgs and are    
    +-----------+   labed with the current training step number of the model.
    |   Ref     |
    +-----------+   
    |   Sample  |
    +-----------+
    |   MSE     |
    +-----------+
    """
    
    test_batch, test_cond, _, mask = next(iter(data))
    logger.log(test_batch.shape)
    test_batch = test_batch[0]
    test_cond = test_cond[0]
    logger.log(test_batch.shape)

    batch_size = len(test_batch)

    if num_samples > batch_size:
        logger.log("Requested number of training samples > batch_size.")
        num_samples = batch_size

    test_batch = test_batch[0:num_samples]
    test_cond = test_cond[0:num_samples]
    if mask[0] != -1 and mask is not None:
        mask = mask[0:num_samples]

    test_batch = test_batch.to(th.float32).to(dist_util.dev())
    test_cond = test_cond.to(th.float32).to(dist_util.dev())
    mask = mask.to(th.float32).to(dist_util.dev())

    test_batch = preprocess(test_batch)

    # Mask input if mask exists
    if mask[0] != -1 and mask is not None:
        test_batch = (test_batch*mask)

    if isinstance(test_cond, th.Tensor) and test_batch.ndim == test_cond.ndim:
        test_xT = preprocess(test_cond)
        if mask[0] != -1 and mask is not None:
            test_xT = (test_xT*mask)
        test_cond = {'xT': test_xT}
    else:
        test_cond['xT'] = preprocess(test_cond['xT'])

    with th.no_grad():
        with autocast(dtype=th.float16):
            emb_test_batch = vae.encode(test_batch).latent_dist.mode()
            emb_test_xT = vae.encode(test_cond['xT']).latent_dist.mode()
        
            logger.log("Generating samples...")
            
            gathered = th.cat((test_xT,test_batch),0).contiguous().detach().cpu()
            # normalize values of gathered to the same scale as the model output 

            emb_sample, path, nfe = karras_sample(
                diffusion,
                model,
                emb_test_xT,
                emb_test_batch,
                steps=40,
                model_kwargs={'xT': emb_test_xT},
                clip_denoised=False,
                sampler='heun',
                sigma_min=args.sigma_min,
                sigma_max=args.sigma_max,
                guidance=1
            )

            sample = vae.decode(emb_sample).sample

    dist.barrier()

    if mask[0] != -1 and mask is not None:
        sample = sample*mask

    sample = sample.contiguous().detach().cpu()
    test_batch = test_batch.contiguous().detach().cpu()

    gathered = th.cat((gathered, sample), 0)
    # Compute solution difference
    sample_difference = th.abs(test_batch-sample)
    gathered = th.cat((gathered, sample_difference), 0)
    # Save the generated sample images
    logger.log("Sampled tensor shape: "+str(sample.shape))
    grid_img = torchvision.utils.make_grid(gathered, nrow=num_samples, normalize=True, scale_each=True)
    torchvision.utils.save_image(grid_img, f'tmp_imgs/eval_ddbm_sample.pdf')


def calculate_metrics(diffusion, model, vae, data, args, num_samples=200):
    """
    Draws a random sample of conditional images from the test dataset and generates
    samples from these in order to compute the F1 (Dice) score of the model. This is
    used for image segementation accuracy evaluation.
    """

    def _sample(test_batch, test_cond, mask):
        """
        Generate a large set of sample images from model for use in computing metrics.
        """

        test_batch = test_batch.to(th.float32).to(dist_util.dev())
        test_cond = test_cond.to(th.float32).to(dist_util.dev())
        mask = mask.to(th.float32).to(dist_util.dev())

        grid_img = torchvision.utils.make_grid(test_batch, nrow=1, normalize=True, scale_each=True)
        torchvision.utils.save_image(grid_img, f'tmp_imgs/test_batch_debug.pdf')
        grid_img = torchvision.utils.make_grid(test_cond, nrow=1, normalize=True, scale_each=True)
        torchvision.utils.save_image(grid_img, f'tmp_imgs/test_cond_debug.pdf')

        test_batch = preprocess(test_batch)

        # Mask input if mask exists
        if mask[0] != -1 and mask is not None:
            test_batch = (test_batch*mask)

        if isinstance(test_cond, th.Tensor) and test_batch.ndim == test_cond.ndim:
            test_xT = preprocess(test_cond)
            # test_xT = test_cond
            if mask[0] != -1 and mask is not None:
                test_xT = (test_xT*mask)
            test_cond = {'xT': test_xT}
        else:
            test_cond['xT'] = preprocess(test_cond['xT'])

        with th.no_grad():
            with autocast(dtype=th.float16):
                emb_test_batch = vae.encode(test_batch).latent_dist.mode()
                emb_test_xT = vae.encode(test_cond['xT']).latent_dist.mode()

                emb_sample, path, nfe = karras_sample(
                    diffusion,
                    model,
                    emb_test_xT,
                    emb_test_batch,
                    steps=40,
                    model_kwargs={'xT': emb_test_xT},
                    clip_denoised=False,
                    sampler='heun',
                    sigma_min=args.sigma_min,
                    sigma_max=args.sigma_max,
                    churn_step_ratio=0.0,
                    rho=7.0,
                    guidance=1.0,
                )

                sample = vae.decode(emb_sample).sample

        if mask[0] != -1 and mask is not None:
            sample = sample*mask

        dist.barrier()

        sample = sample.contiguous().detach().cpu()
        test_xT = test_xT.contiguous().detach().cpu()
        test_batch = test_batch.contiguous().detach().cpu()
        mask = mask.contiguous().detach().cpu()

        grid_img = torchvision.utils.make_grid(sample, nrow=1, normalize=True, scale_each=True)
        torchvision.utils.save_image(grid_img, f'tmp_imgs/sample_debug.pdf')

        return sample, test_batch, mask

    def _compute_scores(gen_img, ref_img):
        """
        Function computes the f1 dice loss between a generated image (mask) and the reference image
        """
        # Ensure imputs are normalized to [0,1]
        gen_img = (gen_img+1.)/2.
        gen_img = gen_img.clamp(0,1)
        ref_img = (ref_img+1.)/2.
        ref_img = ref_img.clamp(0,1)

        # Reassemble batch of patches into image and evaluate metrics on entire image
        patch_size = 512
        complete_gen_img = th.zeros((1, 3, 2048, 2048), dtype=th.float32)
        complete_ref_img = th.zeros((1, 3, 2048, 2048), dtype=th.float32)
        for i in range(16):
            y_offset = patch_size*(i//4)
            x_offset = patch_size*(i%4)
            complete_gen_img[0, :, x_offset:(x_offset + patch_size), y_offset:(y_offset + patch_size)] = gen_img[i]
            complete_ref_img[0, :, x_offset:(x_offset + patch_size), y_offset:(y_offset + patch_size)] = ref_img[i]

        # Compute MSE
        mse = mean_flat((complete_gen_img-complete_ref_img)**2)

        # Compute DICE loss
        # Test if images are single channel and if not convert them to single channel by averaging
        if complete_ref_img.shape[1] == 3:
            complete_ref_img = complete_ref_img.mean(dim=1, keepdim=True)
            complete_gen_img = complete_gen_img.mean(dim=1, keepdim=True)
        elif complete_ref_img.shape[1] > 3:
            raise ValueError(f"Number of output channels must be either {1} or {3}.")
        
        grid_img = torchvision.utils.make_grid(complete_gen_img, nrow=1, normalize=True, scale_each=True)
        torchvision.utils.save_image(grid_img, f'tmp_imgs/gen_img_debug.pdf')
        grid_img = torchvision.utils.make_grid(complete_ref_img, nrow=1, normalize=True, scale_each=True)
        torchvision.utils.save_image(grid_img, f'tmp_imgs/ref_img_debug.pdf')
        
        _complete_gen_img = complete_gen_img
        _complete_ref_img = complete_ref_img
        complete_gen_img = th.flatten(complete_gen_img)
        complete_ref_img = th.flatten(complete_ref_img)

        dice = F1(complete_gen_img, complete_ref_img, num_classes=2)

        complete_gen_img_bin = (complete_gen_img >= 0.5).float()
        complete_ref_img_bin = (complete_ref_img >= 0.5).float()
        _complete_gen_img_bin = (_complete_gen_img >= 0.5).float()
        _complete_ref_img_bin = (_complete_ref_img >= 0.5).float() 
        dice_tol = F1(complete_gen_img_bin, complete_ref_img_bin, num_classes=2)

        grid_img = torchvision.utils.make_grid(_complete_gen_img_bin, nrow=1, normalize=True, scale_each=True)
        torchvision.utils.save_image(grid_img, f'tmp_imgs/gen_img_bin_debug.pdf')
        grid_img = torchvision.utils.make_grid(_complete_ref_img_bin, nrow=1, normalize=True, scale_each=True)
        torchvision.utils.save_image(grid_img, f'tmp_imgs/ref_img_bin_debug.pdf')
        
        # Compute accuracy 
        I = th.ones_like(_complete_ref_img)
        tp = mean_flat(_complete_gen_img_bin*_complete_ref_img_bin)
        tn = mean_flat((I-_complete_gen_img_bin)*(I-_complete_ref_img_bin))
        fp = mean_flat(_complete_gen_img_bin*(I-_complete_ref_img_bin))
        fn = mean_flat((I-_complete_gen_img_bin)*_complete_ref_img_bin)
        accuracy = (tp+tn+1e-10)/(tp+tn+fp+fn+1e-10)
        precision = (tp+1e-10)/(tp+fp+1e-10)
        recall = (tp+1e-10)/(tp+fn+1e-10)

        scores = {}
        scores["mse"] = mse
        scores["dice"] = dice
        scores["dice_tol"] = dice_tol
        scores["accuracy"] = accuracy
        scores["precision"] = precision
        scores["recall"] = recall

        return scores

    if args.batch_size == -1:
        batch_size = args.global_batch_size // dist.get_world_size()
        if args.global_batch_size % dist.get_world_size() != 0:
            logger.log(
                f"warning, using smaller global_batch_size of {dist.get_world_size()*batch_size} instead of {args.global_batch_size}"
            )
    else:
        batch_size = args.batch_size

    gathered_scores = {"mse":0, "dice":0, "dice_tol":0, "accuracy":0, "precision":0, "recall":0}
    num_inter = num_samples//batch_size

    i = 1
    for test_batch, test_cond, _, mask in tqdm(data):
        # Generate samples from model to compute f1 score against and fid
        test_batch = test_batch[0]
        test_cond = test_cond[0] 
        
        sample, target, mask = _sample(test_batch, test_cond, mask)
        scores = _compute_scores(sample, target)
        # update averages 
        gathered_scores["mse"] = ((i-1)/i)*gathered_scores["mse"] + scores["mse"]/i
        gathered_scores["dice"] = ((i-1)/i)*gathered_scores["dice"] + scores["dice"]/i
        gathered_scores["dice_tol"] = ((i-1)/i)*gathered_scores["dice_tol"] + scores["dice_tol"]/i
        gathered_scores["accuracy"] = ((i-1)/i)*gathered_scores["accuracy"] + scores["accuracy"]/i
        gathered_scores["precision"] = ((i-1)/i)*gathered_scores["precision"] + scores["precision"]/i
        gathered_scores["recall"] = ((i-1)/i)*gathered_scores["recall"] + scores["recall"]/i

        i += 1

    logger.log("mse:", gathered_scores["mse"])
    logger.log("dice:", gathered_scores["dice"])
    logger.log("dice_tol:", gathered_scores["dice_tol"])
    logger.log("accuracy:", gathered_scores["accuracy"])
    logger.log("precision:", gathered_scores["precision"])
    logger.log("recall:", gathered_scores["recall"])


def main(args):
    # Profiler code 
    th.backends.cudnn.benchmark = True

    # workdir = get_workdir(args.exp)
    workdir = os.path.join(args.work_dir, args.exp)
    Path(workdir).mkdir(parents=True, exist_ok=True)
    
    dist_util.setup_dist()
    logger.configure(dir=workdir)
    if dist.get_rank() == 0:
        name = args.exp if args.resume_checkpoint == "" else args.exp + '_resume'
        logger.log("creating model and diffusion...")

    if args.resume_checkpoint == "":
        model_ckpts = list(glob(f'{workdir}/*model*[0-9].*'))
        if len(model_ckpts) > 0:
            max_ckpt = max(model_ckpts, key=lambda x: int(x.split('model_')[-1].split('.')[0]))
            if os.path.exists(max_ckpt):
                args.resume_checkpoint = max_ckpt
                if dist.get_rank() == 0:
                    logger.log('Resuming from checkpoint: ', max_ckpt)

    vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae", use_safetensors=False).to(dist_util.dev())
    checkpoint = th.load(os.path.join(args.work_dir,"vae/model_040000.pt"), map_location=dist_util.dev())
    vae.load_state_dict(checkpoint)
    vae.eval()
    # vae.half() # TODO: Added to debug gpu memory usage.

    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(
                # dist_util.load_state_dict(
                #     resume_checkpoint, map_location=dist_util.dev()
                # ),
                th.load(args.resume_checkpoint, map_location=dist_util.dev()),
            )
    model.to(dist_util.dev())
    
    if args.batch_size == -1:
        batch_size = args.global_batch_size // dist.get_world_size()
        if args.global_batch_size % dist.get_world_size() != 0:
            logger.log(
                f"warning, using smaller global_batch_size of {dist.get_world_size()*batch_size} instead of {args.global_batch_size}"
            )
    else:
        batch_size = args.batch_size
        
    if dist.get_rank() == 0:
        logger.log("creating data loader...")


    test_data = load_data(
        data_dir=args.data_dir,
        dataset=args.dataset,
        batch_size=batch_size,
        image_size=args.data_image_size,
        num_channels=args.data_image_channels,
        num_workers=args.num_workers,
    )

    model.eval()
    training_sample(diffusion, model, vae, test_data, 4, args)
    calculate_metrics(diffusion, model, vae, test_data, args)


if __name__ == "__main__":
    args = create_argparser().parse_args()
    main(args)