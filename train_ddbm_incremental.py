"""
Train a diffusion model on images.
"""

import os
from pathlib import Path
from glob import glob
import argparse
import numpy as np
import torch as th
from torcheval.metrics.functional import multiclass_f1_score as F1
import torch.distributed as dist
import torchvision
from torch.cuda.amp import autocast
from ddbm import dist_util, logger
from datasets import load_data
from datasets.augment import AugmentPipe
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


def preprocess(x):
    """
    Preprocessing function taken from train_util.py
    """    
    if x.shape[1] == 3:
        x =  2.*x - 1.
    return x


def training_sample(diffusion, model, vae, data, num_samples, step, args):
    """
    Generates a small selection of samples after pausing training of ddbm model. 
    Intended to be used for visual confirmation of training progress.
    Images are generated using conditional values from the test dataset (not from
    the training dataset).

    Samples are formatted into a grid of the form:
    +-----------+   
    |   Cond    |   These images are saved to the temp folder tmp_imgs and are    
    +-----------+   labed with the current training step number of the model.
    |   Refer   |
    +-----------+   
    |   Sample  |
    +-----------+
    |   MSE     |
    +-----------+
    """    
    test_batch, test_cond, _, mask = next(iter(data))
    batch_size = len(test_batch)

    if num_samples > batch_size:
        logger.log("Requested number of training samples > batch_size.")
        num_samples = batch_size

    test_batch = test_batch[0:num_samples]
    test_cond = test_cond[0:num_samples]
    if mask[0] != -1 and mask is not None:
        mask = mask[0:num_samples]

    test_batch = test_batch.to(dist_util.dev())
    test_cond = test_cond.to(dist_util.dev())
    mask = mask.to(dist_util.dev())

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
        # Pass data into encoder
        with autocast(dtype=th.float32):
            emb_test_batch = vae.encode(test_batch).latent_dist.mode()
            emb_test_xT = vae.encode(test_cond['xT']).latent_dist.mode()

        logger.log("Generating samples...")
        
        gathered = th.cat((test_xT,test_batch),0).contiguous().detach().cpu()
        # normalize values of gathered to the same scale as the model output 

        with autocast(dtype=th.float16):
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

        with autocast(dtype=th.float32):
            sample = vae.decode(emb_sample).sample

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
    torchvision.utils.save_image(grid_img, f'tmp_imgs/{step}.pdf')


def calculate_metrics(diffusion, model, vae, data, step, args, num_samples=1000):
    """
    Draws a random sample of conditional images from the test dataset and generates
    samples from these in order to compute the F1 (Dice) score of the model. This is
    used for image segementation accuracy evaluation.
    """

    def _sample(test_batch, test_cond, mask):
        """
        Generate a large set of sample images from model for use in computing metrics.
        """
        test_batch = test_batch.to(dist_util.dev())
        test_cond = test_cond.to(dist_util.dev())
        mask = mask.to(dist_util.dev())

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
            # test_cond['xT'] = test_cond['xT']

        with th.no_grad():
        # Pass data into encoder
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
                    guidance=1
                )

                sample = vae.decode(emb_sample).sample

        if mask[0] != -1 and mask is not None:
            sample = sample*mask

        sample = sample.contiguous().detach().cpu()
        test_xT = test_xT.contiguous().detach().cpu()
        mask = mask.contiguous().detach().cpu()

        return sample, test_xT, mask

    def _compute_scores(gen_img, ref_img):
        """
        Function computes the f1 dice loss between a generated image (mask) and the reference image
        """
        # Ensure imputs are normalized to [0,1]
        gen_img = (gen_img+1.)/2.
        gen_img = gen_img.clamp(0,1)
        ref_img = (ref_img+1.)/2.
        ref_img = ref_img.clamp(0,1)

        # Compute MSE
        mse_loss = mean_flat((gen_img-ref_img)**2)

        # Compute DICE loss
        # Test if images are single channel and if not convert them to single channel by averaging
        if gen_img.shape[1] == 3:
            ref_img = ref_img.mean(dim=1, keepdim=True)
            gen_img = gen_img.mean(dim=1, keepdim=True)
        elif gen_img.shape[1] > 3:
            raise ValueError(f"Number of output channels must be either {1} or {3}.")
        

        gen_img = th.flatten(gen_img)
        ref_img = th.flatten(ref_img)
        dice = F1(gen_img, ref_img, num_classes=2)
        gen_img_bin = (gen_img >= 0.5).float()
        ref_img_bin = (ref_img >= 0.5).float()

        dice_tol = F1(gen_img_bin, ref_img_bin, num_classes=2)
        
        # Compute accuracy 
        I = th.ones_like(ref_img)
        tp = mean_flat(gen_img_bin*ref_img_bin)
        tn = mean_flat((I-gen_img_bin)*(I-ref_img_bin))
        fp = mean_flat(gen_img_bin*(I-ref_img_bin))
        fn = mean_flat((I-gen_img_bin)*ref_img_bin)
        accuracy = (tp+tn+1e-10)/(tp+tn+fp+fn+1e-10)
        precision = (tp+1e-10)/(tp+fp+1e-10)
        recall = (tp+1e-10)/(tp+fn+1e-10)

        scores = {}
        scores["mse"] = mse_loss
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

    i = 1
    for test_batch, test_cond, _, mask in data:  
        # Generate samples from model to compute f1 score against and fid
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

    logger.log("Current training step:", step)
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
    model.to(dist_util.dev())

    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)
    
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

    data, test_data = load_data(
        data_dir=args.data_dir,
        dataset=args.dataset,
        batch_size=batch_size,
        image_size=args.data_image_size,
        num_channels=args.data_image_channels,
        num_workers=args.num_workers,
    )
    
    if args.use_augment:
        augment = AugmentPipe(
                p=0.12,xflip=1e8, yflip=1, scale=1, rotate_frac=1, aniso=1, translate_frac=1
            )
    else:
        augment = None
        
    logger.log(args)

    logger.log("training...")
    trainloop = TrainLoop(
        vae=vae,
        model=model,
        diffusion=diffusion,
        train_data=data,
        test_data=test_data,
        batch_size=batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        dice_weight=args.dice_weight,
        dice_tol=args.dice_tol,
        log_interval=args.log_interval,
        test_interval=args.test_interval,
        save_interval=args.save_interval,
        save_interval_for_preemption=args.save_interval_for_preemption,
        resume_checkpoint=args.resume_checkpoint,
        workdir=workdir,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
        augment_pipe=augment,
        **sample_defaults()
    )


    print('trainloop initialized')
    # Train model incrementally 
    while True:
        if dist.get_world_size() == 1 and not args.multi_gpu_sampling:
            logger.log("Training without split gpu sampling...")
            model.train()
            step, ema_rate = trainloop.run_loop()
            # Compute model metrics
            th.cuda.empty_cache()
            model.eval()
            training_sample(diffusion, model, vae, test_data, 2, step, args)
            calculate_metrics(diffusion, model, vae, test_data, step, args)
            trainloop.step += 1
        else:
            logger.log("Incremental smapling is not supported on fewer than 2 gpus. Proceeding without incremental sampling.")
            trainloop.run_loop()


if __name__ == "__main__":
    args = create_argparser().parse_args()
    main(args)