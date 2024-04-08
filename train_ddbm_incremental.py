"""
Train a diffusion model on images.
"""

import os
from pathlib import Path
from glob import glob
import argparse
import numpy as np
import torch as th
import torch.distributed as dist
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


def create_argparser():
    defaults = dict(
        data_dir="",
        dataset='edges2handbags',
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        global_batch_size=2048,
        batch_size=-1,
        microbatch=-1,  # -1 disables microbatches
        ema_rate="0.9999",  # comma-separated list of EMA values
        log_interval=50,
        test_interval=500,
        save_interval=10000,
        save_interval_for_preemption=50000,
        resume_checkpoint="",
        exp='',
        use_fp16=False,
        fp16_scale_growth=1e-3,
        debug=False,
        num_workers=2,
        use_augment=False
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


def sample():
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
    test_batch, test_cond, _, mask = next(iter(self.test_data))
    test_batch = self.preprocess(test_batch)

    # Mask input if mask exists
    if mask is not None or mask > 0:
        test_batch = test_batch*mask
        _sample_batch = test_batch[0:(min(10,len(batch)//4))]

    if isinstance(test_cond, th.Tensor) and test_batch.ndim == test_cond.ndim:
        test_xT = self.preprocess(test_cond)
        if mask is not None or mask > 0:
            test_xT = test_xT*mask
            _sample_cond = test_xT[0:(min(10,len(batch)//4))] 
        test_cond = {'xT': test_xT}
    else:
        test_cond['xT'] = self.preprocess(test_cond['xT'])

    # Code for performing incremental image sampling during training.
    # TODO: Make this function more general and accept model paramters during sampling 
    #       rather than the hard coded values used currently.
    #       This sould be modified if training on a dataset of different resolution.
    logger.log("Generating samples...")
    gathered = _sample_cond
    gathered = th.cat((_sample_cond,_sample_batch),0)
    sample, path, nfe = karras_sample(
        self.diffusion,
        self.model,
        _sample_cond.to(dist_util.dev()),
        _sample_batch.to(dist_util.dev()),
        steps=40,
        model_kwargs={'xT': _sample_cond.to(dist_util.dev())},
        device=dist_util.dev(),
        clip_denoised=True,
        sampler='heun',
        sigma_min=0.0001,
        sigma_max=1,
        guidance=1
    )
    sample = sample.contiguous().detach().cpu()
    sample = sample*mask[0:(min(10,len(batch)//4))]
    gathered = th.cat((gathered,sample),0)
    # Compute solution difference
    sample_difference = _sample_batch-sample
    gathered = th.cat((gathered,sample_difference),0)
    # Print ranges of tensors
    logger.log("Min and Max of tensors: ")
    logger.log(str(th.min(_sample_cond[0]))+", "+str(th.max(_sample_cond[0])))
    logger.log(str(th.min(_sample_batch[0])))
    logger.log(str(th.min(sample[0]))+", "+str(th.max(sample[0])))
    # Save the generated sample images
    logger.log("Sampled tensor shape: "+str(sample.shape))
    grid_img = torchvision.utils.make_grid(gathered, nrow=min(10,len(batch)//4), normalize=True)
    torchvision.utils.save_image(grid_img, f'tmp_imgs/{self.step}.pdf')


def calculate_f1_score():
    """
    Draws a random sample of conditional images from the test dataset and generates
    samples from these in order to compute the F1 (Dice) score of the model. This is
    used for image segementation accuracy evaluation.
    """


def main(args):
    # Profiler code 
    th.backends.cudnn.benchmark = True

    workdir = get_workdir(args.exp)
    Path(workdir).mkdir(parents=True, exist_ok=True)
    
    dist_util.setup_dist()
    logger.configure(dir=workdir)
    if dist.get_rank() == 0:
        name = args.exp if args.resume_checkpoint == "" else args.exp + '_resume'
        logger.log("creating model and diffusion...")
    
    data_image_size = args.image_size
    
    if args.resume_checkpoint == "":
        model_ckpts = list(glob(f'{workdir}/*model*[0-9].*'))
        if len(model_ckpts) > 0:
            max_ckpt = max(model_ckpts, key=lambda x: int(x.split('model_')[-1].split('.')[0]))
            if os.path.exists(max_ckpt):
                args.resume_checkpoint = max_ckpt
                if dist.get_rank() == 0:
                    logger.log('Resuming from checkpoint: ', max_ckpt)


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
        image_size=data_image_size,
        num_workers=args.num_workers,
    )
    
    if args.use_augment:
        augment = AugmentPipe(
                p=0.12,xflip=1e8, yflip=1, scale=1, rotate_frac=1, aniso=1, translate_frac=1
            )
    else:
        augment = None
        
    logger.log("training...")
    TrainLoop(
        model=model,
        diffusion=diffusion,
        train_data=data,
        test_data=test_data,
        batch_size=batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
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
    ).run_loop()


if __name__ == "__main__":
    args = create_argparser().parse_args()
    main(args)