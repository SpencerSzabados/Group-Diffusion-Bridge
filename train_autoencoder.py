"""
    Script for training or fine-tuning Stable-Diffusion autoencoder model on
    images.

    All code for autoencoder, distributions, util, autoencoder_modeules is 
    originally from (https://github.com/CompVis/stable-diffusion).
"""


import os
from pathlib import Path
from glob import glob
import argparse
import numpy as np
import torch as th
import torch.distributed as dist
import torchvision

from datasets import load_data
from datasets.augment import AugmentPipe

from ddbm import dist_util, logger
from ddbm.resample import create_named_schedule_sampler
from ddbm.script_util import (
    get_workdir
)

from ddbm.autoencoder import AutoencoderKL

from tqdm import tqdm


def create_argparser():
    defaults = dict(
        data_dir="",
        dataset='edges2handbags',
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        global_batch_size=2048,
        batch_size=-1,
        microbatch=-1,      # -1 disables microbatches
        ema_rate="0.9999",  # comma-separated list of EMA values
        log_interval=50,
        test_interval=50,
        save_interval=10000,
        resume_checkpoint="",
        exp='',
        use_fp16=False,
        fp16_scale_growth=1e-3,
        debug=False,
        num_workers=2,
        use_augment=False
    )
    parser = argparse.ArgumentParser()
    return parser


def main(args):
    # Profiler code 
    th.backends.cudnn.benchmark = True

    workdir = get_workdir(args.exp)
    Path(workdir).mkdir(parents=True, exist_ok=True)
    
    dist_util.setup_dist()
    logger.configure(dir=workdir)

    if dist.get_rank() == 0:
        name = args.exp if args.resume_checkpoint == "" else args.exp + '_resume'
        logger.log("creating autoencoder model...")

    vae = AutoencoderKL(ddconfig,
                 lossconfig,
                 embed_dim,
                 ckpt_path=None,
                 ignore_keys=[],
                 image_key="image",
                 colorize_nlabels=None,
                 monitor=None,)


    


if __name__=="__main__":
    args = create_argparser().parse_args()
    main(args)