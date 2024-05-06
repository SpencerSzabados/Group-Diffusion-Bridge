"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""

import argparse
import os

import numpy as np
import torch as th
import torchvision.utils as vutils
import torch.distributed as dist

from ddbm import dist_util, logger
from ddbm.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)
from ddbm.random_util import get_generator
from ddbm.karras_diffusion import karras_sample, forward_sample

from diffusers.models import AutoencoderKL

from datasets import load_data
from pathlib import Path
from PIL import Image


def create_argparser():
    defaults = dict(
        data_dir="",
        work_dir="",
        dataset='edges2handbags',
        schedule_sampler="uniform",
        sampler='heun',
        model_path="",
        num_samples=4,
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        dice_weight=0.0,
        dice_tol=0.0,
        churn_step_ratio=0, 
        steps=40, 
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
        guidance=1,
        seed=42,
        rho=7.0,
        ts="",
        split='train',
        clip_denoised=True,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser

def get_workdir(exp):
    workdir = f'./workdir/{exp}'
    return workdir

def main():
    args = create_argparser().parse_args()

    # workdir = get_workdir(args.exp)
    workdir = os.path.join(args.work_dir, args.exp)
    Path(workdir).mkdir(parents=True, exist_ok=True)

    ## assume ema ckpt format: ema_{rate}_{steps}.pt
    split = args.model_path.split("_")
    step = int(split[-1].split(".")[0])
    sample_dir = Path(f'/u6/sszabado/models/Group-Diffusion-Bridge/tmp_images/sample_{step}/w={args.guidance}_churn={args.churn_step_ratio}')

    dist_util.setup_dist()
    if dist.get_rank() == 0:
        sample_dir.mkdir(parents=True, exist_ok=True)
    logger.configure(dir=workdir)

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys()),
    )
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model = model.to(dist_util.dev())

    vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae", use_safetensors=False).to(dist_util.dev())
    checkpoint = th.load(os.path.join(args.work_dir,"vae/model_040000.pt"), map_location=dist_util.dev())
    vae.load_state_dict(checkpoint)
    vae.eval()
    
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()

    logger.log("sampling...")
    

    all_images = []
    
    all_dataloaders = load_data(
        data_dir=args.data_dir,
        dataset=args.dataset,
        batch_size=args.batch_size,
        image_size=args.data_image_size,
        num_channels=args.data_image_channels,
        include_test=True,
        seed=args.seed,
        num_workers=args.num_workers,
    )

    if args.split =='train':
        dataloader = all_dataloaders[1]
    elif args.split == 'test':
        dataloader = all_dataloaders[2]
    else:
        raise NotImplementedError
    
    if args.num_samples == -1:
        args.num_samples = len(dataloader.dataset)

    with th.no_grad():
        for i, data in enumerate(dataloader):
            if len(all_images) > args.num_samples:
                break

            x0_image = data[0]
            x0 = x0_image.to(dist_util.dev()) * 2 -1
            
            y0_image = (data[1]*2-1)
            y0 = vae.encode(y0_image.to(dist_util.dev())).latent_dist.mode()

            model_kwargs = {'xT': y0}
            index = data[2].to(dist_util.dev())
                
            sample, path, nfe = karras_sample(
                diffusion,
                model,
                y0,
                x0,
                steps=args.steps,
                model_kwargs=model_kwargs,
                device=dist_util.dev(),
                clip_denoised=args.clip_denoised,
                sampler=args.sampler,
                sigma_min=diffusion.sigma_min,
                sigma_max=diffusion.sigma_max,
                churn_step_ratio=args.churn_step_ratio,
                rho=args.rho,
                guidance=args.guidance
            )
            
            sample = (vae.decode(sample).sample).clamp(-1,1)
            sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
            sample = sample.permute(0, 2, 3, 1)
            sample = sample.contiguous()
            
            gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
            dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
            if index is not None:
                gathered_index = [th.zeros_like(index) for _ in range(dist.get_world_size())]
                dist.all_gather(gathered_index, index)
                gathered_samples = th.cat(gathered_samples)
                gathered_index = th.cat(gathered_index)
                gathered_samples = gathered_samples[th.argsort(gathered_index)]
            else:
                gathered_samples = th.cat(gathered_samples)

            num_display = min(32, sample.shape[0])
            if i == 0 and dist.get_rank() == 0:
                vutils.save_image(sample.permute(0,3,1,2)[:num_display].float(), f'{sample_dir}/sample_{i}.png', normalize=True,  nrow=int(np.sqrt(num_display)))
                if x0 is not None:
                    vutils.save_image(x0_image[:num_display], f'{sample_dir}/x_{i}.png',nrow=int(np.sqrt(num_display)))
                vutils.save_image(y0_image[:num_display]/2+0.5, f'{sample_dir}/y_{i}.png',nrow=int(np.sqrt(num_display)))
                vutils.save_image((vae.decode(y0[:num_display]).sample.detach().cpu())/2+0.5, f'{sample_dir}/y_{i}.png',nrow=int(np.sqrt(num_display)))
            all_images.append(gathered_samples.detach().cpu().numpy())
        
    logger.log(f"created {len(all_images) * args.batch_size * dist.get_world_size()} samples")
        
    arr = np.concatenate(all_images, axis=0)
    arr = arr[:args.num_samples]
    
    if dist.get_rank() == 0:
        shape_str = "x".join([str(x) for x in arr.shape])
        out_path = os.path.join(sample_dir, f"samples_{shape_str}_nfe{nfe}.npz")
        logger.log(f"saving to {out_path}")
        np.savez(out_path, arr)

    dist.barrier()
    logger.log("sampling complete")


if __name__ == "__main__":
    main()
