import copy
import functools
import os

from pathlib import Path

import blobfile as bf
import torch as th
import torch.distributed as dist
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torch.optim import RAdam
from . import dist_util, logger
from .fp16_util import MixedPrecisionTrainer
from .nn import update_ema
from .resample import LossAwareSampler, UniformSampler

from ddbm.random_util import get_generator
from .fp16_util import (
    get_param_groups_and_shapes,
    make_master_params,
    master_params_to_model_params,
)
import numpy as np
from ddbm.script_util import NUM_CLASSES

from ddbm.karras_diffusion import karras_sample

import glob 
# For ImageNet experiments, this was a good default value.
# We found that the lg_loss_scale quickly climbed to
# 20-21 within the first ~1K steps of training.
INITIAL_LOG_LOSS_SCALE = 20.0

class TrainLoop:
    def __init__(
        self,
        *,
        model,
        diffusion,
        train_data,
        test_data,
        batch_size,
        microbatch,
        lr,
        ema_rate,
        log_interval,
        test_interval,
        save_interval,
        save_interval_for_preemption,
        resume_checkpoint,
        workdir,
        use_fp16=False,
        fp16_scale_growth=1e-3,
        schedule_sampler=None,
        weight_decay=0.0,
        lr_anneal_steps=0,
        total_training_steps=10000000,
        augment_pipe=None,
        **sample_kwargs,
    ):
        self.model = model
        self.diffusion = diffusion
        self.data = train_data
        self.test_data = test_data
        self.image_size = model.image_size
        self.batch_size = batch_size
        self.microbatch = microbatch if microbatch > 0 else batch_size
        self.lr = lr
        self.ema_rate = (
            [ema_rate]
            if isinstance(ema_rate, float)
            else [float(x) for x in ema_rate.split(",")]
        )
        self.log_interval = log_interval
        self.workdir = workdir
        self.test_interval = test_interval
        self.save_interval = save_interval
        self.save_interval_for_preemption = save_interval_for_preemption
        self.resume_checkpoint = resume_checkpoint
        self.use_fp16 = use_fp16
        self.fp16_scale_growth = fp16_scale_growth
        self.schedule_sampler = schedule_sampler or UniformSampler(diffusion)
        self.weight_decay = weight_decay
        self.lr_anneal_steps = lr_anneal_steps
        self.total_training_steps = total_training_steps

        self.step = 0
        self.resume_step = 0
        self.global_batch = self.batch_size * dist.get_world_size()

        self.sync_cuda = th.cuda.is_available()


        self._load_and_sync_parameters()
        self.mp_trainer = MixedPrecisionTrainer(
            model=self.model,
            use_fp16=self.use_fp16,
            fp16_scale_growth=fp16_scale_growth,
        )

        self.opt = RAdam(
            self.mp_trainer.master_params, lr=self.lr, weight_decay=self.weight_decay
        )
        if self.resume_step:
            self._load_optimizer_state()
            # Model was resumed, either due to a restart or a checkpoint
            # being specified at the command line.
            self.ema_params = [
                self._load_ema_parameters(rate) for rate in self.ema_rate
            ]
        else:
            self.ema_params = [
                copy.deepcopy(self.mp_trainer.master_params)
                for _ in range(len(self.ema_rate))
            ]

        if th.cuda.is_available():
            self.use_ddp = True
            self.ddp_model = DDP(
                self.model,
                device_ids=[dist_util.dev()],
                output_device=dist_util.dev(),
                broadcast_buffers=False,
                bucket_cap_mb=128,
                find_unused_parameters=False,
            )
        else:
            if dist.get_world_size() > 1:
                logger.warn(
                    "Distributed training requires CUDA. "
                    "Gradients will not be synchronized properly!"
                )
            self.use_ddp = False
            self.ddp_model = self.model

        self.step = self.resume_step

        self.generator = get_generator(sample_kwargs['generator'], self.batch_size,42)
        self.sample_kwargs = sample_kwargs

        self.augment = augment_pipe
    
    def _load_and_sync_parameters(self):
        resume_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        
        if resume_checkpoint:
            self.resume_step = parse_resume_step_from_filename(resume_checkpoint)
            if dist.get_rank() == 0:
                logger.log(f"loading model from checkpoint: {resume_checkpoint}...")
                logger.log('Resume step: ', self.resume_step)
                
            self.model.load_state_dict(
                # dist_util.load_state_dict(
                #     resume_checkpoint, map_location=dist_util.dev()
                # ),
                th.load(resume_checkpoint, map_location=dist_util.dev()),
            )
        
            dist.barrier()

    def _load_ema_parameters(self, rate):
        ema_params = copy.deepcopy(self.mp_trainer.master_params)

        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        ema_checkpoint = find_ema_checkpoint(main_checkpoint, self.resume_step, rate)
        if ema_checkpoint:
            if dist.get_rank() == 0:
                logger.log(f"loading EMA from checkpoint: {ema_checkpoint}...")
            # state_dict = dist_util.load_state_dict(
            #     ema_checkpoint, map_location=dist_util.dev()
            # )
            state_dict = th.load(ema_checkpoint, map_location=dist_util.dev())
            ema_params = self.mp_trainer.state_dict_to_master_params(state_dict)

            dist.barrier()
        return ema_params

    def _load_optimizer_state(self):
        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        if main_checkpoint.split('/')[-1].startswith("latest"):
            prefix = 'latest_'
        else:
            prefix = ''
        opt_checkpoint = bf.join(
            bf.dirname(main_checkpoint), f"{prefix}opt{self.resume_step:06}.pt"
        )
        if bf.exists(opt_checkpoint):
            logger.log(f"loading optimizer state from checkpoint: {opt_checkpoint}")
            # state_dict = dist_util.load_state_dict(
            #     opt_checkpoint, map_location=dist_util.dev()
            # )
            state_dict = th.load(opt_checkpoint, map_location=dist_util.dev())
            self.opt.load_state_dict(state_dict)
            dist.barrier()

    def preprocess(self, x):
        if x.shape[1] == 3:
            x =  x * 2 - 1
        return x

    def run_loop(self):
        self.model.train()
        training = True
        while training:
            for batch, cond, _, mask in self.data:
                
                if not (not self.lr_anneal_steps or self.step < self.total_training_steps):
                    # Save the last checkpoint if it wasn't already saved.
                    if (self.step - 1) % self.save_interval != 0:
                        self.save()
                    return

                # scale to [-1, 1]
                batch = self.preprocess(batch) # TODO: removed for testing 

                # Mask input if mask exists
                if mask[0] != -1 and mask is not None:
                    batch = batch*mask
                
                if self.augment is not None:
                    batch, _ = self.augment(batch)
                if isinstance(cond, th.Tensor) and batch.ndim == cond.ndim:
                    xT = self.preprocess(cond) # TODO: removed for testing
                    # xT = cond
                    # Mask input if mask exists
                    if mask[0] != -1 and mask is not None:
                        cond = xT*mask
                    cond = {'xT': xT}
                else:
                    cond['xT'] = self.preprocess(cond['xT']) # TODO: removed for testing
                    # cond['xT'] = cond['xT']

                took_step = self.run_step(batch, cond, mask=mask)
                if took_step and self.step % self.log_interval == 0:
                    logs = logger.dumpkvs()
                        
                if took_step and self.step % self.save_interval == 0:
                    self.save()
                    # Run for a finite amount of time in integration tests.
                    if os.environ.get("DIFFUSION_TRAINING_TEST", "") and self.step > 0:
                        return

                    test_batch, test_cond, _, mask = next(iter(self.test_data))
                    test_batch = self.preprocess(test_batch) # TODO: removed for testing

                    # Mask input if mask exists
                    if mask[0] != -1 and mask is not None:
                        test_batch = test_batch*mask
             
                    if isinstance(test_cond, th.Tensor) and test_batch.ndim == test_cond.ndim:
                        test_xT = self.preprocess(test_cond) # TODO: removed for testing
                        # test_xT = test_cond
                        if mask[0] != -1 and mask is not None:
                            test_xT = test_xT*mask
                        test_cond = {'xT': test_xT}
                    else:
                        test_cond['xT'] = self.preprocess(test_cond['xT']) # removed for testing
                        # test_cond['xT'] = test_cond['xT']

                    self.run_test_step(test_batch, test_cond, mask)
                    logs = logger.dumpkvs()

                if took_step and self.step % self.save_interval_for_preemption == 0:
                    self.save(for_preemption=True)

                if self.step%self.test_interval == 0 and self.step != 0:
                    training = False
                    break

        return self.step, self.ema_rate
        
    def run_step(self, batch, cond, mask=None):
        self.forward_backward(batch, cond, mask=mask)
        took_step = self.mp_trainer.optimize(self.opt)
        if took_step:
            self.step += 1
            self._update_ema()
        self._anneal_lr()
        self.log_step()
        return took_step

    def run_test_step(self, batch, cond, mask):
        with th.no_grad():
            self.forward_backward(batch, cond, mask=mask, train=False)

    # TODO: make dice and dice_tol arguments rather than hard coded values
    def forward_backward(self, batch, cond, mask=None, dice_weight=1, dice_tol=0.5, train=True):
        if train:
            self.mp_trainer.zero_grad()

        if mask[0] != -1 and mask is not None:
            mask = mask.to(dist_util.dev())

        for i in range(0, batch.shape[0], self.microbatch):
            micro = batch[i : i + self.microbatch].to(dist_util.dev())
            micro_cond = {
                k: v[i : i + self.microbatch].to(dist_util.dev())
                for k, v in cond.items()
            }
            last_batch = (i + self.microbatch) >= batch.shape[0]
            t, weights = self.schedule_sampler.sample(micro.shape[0], dist_util.dev())

            compute_losses = functools.partial(
                self.diffusion.training_bridge_losses,
                self.ddp_model,
                micro,
                t,
                model_kwargs=micro_cond,
                mask=mask,
                dice_weight=dice_weight,
                dice_tol=dice_tol
                )

            if last_batch or not self.use_ddp:
                losses = compute_losses()
            else:
                with self.ddp_model.no_sync():
                    losses = compute_losses()

            if isinstance(self.schedule_sampler, LossAwareSampler) and train:
                self.schedule_sampler.update_with_local_losses(
                    t, losses["loss"].detach()
                )

            loss = (losses["loss"] * weights).mean()
            log_loss_dict(
                self.diffusion, t, {k if train else 'test_'+k: v * weights for k, v in losses.items()}
            )
            if train:
                self.mp_trainer.backward(loss)

    def _update_ema(self):
        for rate, params in zip(self.ema_rate, self.ema_params):
            update_ema(params, self.mp_trainer.master_params, rate=rate)

    def _anneal_lr(self):
        if not self.lr_anneal_steps:
            return
        frac_done = (self.step) / self.lr_anneal_steps
        lr = self.lr * (1 - frac_done)
        for param_group in self.opt.param_groups:
            param_group["lr"] = lr

    def log_step(self):
        logger.logkv("step", self.step)
        logger.logkv("samples", (self.step + 1) * self.global_batch)

    def save(self, for_preemption=False):
        def maybe_delete_earliest(filename):
            wc = filename.split(f'{(self.step):06d}')[0]+'*'
            freq_states = list(glob.glob(os.path.join(get_blob_logdir(), wc)))
            if len(freq_states) > 3:
                earliest = min(freq_states, key=lambda x: x.split('_')[-1].split('.')[0])
                os.remove(earliest)     

        # if dist.get_rank() == 0 and for_preemption:
        #     maybe_delete_earliest(get_blob_logdir())
        def save_checkpoint(rate, params):
            state_dict = self.mp_trainer.master_params_to_state_dict(params)
            if dist.get_rank() == 0:
                logger.log(f"saving model {rate}...")
                if not rate:
                    filename = f"model_{(self.step):06d}.pt"
                else:
                    filename = f"ema_{rate}_{(self.step):06d}.pt"
                if for_preemption:
                    filename = f"freq_{filename}"
                    maybe_delete_earliest(filename)
                
                with bf.BlobFile(bf.join(get_blob_logdir(), filename), "wb") as f:
                    th.save(state_dict, f)

        for rate, params in zip(self.ema_rate, self.ema_params):
            save_checkpoint(rate, params)

        if dist.get_rank() == 0:
            filename = f"opt_{(self.step):06d}.pt"
            if for_preemption:
                filename = f"freq_{filename}"
                maybe_delete_earliest(filename)
                
            with bf.BlobFile(
                bf.join(get_blob_logdir(), filename),
                "wb",
            ) as f:
                th.save(self.opt.state_dict(), f)

        # Save model parameters last to prevent race conditions where a restart
        # loads model at step N, but opt/ema state isn't saved for step N.
        save_checkpoint(0, self.mp_trainer.master_params)
        dist.barrier()


def parse_resume_step_from_filename(filename):
    """
    Parse filenames of the form path/to/model_NNNNNN.pt, where NNNNNN is the
    checkpoint's number of steps.
    """
    split = filename.split("model_")
    if len(split) < 2:
        return 0
    split1 = split[-1].split(".")[0]
    try:
        return int(split1)
    except ValueError:
        return 0


def get_blob_logdir():
    # You can change this to be a separate path to save checkpoints to
    # a blobstore or some external drive.
    return logger.get_dir()


def find_resume_checkpoint():
    # On your infrastructure, you may want to override this to automatically
    # discover the latest checkpoint on your blob storage, etc.
    return None


def find_ema_checkpoint(main_checkpoint, step, rate):
    if main_checkpoint is None:
        return None
    if main_checkpoint.split('/')[-1].startswith("latest"):
        prefix = 'latest_'
    else:
        prefix = ''
    filename = f"{prefix}ema_{rate}_{(step):06d}.pt"
    path = bf.join(bf.dirname(main_checkpoint), filename)
    if bf.exists(path):
        return path
    return None


def log_loss_dict(diffusion, ts, losses):
    for key, values in losses.items():
        logger.logkv_mean(key, values.mean().item())
        # Log the quantiles (four quartiles, in particular).
        for sub_t, sub_loss in zip(ts.cpu().numpy(), values.detach().cpu().numpy()):
            quartile = int(4 * sub_t / diffusion.num_timesteps)
            logger.logkv_mean(f"{key}_q{quartile}", sub_loss)