"""
    Script for training or fine-tuning Stable-Diffusion autoencoder model on
    images.

    All code for autoencoder, distributions, util, autoencoder_modeules is 
    originally from (https://github.com/CompVis/stable-diffusion).
"""


import signal 
import time
import os
from pathlib import Path
from glob import glob
import blobfile as bf
import argparse
import numpy as np
import torch as th
from torcheval.metrics.functional import multiclass_f1_score as F1
import torch.distributed as dist
import torchvision
from torch.optim import RAdam
import torch.backends.cudnn as cudnn
from datasets.image_datasets import load_data
from datasets.augment import AugmentPipe

from ddbm import dist_util, logger
from ddbm.resample import create_named_schedule_sampler
from ddbm.script_util import (
    get_workdir
)

from diffusers.models import AutoencoderKL
import lpips


class VAETrainLoop():
    """
    Class for fine-tuning stable diffusions' auto-encoder on custom datasets with 
    additional losses.
    """

    def __init__(
        self,
        model,
        resume_checkpoint,
        data,
        image_size,
        batch_size,
        lr,
        ema_rate,
        weight_decay=0.0,
        lr_anneal_steps=0,
        weight_l2=0.5,
        weight_lpips=0.002,
        fp16=False,
        eqv='Z2',
        decoder_only=False,
        log_interval=1000,
        sample_interval=1000,
        save_interval=10_000,
        total_training_steps=100_000,
        augment=None,
        num_workers=1
    ):
        # Data parameters
        if type(data) != list:
            self.data = [data]
        else:
            self.data = data
        self.image_size = image_size
        self.batch_size = batch_size
        self.augment = augment
        # Model training paramters
        self.step = 0
        self.lr = lr
        self.ema_rate = ema_rate
        self.weight_decay = weight_decay
        self.weight_l2 = weight_l2
        self.weight_lpips = weight_lpips
        self.lr_anneal_steps = lr_anneal_steps
        self.total_training_steps = total_training_steps
        # Logging paramters 
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.sample_interval = sample_interval 
        # Model states
        self.model = model
        self.resume_step = self.step
        self.resume_checkpoint = resume_checkpoint
        self.model_params = list(self.model.parameters())
        self.opt = RAdam(
            self.model_params, lr=self.lr, weight_decay=self.weight_decay
        )
        self.fp16 = fp16
        self.eqv = eqv
        self.decoder_only = decoder_only

        self.loss_fn_l2 = th.nn.MSELoss()
        self.loss_fn_lpips = lpips.LPIPS().to(dist_util.dev())

        if self.resume_checkpoint != "":
            self.resume_step = self._parse_resume_step_from_filename(resume_checkpoint)
            self.step = self.resume_step
            if dist.get_rank() == 0:
                logger.log(f"loading model from checkpoint: {resume_checkpoint}...")
                logger.log('Resume step: ', self.resume_step)
                
            self.model.load_state_dict(
                th.load(resume_checkpoint, map_location=dist_util.dev()),
            )
            dist.barrier()

        signal.signal(signal.SIGUSR1, self._sig_handler)

    def _sig_handler(self, signum, frame):
        logger.log(f"Recived SLURM SIGNAL {signum}, stopping up training...")
        self.save()

    def _parse_resume_step_from_filename(self, filename):
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

    def _master_params_to_state_dict(self, model, params):
        # TODO: update this function to work with fp16 multiple paramters 
        state_dict = model.state_dict()
        for i, (name, _value) in enumerate(model.named_parameters()):
            assert name in state_dict
            state_dict[name] = params[i]
        return state_dict   
    
    def save(self, for_preemption=False):
        def _maybe_delete_earliest(filename):
            wc = filename.split(f'{(self.step):06d}')[0]+'*'
            freq_states = list(glob.glob(os.path.join(logger.get_dir(), wc)))
            if len(freq_states) > 3:
                earliest = min(freq_states, key=lambda x: x.split('_')[-1].split('.')[0])
                os.remove(earliest)     

        def _save_checkpoint(params, for_preemption):
            state_dict = self._master_params_to_state_dict(self.model, self.model_params)
            if dist.get_rank() == 0:
                logger.log(f"saving model {self.step}...")
                filename = f"model_{(self.step):06d}.pt"
    
                if for_preemption:
                    filename = f"premt_{filename}"
                    _maybe_delete_earliest(filename)
                
                with bf.BlobFile(bf.join(logger.get_dir(), filename), "wb") as f:
                    th.save(state_dict, f)

        # Save model parameters last to prevent race conditions where a restart
        # loads model at step N, but opt/ema state isn't saved for step N.
        _save_checkpoint(self.model_params, for_preemption)
        dist.barrier()

    def log_step(self, loss_terms):
        logger.logkv("step", self.step)
        logger.logkv("loss_l2", loss_terms["loss_l2"])
        logger.logkv("loss_lpips", loss_terms["loss_lpips"])
        logger.logkv("loss", loss_terms["loss"])

    def dice_coefficient(self, pred, target, epsilon=1e-6):
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        
        # Calculate Dice coefficient
        dice = F1(pred_flat, target_flat, num_classes=2)

        return dice

    def dice_loss(self, pred, target):
        # Dice loss is 1 minus the Dice coefficient
        return 1 - self.dice_coefficient(pred, target)

    def get_eqv_op_pairs_aux_random(self, eqv):
        # Randomly sample a pair of augmentation operators.
        if eqv == 'H':
            def aug_op(x):
                return th.flip(x, [-1])

            def aug_op_inv(x):
                return th.flip(x, [-1])

        elif eqv == 'V':
            def aug_op(x):
                return th.flip(x, [-2])
                    
            def aug_op_inv(x):
                return th.flip(x, [-2])
            
        elif eqv == 'C4':
            k = np.random.randint(1,4)

            def aug_op(x):
                return th.rot90(x, k = k, dims = [-1, -2])
            
            def aug_op_inv(x):
                return th.rot90(x, k = k, dims = [-2, -1])
            
        elif eqv == 'D4':
            k = np.random.randint(0, 4)
            v_flip = np.random.randint(0, 2)

            def aug_op(x):
                return th.rot90(th.flip(x, [-2]) if v_flip else x, k = k, dims = [-1, -2])
            
            def aug_op_inv(x):
                x_ = th.rot90(x, k = k, dims = [-2, -1])
                return th.flip(x_, [-2]) if v_flip else x_

        else:
            raise NotImplementedError
        
        return aug_op, aug_op_inv


    def get_eqv_op_pairs_aux_full(self, eqv):
        # return all pairs of inv operator aug_op, aug_op_inv s.t aug_op (compose) aug_op_inv = Identity
        Id = lambda x: x
        
        if eqv == 'H':
            aug_pairs = [[Id, Id]]
            def aug_op(x):
                return th.flip(x, [-1])

            def aug_op_inv(x):
                return th.flip(x, [-1])
            
            aug_pairs.append([aug_op, aug_op_inv])

        elif eqv == 'V':
            aug_pairs = [[Id, Id]]
            def aug_op(x):
                return th.flip(x, [-2])
                    
            def aug_op_inv(x):
                return th.flip(x, [-2])
            
            aug_pairs.append([aug_op, aug_op_inv])
            
        elif eqv == 'C4':
            aug_pairs = [[Id, Id]]
            for k in range(1, 4):
                def aug_op(x):
                    return th.rot90(x, k = k, dims = [-1, -2])

                def aug_op_inv(x):
                    return th.rot90(x, k = k, dims = [-2, -1])

                aug_pairs.append([aug_op, aug_op_inv])
    
        elif eqv == 'D4':
            aug_pairs = []

            for k in range(0, 4):
                for v_flip in [True, False]:
                    def aug_op(x):
                        return th.rot90(th.flip(x, [-2]) if v_flip else x, k = k, dims = [-1, -2])
                    def aug_op_inv(x):
                        x_ = th.rot90(x, k = k, dims = [-2, -1])
                        return th.flip(x_, [-2]) if v_flip else x_

                    aug_pairs.append([aug_op, aug_op_inv])  

        else:
            raise NotImplementedError
        
        return aug_pairs

    def get_eqv_op_pairs(self, eqv, all_ops=True):
        # return pairs of inv operator aug_op, aug_op_inv s.t aug_op (compose) aug_op_inv = Idenity
        # the operator is randomly picked (with the identity one) if all_ops is False else all 
        # the operators are returned in a list (including the idenity one)
        if not all_ops:
            return self.get_eqv_op_pairs_aux_random(eqv)
        
        else:
            return self.get_eqv_op_pairs_aux_full(eqv)

    def encode(self, x):
        z = self.model.encode(x)
        return z
        
    def decode(self, z):    
        x = self.model.decode(z)
        return x

    def forward(self, x, sample=True):
        if sample:
            z = self.encode(x).latent_dist.sample()
        else:
            z = self.encode(x).latent_dist.mode()

        x_hat = self.decode(z).sample
        
        return x_hat

    def sample(self):
        self.model.eval()

        batch = th.cat([next(data)[0] for data in self.data])
        batch = batch.to(dist_util.dev())
        x_hat = self.forward(batch, sample=False)
        sample_grid = th.cat([batch, x_hat], dim=0)

        sample_dir = bf.join(logger.get_dir(), "samples")
        if dist.get_rank() == 0:
            os.makedirs(sample_dir, exist_ok=True)
            
        torchvision.utils.save_image((sample_grid+1.)/2., bf.join(sample_dir, f"{self.step}.png"), nrow=batch.shape[0])       

        if self.decoder_only:
            self.model.encoder.eval()
            self.model.decoder.train()
        else: 
            self.model.train()

    def train_step(self, batch):
        assert self.model.encoder.training or self.model.decoder.training

        def _compute_losses(x):
            if self.eqv == 'Z2':
                x_hat = self.forward(x)
                loss_l2 = self.loss_fn_l2(x,x_hat)
                loss_lpips = self.loss_fn_lpips(x,x_hat).mean()
                loss_terms = {
                    "loss":self.weight_l2*loss_l2 + self.weight_lpips*loss_lpips,
                    "loss_l2":loss_l2,
                    "loss_lpips": loss_lpips
                }
                loss = (
                    self.weight_l2*loss_l2 + self.weight_lpips*loss_lpips
                )
            elif self.eqv == 'C4':
                if self.eqv_reg == 'REG':
                    aug_op, aug_op_inv = self.get_eqv_op_pairs(inv_reg = self.inv_type, all_ops = False)   
                    posterior = self.model.encode(x).latent_dist
                    with th.no_grad():
                        posterior_aug = self.model.encode(aug_op(x)).latent_dist            

                    m = posterior.mode()
                    m_aug = aug_op_inv(posterior_aug.mode().detach())

                    std = posterior.std
                    std_aug = aug_op_inv(posterior_aug.std.detach())

                    z = self._get_sample_(m, std)

                    ## decoder
                    aug_op, aug_op_inv = get_eqv_op_pairs(inv_reg = self.inv_type, all_ops = False)
                    x_hat = self.model.decode(z).sample
                    with th.no_grad():
                        x_hat_aug = aug_op_inv(self.model.decode(aug_op(z)).sample.detach())


                    ## compute loss

                    # VAE reconstruction loss
                    loss_l2 = self.loss_fn_l2(x, x_hat)
                    loss_lpips = self.loss_fn_lpips(x, x_hat).mean()

                    
                    if use_dice_loss:
                        loss_dice = dice_loss(0.5 * x.mean(dim = 1, keepdim = True) + 0.5, 0.5 * x_hat.mean(dim = 1, keepdim = True).clamp(-1,1) + 0.5).mean()
                        with th.no_grad():
                            disc_loss_dice = dice_loss(x.mean(dim = 1, keepdim = True)>0, x_hat.mean(dim = 1, keepdim = True)>0).mean()

                    
                    # VAE inv regularization loss
                    enc_inv_m_l2 = self.reg_l2_enc_m(m, m_aug)
                    enc_inv_std_l2 = self.reg_l2_enc_std(std, std_aug)
                    dec_inv_l2 = self.reg_l2_dec(x_hat, x_hat_aug)
                    inv_loss = enc_inv_m_l2 + enc_inv_std_l2 + dec_inv_l2

                    training_loss = self.weight_l2*loss_l2 + self.weight_lpips*loss_lpips + self.reg_loss_weight * inv_loss
                    if use_dice_loss:
                        training_loss = training_loss + self.dice_loss_weight * loss_dice

                elif self.eqv_reg == 'FA':
                    

            return loss, loss_terms

        # mini-batch training 
        for i, batch_ in enumerate([batch[:len(batch)//2], batch[len(batch)//2:]]):
            batch_ = batch_.to(dist_util.dev())
            loss, loss_terms = _compute_losses(batch_)
            loss = loss/self.opt_freq/2
            loss.backward()

        if self.step % self.opt_freq == 0:
            self.opt.step()
            # Zero grad for next training step
            self.opt.zero_grad()

        self.step += 1
        self.log_step(loss_terms)

    def run_loop(self):
        if self.decoder_only:
            self.model.encoder.eval()
            self.model.decoder.train()
        else: 
            self.model.train()

        while self.step < self.total_training_steps \
              or self.step < self.lr_anneal_steps:
            
            # Save model incrementally
            if (self.save_interval != -1 and 
                self.step > 0 and 
                self.step % self.save_interval == 0
            ):
                self.save()
                th.cuda.empty_cache()
                dist.barrier()

            if (self.save_interval != -1 and 
                self.step > 0 and 
                self.step % self.sample_interval == 0
            ):
                self.sample()
                dist.barrier()

            batch = th.cat([next(data)[0] for data in self.data])
            self.train_step(batch)
 
            # Log incrementally
            if self.step % self.log_interval == 0:
                logs = logger.dumpkvs()

        return self.step


def add_dict_to_argparser(parser, default_dict):
    
    def _str2bool(v):
        """
        https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
        """
        if isinstance(v, bool):
            return v
        if v.lower() in ("yes", "true", "t", "y", "1"):
            return True
        elif v.lower() in ("no", "false", "f", "n", "0"):
            return False
        else:
            raise argparse.ArgumentTypeError("boolean value expected")

    for k, v in default_dict.items():
        v_type = type(v)
        if v is None:
            v_type = str
        elif isinstance(v, bool):
            v_type = _str2bool
        parser.add_argument(f"--{k}", default=v, type=v_type)


def create_argparser():
    defaults = dict(
        workdir="",
        data_dir="",
        image_size=64,
        global_batch_size=2048,
        batch_size=-1,
        microbatch=-1,      # -1 disables microbatches
        lr=1e-4,
        ema_rate="0.9999",  # comma-separated list of EMA values
        weight_decay=0.0,
        lr_anneal_steps=0,
        weight_l2=0.5,
        weight_lpips=0.002,
        fp16=False,
        eqv='Z2',
        decoder_only=False,
        log_interval=50,
        sample_interval=1000,
        save_interval=10_000,
        total_training_steps=100_000,
        resume_checkpoint="",
        augment=False,
        num_workers=1,
    )
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


def main(args):
    # Profiler code 
    th.backends.cudnn.benchmark = True

    workdir = args.workdir
    Path(workdir).mkdir(parents=True, exist_ok=True)
    
    dist_util.setup_dist()
    logger.configure(dir=workdir)

    if dist.get_rank() == 0:
        logger.log("Creating autoencoder model...")

    if args.resume_checkpoint == "":
        model_ckpts = list(glob(f'{workdir}/*model*[0-9].*'))
        if len(model_ckpts) > 0:
            max_ckpt = max(model_ckpts, key=lambda x: int(x.split('model_')[-1].split('.')[0]))
            if os.path.exists(max_ckpt):
                args.resume_checkpoint = max_ckpt
                if dist.get_rank() == 0:
                    logger.log('Resuming from checkpoint: ', max_ckpt)

    model = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae", use_safetensors=False).to(dist_util.dev())

    if dist.get_rank() == 0:
        logger.log("creating data loader...")

    if args.batch_size == -1:
        batch_size = args.global_batch_size // dist.get_world_size()
        if args.global_batch_size % dist.get_world_size() != 0:
            logger.log(
                f"warning, using smaller global_batch_size of {dist.get_world_size()*batch_size} instead of {args.global_batch_size}"
            )
    else:
        batch_size = args.batch_size

    data = [load_data(
        data_dir=data_dir,
        batch_size=batch_size,
        image_size=args.image_size,
        num_workers=args.num_workers,
        class_cond=False,
    ) for data_dir in args.data_dir.split(",")]
    
    if args.augment:
        augment = AugmentPipe(
                p=0.12,xflip=1e8, yflip=1, scale=1, rotate_frac=1, aniso=1, translate_frac=1
            )
    else:
        augment = None

    trainloop = VAETrainLoop(
            model=model,
            resume_checkpoint=args.resume_checkpoint,
            data=data,
            image_size=args.image_size,
            batch_size=batch_size,
            lr=args.lr,
            ema_rate=args.ema_rate,
            weight_decay=args.weight_decay,
            weight_l2=args.weight_l2,
            weight_lpips=args.weight_lpips,
            fp16=args.fp16,
            eqv=args.eqv,
            decoder_only=args.decoder_only,
            lr_anneal_steps=args.lr_anneal_steps,
            log_interval=args.log_interval,
            sample_interval=args.sample_interval,
            save_interval=args.save_interval,
            total_training_steps=args.total_training_steps,
            augment=args.augment,
            num_workers=args.num_workers
        )
    
    logger.log("Training...")
    trainloop.run_loop()
    

if __name__=="__main__":
    args = create_argparser().parse_args()
    main(args)