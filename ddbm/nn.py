"""
Various utilities for neural networks.
"""

import math

import torch as th
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.nn.modules.utils import _pair
import torch.nn.utils.parametrize as parametrize

# PyTorch 1.7 has SiLU, but we support PyTorch 1.5.
class SiLU(nn.Module):
    def forward(self, x):
        return x * th.sigmoid(x)


class GroupNorm32(nn.GroupNorm):
    def forward(self, x):
        return super().forward(x.float()).type(x.dtype)


# ---[ Symetrize operator blocks ]---------------
"""
    As discussed in "Structure Preserving GANs by Birrell et.al. (2022)"
    objectives (e.g., probability distributions on images) can be symmetrized, 
    that is, reduced to a set of equivalance classes induced by desired group symmetry
    properties. The following blocks implement operations that gauarantee this behaviour.
"""
class Vertical_Symmetric(nn.Module):
    def forward(self, X):
        _, _, h, w = X.shape
        upper_channel = h//2
        if not hasattr(self, 'upper_mask'):
            self.upper_mask = nn.Parameter(th.tensor([1.0]* upper_channel + [0.0] * (h - upper_channel), device = X.device)[None, None, :, None], requires_grad = False)

        return X * self.upper_mask + th.flip(X, dims=[-2]) * (1 - self.upper_mask)

    def right_inverse(self, A):
        return A
    

class Horizontal_Symmetric(nn.Module):
    def forward(self, X):
        _, _, h, w = X.shape
        left_channel = w//2
        if not hasattr(self, 'left_mask'):
            self.left_mask = nn.Parameter(th.tensor([1.0]* left_channel + [0.0] * (w - left_channel), device = X.device)[None, None, None, :], requires_grad = False)
        return X * self.left_mask + th.flip(X, dims=[-1]) * (1 - self.left_mask)
    
    def right_inverse(self, A):
        return A


class C4_Symmetric(nn.Module):
    def forward(self, X):
        _, _, h, w = X.shape
        assert h == w, 'the initialization assumes h == w'
        upper_channel = h//2
        if h % 2 == 0:
            
            tmp_ = th.tensor([[1]*upper_channel + [0]*(h - upper_channel)], device = X.device)
            up_left_mask = nn.Parameter((tmp_.T @ tmp_)[None, None, :, :], requires_grad = False)
            
            X_ = X * up_left_mask
            X__ = None
            for rot_ in range(3):
                X__ = th.rot90(X_, 1, [-1, -2]) if X__ is None else  th.rot90(X__, 1, [-1, -2])
                X_ = X_ + X__
            return X_
        else:
            tmp_A = th.tensor([[1.0]*upper_channel + [0.0]*(h - upper_channel)], device = X.device)
            tmp_B = th.tensor([[1.0]*(upper_channel + 1) + [0.0]*(h - (upper_channel + 1))], device = X.device)
            up_left_mask = nn.Parameter((tmp_A.T @ tmp_B)[None, None, :, :], requires_grad=False)

            center_elem_mask = th.zeros(h, w, device = X.device)
            center_elem_mask[h//2, h//2] = 1.0
            center_elem_mask = nn.Parameter(center_elem_mask, requires_grad=False)

            X_ = X * center_elem_mask.to(X.device)
            X__ = None
            for rot_ in range(4):
                X__ = th.rot90(X * up_left_mask.to(X.device), 1, [-1, -2]) if X__ is None else th.rot90(X__, 1, [-1, -2])
                X_ = X_ + X__
            return X_
        
    def right_inverse(self, A):
        return A
        

class D4_Symmetric(nn.Module):
    def forward(self, X):
        # make the weights symmetric 
        X = X.triu() + X.triu(1).transpose(-1, -2)

        _, _, h, w = X.shape
        assert h == w, 'the initialization assumes h == w'
        upper_channel = h//2
        if h % 2 == 0:
            
            tmp_ = th.tensor([[1]*upper_channel + [0] * ( h - upper_channel)], dtype = X.dtype, device = X.device)
            up_left_mask = (tmp_.T @ tmp_)[None, None, :, :]
            
            X_ = X * self.up_left_mask
            X__ = None
            for rot_ in range(3):
                X__ = th.rot90(X_, 1, [-1, -2]) if X__ is None else  th.rot90(X__, 1, [-1, -2])
                X_ = X_ + X__
            return X_
        else:
            tmp_A = th.tensor([[1.0]*upper_channel + [0.0] * ( h - upper_channel)], dtype = X.dtype, device = X.device)
            tmp_B = th.tensor([[1.0]*(upper_channel + 1) + [0.0] * ( h - (upper_channel + 1))], dtype = X.dtype, device = X.device)
            up_left_mask =(tmp_A.T @ tmp_B)[None, None, :, :]

            center_elem_mask = th.zeros(h, w, dtype = X.dtype, device = X.device)
            center_elem_mask[h//2, h//2] = 1.0

            X_ = X * center_elem_mask
            X__ = None
            for rot_ in range(4):
                X__ = th.rot90(X * up_left_mask, 1, [-1, -2]) if X__ is None else th.rot90(X__, 1, [-1, -2])
                X_ = X_ + X__
            return X_
    
    def right_inverse(self, A):
        return A

class KernelGConv2d(nn.Module):
    """
    Group equivariant convolution layer. Implemetation is based on manipulating the convolutional kernel.
    
    :parm g_input: One of ('Z2', 'C4', 'D4'). Use 'Z2' for the first layer. Use 'C4' or 'D4' for later layers.
        The parameter value 'Z2' specifies the data being convolved is from the Z^2 plane (discrete mesh).
    :parm g_output: One of ('C4', 'D4'). What kind of transformations to use (rotations or roto-reflections).
        The value of g_input of the subsequent layer should match the value of g_output from the previous.
    :parm in_channels: The number of input channels. 
    :parm out_channels: The number of output channels.
    """

    def __init__(self, 
                g_output, 
                in_channels, 
                out_channels, 
                kernel_size=3, 
                stride=1,
                padding=0, 
                bias=True) -> None:
        super(KernelGConv2d, self).__init__()

        self.g_output = g_output
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.conv_weight = nn.Parameter(th.randn(out_channels, in_channels, kernel_size, kernel_size))
        
        if bias:
            self.bias = nn.Parameter(th.randn(out_channels)[None, :,  None, None])

        # init.xavier_normal_(self.conv_weight)
        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels
        for k in _pair(self.kernel_size):
            n *= k
        stdv = 1. / math.sqrt(n)
        self.conv_weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        out = self.bias

        if self.g_output == 'V':
            out = out + F.conv2d(
                input = input, 
                weight = self.conv_weight, 
                bias = None, 
                stride = self.stride, 
                padding = self.padding,
                dilation = 1,
                groups = 1
            )

            out = out + F.conv2d(
                input = input, 
                weight = th.flip(self.conv_weight, dims = [-2]),
                bias = None, 
                stride = self.stride, 
                padding = self.padding,
                dilation = 1,
                groups = 1
            )
           
        elif self.g_output == 'H':
            out = out + F.conv2d(
                input = input, 
                weight = self.conv_weight, 
                bias = None, 
                stride = self.stride, 
                padding = self.padding,
                dilation = 1,
                groups = 1
            )

            out = out + F.conv2d(
                input = input, 
                weight = th.flip(self.conv_weight, dims = [-1]),
                bias = None, 
                stride = self.stride, 
                padding = self.padding,
                dilation = 1,
                groups = 1
            )
        elif self.g_output == 'C4':
            for k in range(4):
                out = out + F.conv2d(
                    input = input, 
                    weight = th.rot90(self.conv_weight, k = k, dims = [-1, -2]),
                    bias = None, 
                    stride = self.stride, 
                    padding = self.padding,
                    dilation = 1,
                    groups = 1
                )
        elif self.g_output == "D4":
            kernel = self.conv_weight
            for k in range(4):
                kernel = th.rot90(kernel, k=1, dims=[-1, -2])
                out = out + F.conv2d(
                    input = input, 
                    weight = kernel,
                    bias = None, 
                    stride = self.stride, 
                    padding = self.padding,
                    dilation = 1,
                    groups = 1
                )
            kernel = th.flip(self.conv_weight, dims=[-2])
            out += out
            for k in range(4):
                kernel = th.rot90(kernel, k=1, dims=[-1, -2])
                out = out + F.conv2d(
                    input = input, 
                    weight = kernel,
                    bias = None, 
                    stride = self.stride, 
                    padding = self.padding,
                    dilation = 1,
                    groups = 1
                )
        else:
            raise NotImplementedError
        
        return out


def conv_nd(dims, *args, **kwargs):
    print(f'Initializing normal cnn layer.')
    if dims == 1:
        return nn.Conv1d(*args, **kwargs)
    elif dims == 2:
        return nn.Conv2d(*args, **kwargs)
    elif dims == 3:
        return nn.Conv3d(*args, **kwargs)


def gconv_nd(dims, g_output, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D group equivariant convolution layer or normal layers.
    """
    if dims == 1:
        print(f'Initializing normal cnn layer.')
        return nn.Conv1d(*args, **kwargs)
    elif dims == 2:
        # Deal we special case 
        if g_output == 'Z2':
            return nn.Conv2d(*args, **kwargs)
        else:
            if len(g_output.split('_')) > 1:
                g_output, suffix = g_output.split('_')
            else:
                suffix = None
            # If simple kernel symmetric layer  is desired
            if suffix == 'K':
                print(f'Initializing K weight tied equiv. cnn layer: {g_output}')
                return KernelGConv2d(g_output, *args, **kwargs)
            # If masking kernel symmetric kerel layer is desired 
            elif suffix == 'S':
                print(f'Initializing G weight tied equiv. cnn layer: {g_output}')
                if g_output == 'H':
                    layer = nn.Conv2d(*args, **kwargs)
                    parametrize.register_parametrization(layer, "weight", Horizontal_Symmetric())  
                    return layer 
                elif g_output == 'V':
                    layer = nn.Conv2d(*args, **kwargs)
                    parametrize.register_parametrization(layer, "weight", Vertical_Symmetric())   
                    return layer
                elif g_output == 'C4':
                    layer = nn.Conv2d(*args, **kwargs)
                    parametrize.register_parametrization(layer, "weight", C4_Symmetric())
                    return layer   
                elif g_output == 'D4':
                    layer = nn.Conv2d(*args, **kwargs)
                    # layer.weight.data = layer.weight.data.half()
                    parametrize.register_parametrization(layer, "weight", D4_Symmetric())  
                    return layer
            elif suffix == None:
                raise NotImplementedError(f"unsupported g_input g_ouput combination in gconv_nd: {g_output}\n or unsupported suffix: {suffix}")
    elif dims == 3:
        print(f'Initializing normal cnn layer.')
        return nn.Conv3d(*args, **kwargs)
    else:
        raise ValueError(f"unsupported dimensions for equivariant in gconv_nd: {dims}")
    

def linear(*args, **kwargs):
    """
    Create a linear module.
    """
    return nn.Linear(*args, **kwargs)


def avg_pool_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D average pooling module.
    """
    if dims == 1:
        return nn.AvgPool1d(*args, **kwargs)
    elif dims == 2:
        return nn.AvgPool2d(*args, **kwargs)
    elif dims == 3:
        return nn.AvgPool3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


def update_ema(target_params, source_params, rate=0.99):
    """
    Update target parameters to be closer to those of source parameters using
    an exponential moving average.

    :param target_params: the target parameter sequence.
    :param source_params: the source parameter sequence.
    :param rate: the EMA rate (closer to 1 means slower).
    """
    for targ, src in zip(target_params, source_params):
        targ.detach().mul_(rate).add_(src, alpha=1 - rate)


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


def scale_module(module, scale):
    """
    Scale the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().mul_(scale)
    return module


def mean_flat(tensor):
    """
    Take the mean over all non-batch dimensions.
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))


def append_dims(x, target_dims):
    """Appends dimensions to the end of a tensor until it has target_dims dimensions."""
    dims_to_append = target_dims - x.ndim
    if dims_to_append < 0:
        raise ValueError(
            f"input has {x.ndim} dims but target_dims is {target_dims}, which is less"
        )
    return x[(...,) + (None,) * dims_to_append]


def append_zero(x):
    return th.cat([x, x.new_zeros([1])])


def normalization(channels):
    """
    Make a standard normalization layer.

    :param channels: number of input channels.
    :return: an nn.Module for normalization.
    """
    return GroupNorm32(32, channels)


def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.

    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = th.exp(
        -math.log(max_period) * th.arange(start=0, end=half, dtype=th.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = th.cat([th.cos(args), th.sin(args)], dim=-1)
    if dim % 2:
        embedding = th.cat([embedding, th.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


def checkpoint(func, inputs, params, flag):
    """
    Evaluate a function without caching intermediate activations, allowing for
    reduced memory at the expense of extra compute in the backward pass.

    :param func: the function to evaluate.
    :param inputs: the argument sequence to pass to `func`.
    :param params: a sequence of parameters `func` depends on but does not
                   explicitly take as arguments.
    :param flag: if False, disable gradient checkpointing.
    """
    if flag:
        args = tuple(inputs) + tuple(params)
        return CheckpointFunction.apply(func, len(inputs), *args)
    else:
        return func(*inputs)


class CheckpointFunction(th.autograd.Function):
    @staticmethod
    def forward(ctx, run_function, length, *args):
        ctx.run_function = run_function
        ctx.input_tensors = list(args[:length])
        ctx.input_params = list(args[length:])
        with th.no_grad():
            output_tensors = ctx.run_function(*ctx.input_tensors)
        return output_tensors

    @staticmethod
    def backward(ctx, *output_grads):
        ctx.input_tensors = [x.detach().requires_grad_(True) for x in ctx.input_tensors]
        with th.enable_grad():
            # Fixes a bug where the first op in run_function modifies the
            # Tensor storage in place, which is not allowed for detach()'d
            # Tensors.
            shallow_copies = [x.view_as(x) for x in ctx.input_tensors]
            output_tensors = ctx.run_function(*shallow_copies)
        input_grads = th.autograd.grad(
            output_tensors,
            ctx.input_tensors + ctx.input_params,
            output_grads,
            allow_unused=True,
        )
        del ctx.input_tensors
        del ctx.input_params
        del output_tensors
        return (None, None) + input_grads
