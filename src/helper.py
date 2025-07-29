# %%capture
# !pip install beartype

# +
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from random import random
from beartype.typing import List, Union
from beartype import beartype
from tqdm.auto import tqdm
from functools import partial, wraps
from contextlib import contextmanager, nullcontext
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from torch import nn, einsum
from torch.cuda.amp import autocast
from torch.special import expm1
import torchvision.transforms as T


from einops import rearrange, repeat, reduce, pack, unpack
from einops.layers.torch import Rearrange


 
# +
# helper functions


def first(arr, d = None):
    if len(arr) == 0:
        return d
    return arr[0]

def resize_image_to(
    image,
    target_image_size,
    clamp_range = None,
    mode = 'nearest'):
    orig_image_size = image.shape[-1]

    if orig_image_size == target_image_size:
        return image

    out = F.interpolate(image, target_image_size, mode = mode)

    if exists(clamp_range):
        out = out.clamp(*clamp_range)

    return out

def eval_decorator(fn):
    def inner(model, *args, **kwargs):
        was_training = model.training
        model.eval()
        out = fn(model, *args, **kwargs)
        model.train(was_training)
        return out
    return inner

def module_device(module):
    return next(module.parameters()).device

def cast_uint8_images_to_float(images):
    if not images.dtype == torch.uint8:
        return images
    return images / 255

def exists(val):
    return val is not None

def identity(t, *args, **kwargs):
    return t

def divisible_by(numer, denom):
    return (numer % denom) == 0

def maybe(fn):
    @wraps(fn)
    def inner(x):
        if not exists(x):
            return x
        return fn(x)
    return inner

def once(fn):
    called = False
    @wraps(fn)
    def inner(x):
        nonlocal called
        if called:
            return
        called = True
        return fn(x)
    return inner

print_once = once(print)

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def cast_tuple(val, length = None):
    if isinstance(val, list):
        val = tuple(val)

    output = val if isinstance(val, tuple) else ((val,) * default(length, 1))

    if exists(length):
        assert len(output) == length

    return output

def cast_tuple(input_dict):
    return {key: value for key, value in input_dict.items() if exists(value)}


def zero_init_(m):
    nn.init.zeros_(m.weight)
    if exists(m.bias):
        nn.init.zeros_(m.bias)

# helper classes

class Identity(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        return x



# +
# tensor helpers

def log(t, eps: float = 1e-12):
    return torch.log(t.clamp(min = eps))

def l2norm(t):
    return F.normalize(t, dim = -1)

def right_pad_dims_to(x, t):
    padding_dims = x.ndim - t.ndim
    if padding_dims <= 0:
        return t
    return t.view(*t.shape, *((1,) * padding_dims))

def masked_mean(t, *, dim, mask = None):
    if not exists(mask):
        return t.mean(dim = dim)

    denom = mask.sum(dim = dim, keepdim = True)
    mask = rearrange(mask, 'b n -> b n 1')
    masked_t = t.masked_fill(~mask, 0.)

    return masked_t.sum(dim = dim) / denom.clamp(min = 1e-5)

def resize_image_to(
    image,
    target_image_size,
    clamp_range = None,
    mode = 'nearest'
):
    orig_image_size = image.shape[-1]

    if orig_image_size == target_image_size:
        return image

    out = F.interpolate(image, target_image_size, mode = mode)

    if exists(clamp_range):
        out = out.clamp(*clamp_range)

    return out


# -


# image normalization functions
# ddpms expect images to be in the range of -1 to 

def normalize_neg_one_to_one(img):
    return img * 2 - 1

def unnormalize_zero_to_one(normed_img):
    return (normed_img + 1) * 0.5

# classifier free guidance functions

def prob_mask_like(shape, prob, device):
    if prob == 1:
        return torch.ones(shape, device = device, dtype = torch.bool)
    elif prob == 0:
        return torch.zeros(shape, device = device, dtype = torch.bool)
    else:
        return torch.zeros(shape, device = device).float().uniform_(0, 1) < prob

# gaussian diffusion with continuous time helper functions and classes
# large part of this was thanks to @crowsonkb at https://github.com/crowsonkb/v-diffusion-jax/blob/master/diffusion/utils.py

@torch.jit.script
def beta_linear_log_snr(t):
    return -torch.log(expm1(1e-4 + 10 * (t ** 2)))

@torch.jit.script
def alpha_cosine_log_snr(t, s: float = 0.008):
    return -log((torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** -2) - 1, eps = 1e-5) # not sure if this accounts for beta being clipped to 0.999 in discrete version

def log_snr_to_alpha_sigma(log_snr):
    return torch.sqrt(torch.sigmoid(log_snr)), torch.sqrt(torch.sigmoid(-log_snr))


