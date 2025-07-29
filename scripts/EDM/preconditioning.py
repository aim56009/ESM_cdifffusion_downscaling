# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Preconditioning schemes used in the paper"Elucidating the Design Space of 
Diffusion-Based Generative Models".
"""

import importlib
import warnings
from dataclasses import dataclass
from typing import List, Union

import numpy as np
import nvtx
import torch

from EDM.song_unet import SongUNet  # noqa: F401 for globals
from EDM.meta import ModelMetaData


# +
#network_module = importlib.import_module("diffusion")

# +
# instead of Module use torch.nn.Module
# -

@dataclass
class EDMPrecondSRMetaData(ModelMetaData):
    """EDMPrecondSR meta data"""

    name: str = "EDMPrecondSR"
    # Optimization
    jit: bool = False
    cuda_graphs: bool = False
    amp_cpu: bool = False
    amp_gpu: bool = True
    torch_fx: bool = False
    # Data type
    bf16: bool = False
    # Inference
    onnx: bool = False
    # Physics informed
    func_torch: bool = False
    auto_grad: bool = False


class EDMPrecondSR(torch.nn.Module):
    """
    Improved preconditioning proposed in the paper "Elucidating the Design Space of
    Diffusion-Based Generative Models" (EDM) for super-resolution tasks

    Parameters
    ----------
    img_resolution : int
        Image resolution.
    img_channels : int
        Number of color channels.
    img_in_channels : int
        Number of input color channels.
    img_out_channels : int
        Number of output color channels.
    use_fp16 : bool
        Execute the underlying model at FP16 precision?, by default False.
    sigma_min : float
        Minimum supported noise level, by default 0.0.
    sigma_max : float
        Maximum supported noise level, by default inf.
    sigma_data : float
        Expected standard deviation of the training data, by default 0.5.
    model_type :str
        Class name of the underlying model, by default "SongUNetPosEmbd".
    **model_kwargs : dict
        Keyword arguments for the underlying model.

    Note
    ----
    References:
    - Karras, T., Aittala, M., Aila, T. and Laine, S., 2022. Elucidating the
    design space of diffusion-based generative models. Advances in Neural Information
    Processing Systems, 35, pp.26565-26577.
    - Mardani, M., Brenowitz, N., Cohen, Y., Pathak, J., Chen, C.Y.,
    Liu, C.C.,Vahdat, A., Kashinath, K., Kautz, J. and Pritchard, M., 2023.
    Generative Residual Diffusion Modeling for Km-scale Atmospheric Downscaling.
    arXiv preprint arXiv:2309.15214.
    """

    def __init__(
        self,
        img_resolution,
        img_channels,
        img_in_channels,
        img_out_channels,
        use_fp16=False,
        sigma_min=0.0,
        sigma_max=float("inf"),
        sigma_data=0.5,
        model_type="SongUNetPosEmbd",
        scale_cond_input=True,
        **model_kwargs,
    ):
        super().__init__(meta=EDMPrecondSRMetaData)
        self.img_resolution = img_resolution
        self.img_channels = img_channels  # TODO: this is not used, remove it
        self.img_in_channels = img_in_channels
        self.img_out_channels = img_out_channels
        self.use_fp16 = use_fp16
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_data = sigma_data
        self.scale_cond_input = scale_cond_input

        model_class = getattr(network_module, model_type)
        self.model = model_class(
            img_resolution=img_resolution,
            in_channels=img_in_channels + img_out_channels,
            out_channels=img_out_channels,
            **model_kwargs,
        )  # TODO needs better handling
        self.scaling_fn = self._get_scaling_fn()

    def _get_scaling_fn(self):
        if self.scale_cond_input:
            warnings.warn(
                "scale_cond_input=True does not properly scale the conditional input. "
                "(see https://github.com/NVIDIA/modulus/issues/229). "
                "This setup will be deprecated. "
                "Please set scale_cond_input=False.",
                DeprecationWarning,
            )
            return self._legacy_scaling_fn
        else:
            return self._scaling_fn

    @staticmethod
    def _scaling_fn(x, img_lr, c_in):
        return torch.cat([c_in * x, img_lr.to(x.dtype)], dim=1)

    @staticmethod
    def _legacy_scaling_fn(x, img_lr, c_in):
        return c_in * torch.cat([x, img_lr.to(x.dtype)], dim=1)

    @nvtx.annotate(message="EDMPrecondSR", color="orange")
    def forward(
        self,
        x,
        img_lr,
        sigma,
        force_fp32=False,
        **model_kwargs,
    ):
        # Concatenate input channels
        x = x.to(torch.float32)
        sigma = sigma.to(torch.float32).reshape(-1, 1, 1, 1)
        dtype = (
            torch.float16
            if (self.use_fp16 and not force_fp32 and x.device.type == "cuda")
            else torch.float32
        )

        c_skip = self.sigma_data**2 / (sigma**2 + self.sigma_data**2)
        c_out = sigma * self.sigma_data / (sigma**2 + self.sigma_data**2).sqrt()
        c_in = 1 / (self.sigma_data**2 + sigma**2).sqrt()
        c_noise = sigma.log() / 4

        if img_lr is None:
            arg = c_in * x
        else:
            arg = self.scaling_fn(x, img_lr, c_in)
        arg = arg.to(dtype)

        F_x = self.model(
            arg,
            c_noise.flatten(),
            class_labels=None,
            **model_kwargs,
        )

        if (F_x.dtype != dtype) and not torch.is_autocast_enabled():
            raise ValueError(
                f"Expected the dtype to be {dtype}, but got {F_x.dtype} instead."
            )

        D_x = c_skip * x + c_out * F_x.to(torch.float32)
        return D_x

    @staticmethod
    def round_sigma(sigma: Union[float, List, torch.Tensor]):
        """
        Convert a given sigma value(s) to a tensor representation.
        See EDMPrecond.round_sigma
        """
        return EDMPrecond.round_sigma(sigma)


