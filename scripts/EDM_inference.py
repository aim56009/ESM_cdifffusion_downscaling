# # SETUP

# %%capture
# !pip install xarray
# !pip install wandb
# !pip install collections
# !pip install pysteps
# !pip install beartype
# !pip install scikit-image
# !pip install netcdf4

# +
import os
import math
import copy
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import xarray as xr
import wandb
import IPython.display as display
import logging
import torch.nn.functional as F
import collections
import copy
import torchvision.transforms as transforms


from PIL import Image
from tqdm import tqdm
from torch import optim
from PIL import Image
from torch.utils import data
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import random_split
from inspect import isfunction
from functools import partial
from abc import abstractmethod
from skimage.metrics import structural_similarity as ssim
from pysteps.utils.spectral import rapsd, corrcoef
import matplotlib.ticker as ticker

import tqdm
#from src.psd_utils import SpatialSpectralDensity_4_diff_res, SpatialSpectralDensity_diff_res

from src.utils import *
from src.utils_essential import *


from src.base_network import BaseNetwork
from src.edm_generation import diffusion_step, RandomGenerator, stochastic_sampler
# -

from src.dataloader_sr import gfdl_eval_256, era5_upscaled_1d_256, era5_0_25d_256, qm_gfdl_trafo_units_hr
from src.dataloader_sr import QM_GFDL_LR_Dataset_256

config = {"run_name": "revision_palette",     
          "epochs":        400,
          "batch_size":    1, 
          "lr":            1e-5, 
          "image_size":    256,             
          "device":        "cuda", 
          "num_workers":   8, 
}
#wandb.config.update({"batch_size": 32})

wandb.init(project='climate-diffusion', entity='Michi',config=config, save_code=True)

logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, datefmt="%I:%M:%S")

# # Dataloaders 

# +
era5_hr_ds = era5_0_25d_256(stage='train')


era5_hr_dl = data.DataLoader(era5_hr_ds, batch_size=wandb.config.batch_size, shuffle=False, drop_last=True,
                                     num_workers=wandb.config.num_workers)

sample_era5_025_tr = next(iter(era5_hr_dl))
print("HR ERA5", sample_era5_025_tr.shape)

# +
era5_lr_ds = era5_upscaled_1d_256(stage='train')

era5_lr_dl = data.DataLoader(era5_lr_ds, batch_size=wandb.config.batch_size, shuffle=False, drop_last=True,
                                     num_workers=wandb.config.num_workers)

sample_era5_1d_256p = next(iter(era5_lr_dl))
print("LR ERA5", sample_era5_1d_256p.shape)
# -

# ## validation

# +
bs_valid = 5

eval_true = True
# -

if eval_true == True:
    era5_p_1d_256_v = era5_upscaled_1d_256(stage='valid')

    dataloader_era5_val_1d_256 = data.DataLoader(era5_p_1d_256_v, batch_size=bs_valid, shuffle=False, drop_last=True,
                                         num_workers=wandb.config.num_workers)

    era5_lr = next(iter(dataloader_era5_val_1d_256))
    print(era5_lr.shape)

if eval_true == True:
    era5_p025 = era5_0_25d_256(stage='valid')
    dataloader_era5_val_p025 = data.DataLoader(era5_p025, batch_size=bs_valid, shuffle=False, drop_last=True,
                                         num_workers=wandb.config.num_workers)

    era5_hr = next(iter(dataloader_era5_val_p025))
    print(era5_hr.shape)

#  qm-us inference QM-corrected gfdl
if eval_true == True:

    bc_gfdl_dataset_val = QM_GFDL_LR_Dataset_256('data/11_01_deltaQM_debiased_gfdl_valid_custom_dl.pth')

    dataloader_bc_gfdl_val = data.DataLoader(bc_gfdl_dataset_val, batch_size=bs_valid, shuffle=False,
                                      drop_last=True,num_workers=2)


    bc_gfld_sample = next(iter(dataloader_bc_gfdl_val))
    print("QM+US - GFDL LR 256 shape:",bc_gfld_sample.shape)

# # add EDM UNET

from EDM.preconditioning import *
from EDM.song_unet import SongUNet


class EDMLossSR:
    """
    Variation of the loss function proposed in the EDM paper for Super-Resolution.

    Parameters
    ----------
    P_mean: float, optional
        Mean value for `sigma` computation, by default -1.2.
    P_std: float, optional:
        Standard deviation for `sigma` computation, by default 1.2.
    sigma_data: float, optional
        Standard deviation for data, by default 0.5.
    """

    def __init__(
        self, P_mean: float = -1.2, P_std: float = 1.2, sigma_data: float = 0.5
    ):
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data

    def __call__(self, net, img_clean, img_lr, labels=None, augment_pipe=None):
        """
        Calculate and return the loss corresponding to the EDM formulation.

        The method adds random noise to the input images and calculates the loss as the
        square difference between the network's predictions and the input images.
        The noise level is determined by 'sigma', which is computed as a function of
        'P_mean' and 'P_std' random values. The calculated loss is weighted as a
        function of 'sigma' and 'sigma_data'.

        Parameters:
        ----------
        net: torch.nn.Module
            The neural network model that will make predictions.

        images: torch.Tensor
            Input images to the neural network.

        labels: torch.Tensor
            Ground truth labels for the input images.

        augment_pipe: callable, optional
            An optional data augmentation function that takes images as input and
            returns augmented images. If not provided, no data augmentation is applied.

        Returns:
        -------
        torch.Tensor
            A tensor representing the loss calculated based on the network's
            predictions.
        """
        rnd_normal = torch.randn([img_clean.shape[0], 1, 1, 1], device=img_clean.device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        weight = (sigma**2 + self.sigma_data**2) / (sigma * self.sigma_data) ** 2

        # augment for conditional generaiton
        img_tot = torch.cat((img_clean, img_lr), dim=1)
        y_tot, augment_labels = (
            augment_pipe(img_tot) if augment_pipe is not None else (img_tot, None)
        )
        y = y_tot[:, : img_clean.shape[1], :, :]
        y_lr = y_tot[:, img_clean.shape[1] :, :, :]

        n = torch.randn_like(y) * sigma
        
        test = y + n
        D_yn = net(y + n, y_lr, sigma, labels, augment_labels=augment_labels).to("cuda")

        #loss = weight * ((D_yn - y) ** 2)
        #loss = loss.sum() / batch_size_per_gpu
        #loss_accum += loss / num_accumulation_rounds
        
        ### compute losses like in Imagen ###
        losses = weight * F.mse_loss(D_yn, y, reduction = 'none')
        losses = reduce(losses, 'b ... -> b', 'mean')
        losses = losses.mean()
        return losses


# # define generate function

def generate_fn(image_lr, net_res):
        img_shape_x = image_lr.shape[2]
        img_shape_y = image_lr.shape[3]
        
        with nvtx.annotate("generate_fn", color="green"):
            image_lr_patch = image_lr
            image_lr_patch = image_lr_patch.to(memory_format=torch.channels_last)

            with nvtx.annotate("diffusion model", color="purple"):
                image_res = diffusion_step(
                    net=net_res,   
                    sampler_fn=sampler_fn,
                    seed_batch_size=image_lr.shape[0], 
                    img_shape=256,
                    img_out_channels=1,
                    img_lr=image_lr_patch.expand(image_lr.shape[0], -1, -1, -1).to(memory_format=torch.channels_last),
                    device="cuda",
                    hr_mean=None,
                )
 
            image_out = image_res
            return image_out


# # noise 

class GaussianDiffusionContinuousTimes(BaseNetwork):
    def __init__(self, *, noise_schedule, timesteps = 1000,**kwargs):
        super(GaussianDiffusionContinuousTimes, self).__init__(**kwargs)
        
        if noise_schedule == "linear":
            self.log_snr = beta_linear_log_snr
        elif noise_schedule == "cosine":
            self.log_snr = alpha_cosine_log_snr
        else:
            raise ValueError(f'invalid noise schedule {noise_schedule}')

        self.num_timesteps = timesteps

    def get_times(self, batch_size, noise_level, *, device):
        return torch.full((batch_size,), noise_level, device = device, dtype = torch.float32)

    def sample_random_times(self, batch_size, *, device):
        return torch.zeros((batch_size,), device = device).float().uniform_(0, 1)

    def get_condition(self, times):
        return maybe(self.log_snr)(times)

    def get_sampling_timesteps(self, batch, *, device):
        times = torch.linspace(1., 0., self.num_timesteps + 1, device = device)
        times = repeat(times, 't -> b t', b = batch)
        times = torch.stack((times[:, :-1], times[:, 1:]), dim = 0)
        times = times.unbind(dim = -1)
        return times

    def q_posterior(self, x_start, x_t, t, *, t_next = None):
        t_next = default(t_next, lambda: (t - 1. / self.num_timesteps).clamp(min = 0.))

        """ https://openreview.net/attachment?id=2LdBqxc1Yv&name=supplementary_material """
        log_snr = self.log_snr(t)
        log_snr_next = self.log_snr(t_next)
        log_snr, log_snr_next = map(partial(right_pad_dims_to, x_t), (log_snr, log_snr_next))

        alpha, sigma = log_snr_to_alpha_sigma(log_snr)
        alpha_next, sigma_next = log_snr_to_alpha_sigma(log_snr_next)

        # c - as defined near eq 33
        c = -expm1(log_snr - log_snr_next)
        posterior_mean = alpha_next * (x_t * (1 - c) / alpha + c * x_start)
        
        '''
        alpha_next * (x_t * (1 - c) / alpha + c * x_start)
        alpha_next * ( (x_t - x_t*c) / alpha + c * x_start)
        alpha_next * ( x_t/ alpha - c*x_t/ alpha + c * x_start)
        x_t*alpha_next/ alpha - c*x_t*alpha_next/ alpha + c * alpha_next*x_start)
        alpha_next/ alpha ( x_t - c * x_t + c * alpha * x_start)      witht  x_start = (x_t -simga*noise)/alpha
        alpha_next/alpha (x_t - c* x_t + c * alpha *[(x_t-simga*noise)/alpha])
        alpha_next/alpha (x_t - c* x_t + c*x_t - c*simga*noise)
        alpha_next/alpha (x_t - c*simga*noise)
        alpha_next/alpha (x_t + c) (- alpha/alpha_next * sigma * noise)
        '''
        
        # following (eq. 33)
        posterior_variance = (sigma_next ** 2) * c
        posterior_log_variance_clipped = log(posterior_variance, eps = 1e-20)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def q_sample(self, x_start, t, noise = None):
        dtype = x_start.dtype

        if isinstance(t, float):
            batch = x_start.shape[0]
            t = torch.full((batch,), t, device = x_start.device, dtype = dtype)

        noise = default(noise, lambda: torch.randn_like(x_start))
        log_snr = self.log_snr(t).type(dtype)
        log_snr_padded_dim = right_pad_dims_to(x_start, log_snr)
        alpha, sigma =  log_snr_to_alpha_sigma(log_snr_padded_dim)

        return alpha * x_start + sigma * noise, log_snr, alpha, sigma

    def q_sample_from_to(self, x_from, from_t, to_t, noise = None):
        shape, device, dtype = x_from.shape, x_from.device, x_from.dtype
        batch = shape[0]

        if isinstance(from_t, float):
            from_t = torch.full((batch,), from_t, device = device, dtype = dtype)

        if isinstance(to_t, float):
            to_t = torch.full((batch,), to_t, device = device, dtype = dtype)

        noise = default(noise, lambda: torch.randn_like(x_from))

        log_snr = self.log_snr(from_t)
        log_snr_padded_dim = right_pad_dims_to(x_from, log_snr)
        alpha, sigma = log_snr_to_alpha_sigma(log_snr_padded_dim)

        log_snr_to = self.log_snr(to_t)
        log_snr_padded_dim_to = right_pad_dims_to(x_from, log_snr_to)
        alpha_to, sigma_to = log_snr_to_alpha_sigma(log_snr_padded_dim_to)

        return x_from * (alpha_to / alpha) + noise * (sigma_to * alpha - sigma * alpha_to) / alpha

    def predict_start_from_v(self, x_t, t, v):
        log_snr = self.log_snr(t)
        log_snr = right_pad_dims_to(x_t, log_snr)
        alpha, sigma = log_snr_to_alpha_sigma(log_snr)
        return alpha * x_t - sigma * v

    def predict_start_from_noise(self, x_t, t, noise):
        log_snr = self.log_snr(t)
        log_snr = right_pad_dims_to(x_t, log_snr)
        alpha, sigma = log_snr_to_alpha_sigma(log_snr)
        return (x_t - sigma * noise) / alpha.clamp(min = 1e-8)
        #x_0 * alpha = x_t -simga*noise
        #noise = (x_t - x_0*alpha)/sigma


# # Diffusion

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


network_module = importlib.import_module("EDM.song_unet")


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
        scale_cond_input=False,
        **model_kwargs,
    ):
       # super().__init__(meta=EDMPrecondSRMetaData)  !! changed
        super().__init__()
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

        Parameters
        ----------
        sigma : Union[float list, torch.Tensor]
            The sigma value(s) to convert.

        Returns
        -------
        torch.Tensor
            The tensor representation of the provided sigma value(s).
        """
        return torch.as_tensor(sigma)


# # Training

class BaseModel():
    def __init__(self, phase,  dataloader, metrics, n_epochs=10, batch_size = 8, 
                  save_checkpoint_epoch=10,resume_state=False,
                 save_path_base="/dss/dssfs04/lwp-dss-0002/pn49fu/pn49fu-dss-0002/ge74xuf2/diffusion", **kwargs):
        """ init model with basic input, which are from __init__(**kwargs) function in inherited class """
        self.phase = phase
        self.device = config['device']
        self.n_epochs = n_epochs
        self.resume_state = resume_state
        self.save_checkpoint_epoch = save_checkpoint_epoch
        ''' optimizers and schedulers '''
        self.schedulers = []
        self.optimizers = []
        ''' process record '''
        self.batch_size = batch_size
        self.epoch = 0
        self.iter = 0 
        self.phase_loader = dataloader
        self.metrics = metrics
        self.results_dict = CustomResult([],[]) # {"name":[], "result":[]}
        self.save_path_base = save_path_base


   
    def train(self):
        while self.epoch <= self.n_epochs: 
            self.epoch += 1
            
            train_log, condition_after_unet, original_img = self.train_step()
            
            print("epoch:", self.epoch)
            
            if self.epoch % self.save_checkpoint_epoch == 0:
                print('Saving the self at the end of epoch {:.0f}'.format(self.epoch))
                self.save_everything()
            
            if self.epoch % 5 == 0:
                output_sampled = generate_fn(condition_after_unet.to("cuda"), net_res=net_res)
                #output_sampled = imagen.restoration(start_image_or_video=condition_after_unet.to(config["device"]))
                
                output_sampled_test = wandb.Image(output_sampled)
                wandb.log({"diffusion gen img": output_sampled_test})
                condition_after_unet_wb = wandb.Image(condition_after_unet)
                wandb.log({"condition img": condition_after_unet_wb})
                original_img_wb = wandb.Image(original_img)
                wandb.log({"original img": original_img_wb})
                
                print("proxy-ERA5")
                plot_images_no_lab(condition_after_unet)
                print("SR proxy-ERA5")
                plot_images_no_lab(output_sampled)
                print("ERA5")
                plot_images_no_lab(original_img)
                
                
                print("BLUE: ORIGINAL DATA VS ORANGE: DIFFUSION MODEL GENERATED")
                latitudinal_mean_three(original=original_img, generated=output_sampled, 
                                       label=condition_after_unet.detach() , var="p")
                print("BLUE: ORIGINAL DATA VS ORANGE: DIFFUSION MODEL GENERATED, GREEN: UNET GENERATED")
                histograms_three(original=original_img.detach(), generated=output_sampled.detach(),
                                 label= condition_after_unet.detach(),xlim_end=None, var="p")
                
                
                ssd = SpatialSpectralDensity_diff_res( 
                                      original_img.detach().cpu().numpy()
                                     ,output_sampled.detach().cpu().numpy()
                                     ,condition_after_unet.detach().cpu().numpy()
                                     ,new_labels = ["era5 hr"," sr era5","era5 lr"])
                ssd.run(num_times=None)
                ssd.plot_psd(fname="",model_resolution=0.25,model_resolution_2=0.25)
                
                plt.show()
                  

    def test(self):
        pass

    @abstractmethod
    def train_step(self):
        raise NotImplementedError('You must specify how to train your networks.')


    def test_step(self):
        pass
    
    def print_network(self, network):
        """ print network structure, only work on GPU 0 """
        if isinstance(network, nn.DataParallel) or isinstance(network, nn.parallel.DistributedDataParallel):
            network = network.module
        
        s, n = str(network), sum(map(lambda x: x.numel(), network.parameters()))
        net_struc_str = '{}'.format(network.__class__.__name__)
        print('Network structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
        print(s)

    def save_network(self, network, network_label):
        """ save network structure, only work on GPU 0 """
        save_filename = '{}_{}.pth'.format(self.epoch, network_label)
        #save_path = os.path.join(os.path.join("models", config['run_name'], save_filename))
        save_path = os.path.join(self.save_path_base,  config['run_name'],save_filename)
        
        if isinstance(network, nn.DataParallel) or isinstance(network, nn.parallel.DistributedDataParallel):
            network = network.module
        state_dict = network.state_dict()
        for key, param in state_dict.items():
            state_dict[key] = param.cpu()
        torch.save(state_dict, save_path)

    def load_network(self, network, network_label, model_path, strict=True):        
        if not os.path.exists(model_path):
            print('Pretrained model in [{:s}] is not existed, Skip it'.format(model_path))
            return

        print('Loading pretrained model from [{:s}] ...'.format(model_path))
        network.load_state_dict(torch.load(model_path), strict=strict)
        network.to(self.device)

    def save_training_state(self):
        """ saves training state during training, only work on GPU 0 """

        assert isinstance(self.optimizers, list) and isinstance(self.schedulers, list), 'optimizers and schedulers must be a list.'
        state = {'epoch': self.epoch, 'iter': self.iter, 'schedulers': [], 'optimizers': []}
        for s in self.schedulers:
            state['schedulers'].append(s.state_dict())
        for o in self.optimizers:
            state['optimizers'].append(o.state_dict())
        save_filename = '{}.state'.format(self.epoch)
        save_path = os.path.join(os.path.join(self.save_path_base, config['run_name'], save_filename))
        torch.save(state, save_path)

    def resume_training(self):
        """ resume the optimizers and schedulers for training, only work when phase is test or resume training enable """
        if self.phase!='train' or self.resume_state is None:
            return
        print('Beign loading training states'.format())
        assert isinstance(self.optimizers, list) and isinstance(self.schedulers, list), 'optimizers and schedulers must be a list.'
        
        state_path = "{}.state".format(self.resume_state)
        
        if not os.path.exists(state_path):
            print('Training state in [{:s}] is not existed, Skip it'.format(state_path))
            return

        print('Loading training state for [{:s}] ...'.format(state_path))
        resume_state = torch.load(state_path)#.to(self.device) 
        
        resume_optimizers = resume_state['optimizers']
        resume_schedulers = resume_state['schedulers']
        assert len(resume_optimizers) == len(self.optimizers), 'Wrong lengths of optimizers {} != {}'.format(len(resume_optimizers), len(self.optimizers))
        assert len(resume_schedulers) == len(self.schedulers), 'Wrong lengths of schedulers {} != {}'.format(len(resume_schedulers), len(self.schedulers))
        for i, o in enumerate(resume_optimizers):
            self.optimizers[i].load_state_dict(o)
        for i, s in enumerate(resume_schedulers):
            self.schedulers[i].load_state_dict(s)

        self.epoch = resume_state['epoch']
        self.iter = resume_state['iter']

    def load_everything(self):
        pass 
    
    @abstractmethod
    def save_everything(self):
        raise NotImplementedError('You must specify how to save your networks, optimizers and schedulers.')


class Palette(BaseModel):
    def __init__(self, networks, losses, sample_num, task, optimizers, log_iter,
                 model_path, dataloader_circ_1, dataloader_circ_2, ema_scheduler=None,scale=1,  **kwargs):
        ''' must to init BaseModel with kwargs '''
        super(Palette, self).__init__(**kwargs)

        ''' networks, dataloder, optimizers, losses, etc. '''
        self.model_path = model_path
        self.log_iter = log_iter
        self.loss_fn = losses
        self.netG = networks
        self.dataloader_circ_1 = dataloader_circ_1
        self.dataloader_circ_2 = dataloader_circ_2

        if ema_scheduler is not None:
            self.ema_scheduler = ema_scheduler
            self.netG_EMA = copy.deepcopy(self.netG)
            self.EMA = EMA(beta=self.ema_scheduler['ema_decay'])
        else:
            self.ema_scheduler = None
        
        ''' networks can be a list, and must convert by self.set_device function if using multiple GPU. '''
        self.netG.to(self.device)
        if self.ema_scheduler is not None:
            self.netG_EMA = self.netG.to(self.device) 
        self.load_networks(self.model_path)

        self.optG = torch.optim.Adam(list(filter(lambda p: p.requires_grad, self.netG.parameters())), **optimizers)
        self.optimizers.append(self.optG)
        self.resume_training() 

        #self.netG.set_loss(self.loss_fn)

        self.sample_num = sample_num
        self.task = task
        self.scale=scale
        
    def set_input(self, data):
        ''' must use set_device in tensor '''
        self.cond_image = data.get('cond_image').to(self.device)
        self.gt_image = data.get('gt_image').to(self.device)
    
        self.path = data['path']
        self.batch_size = len(data['path'])
    
    def get_current_visuals(self, phase='train'):
        dict = {
            'gt_image': (self.gt_image.detach()[:].float().cpu()+1)/2,
            'cond_image': (self.cond_image.detach()[:].float().cpu()+1)/2,
        }

        if phase != 'train':
            dict.update({
                'output': (self.output.detach()[:].float().cpu()+1)/2
            })
        return dict

    def save_current_results(self):
        ret_path = []
        ret_result = []
        for idx in range(self.batch_size):
            ret_path.append('GT_{}'.format(self.path[idx]))
            ret_result.append(self.gt_image[idx].detach().float().cpu())

            ret_path.append('Process_{}'.format(self.path[idx]))
            ret_result.append(self.visuals[idx::self.batch_size].detach().float().cpu())
            
            ret_path.append('Out_{}'.format(self.path[idx]))
            ret_result.append(self.visuals[idx-self.batch_size].detach().float().cpu())
        

        self.results_dict = self.results_dict._replace(name=ret_path, result=ret_result)
        return self.results_dict._asdict()


    def train_step(self):
        self.netG.train()
        total_loss = 0.0  
        
        total_batches = len(self.phase_loader.dataset) // self.phase_loader.batch_size
        pbar = tqdm(total=total_batches, position=0, leave=True)
        for i, elements in enumerate(zip(self.phase_loader, self.dataloader_circ_1)):
            self.optG.zero_grad()
            self.gt_image, self.cond_image = elements
            self.gt_image = self.gt_image.float().to(self.device)
            self.cond_image,_ ,_ ,_  = noise_sched.q_sample(self.cond_image.to("cpu"),torch.tensor(50))
            self.cond_image = torch.clip(self.cond_image,-1,1).to(self.device)

            loss = edm_loss(edm_diffusion, self.gt_image, self.cond_image)        

            # Backward pass and optimization
            loss.backward()
            self.optG.step()

            # Accumulate total loss for the epoch
            total_loss += loss.item()
            self.iter += self.batch_size

            if self.ema_scheduler is not None:
                if self.iter > self.ema_scheduler['ema_start'] and self.iter % self.ema_scheduler['ema_iter'] == 0:
                    self.EMA.update_model_average(self.netG_EMA, self.netG)

            for scheduler in self.schedulers:
                scheduler.step()

            pbar.update()


        avg_loss = total_loss / len(self.phase_loader)
        print(f"Avg Loss for Epoch: {avg_loss}")
        wandb.log({"loss": avg_loss})

        return avg_loss, self.cond_image, self.gt_image
    
    

    def test(self, use_tqdm=False):
        self.netG.eval()
        with torch.no_grad():
            total_batches = len(self.phase_loader.dataset) // self.phase_loader.batch_size
            pbar = tqdm(total=total_batches, position=0, leave=True)
            for i, elements in enumerate(zip(self.phase_loader, self.dataloader_circ_2)):
                if i == 0:
                    self.gt_image, self.cond_image  = elements
                    break 
                
            self.gt_image = self.gt_image.float().to(self.device)
            self.cond_image,_ ,_ ,_  = noise_sched.q_sample(self.cond_image.to("cpu"),torch.tensor(50))
            self.cond_image = torch.clip(self.cond_image,-1,1).to(self.device)
            self.output = generate_fn(image_lr=self.cond_image.to("cuda"), net_res=net_res)
                
            print("diffusion generated sample")
            plot_images_no_lab(self.output[:8])
            print("condition sample")
            plot_images_no_lab(self.cond_image[:8])
            print("original sample")
            plot_images_no_lab(self.gt_image[:8])


            print("BLUE: ORIGINAL DATA VS ORANGE: DIFFUSION MODEL GENERATED")
            latitudinal_mean_three(self.gt_image.detach(),
                             self.output.detach(),
                             self.cond_image.detach()
                            ,label_name=["hr era5","sr gfdl","lr gfdl"]) 

            print("BLUE: ORIGINAL DATA VS ORANGE: DIFFUSION MODEL GENERATED, GREEN: UNET GENERATED")
            histograms_three(self.gt_image.detach(),
                             self.output.detach(),
                             self.cond_image.detach()
                             ,xlim_end=None, var="p"
                             ,label_name=["hr era5","sr gfdl","lr gfdl"])
            
            ssd = SpatialSpectralDensity_diff_res( 
                                     self.gt_image.detach().cpu().numpy()
                                     ,self.output.detach().cpu().numpy()
                                     ,self.cond_image.detach().cpu().numpy()
                                     ,new_labels = ["hr era5","sr gfdl","lr gfdl"])
            ssd.run(num_times=None)
            ssd.plot_psd(fname="",model_resolution=0.25,model_resolution_2=0.25)

        return self.output


    def load_networks(self, model_path):
        """ save pretrained model and training state, which only do on GPU 0. """
        netG_label = self.netG.__class__.__name__
        self.load_network(network=self.netG, network_label=netG_label, model_path=model_path, strict=False)
        if self.ema_scheduler is not None:
            self.load_network(network=self.netG_EMA, network_label=netG_label+'_ema',model_path=model_path, strict=False)
          
        
    def load_pretrain_diffusion(self, model_path):
        self.netG.load_state_dict(torch.load(model_path), strict=False)
        self.netG.to(self.device)
        
        if self.ema_scheduler is not None:
            self.netG_EMA.load_state_dict(torch.load(model_path), strict=False)
            self.netG_EMA.to(self.device)
            return self.netG_EMA
        return self.netG
        
    
    def save_everything(self):
        """ load pretrained model and training state. """
        netG_label = self.netG.__class__.__name__
        self.save_network(network=self.netG, network_label=netG_label)
        if self.ema_scheduler is not None:
            self.save_network(network=self.netG_EMA, network_label=netG_label+'_ema')
        self.save_training_state()

CustomResult = collections.namedtuple('CustomResult', 'name result')


def mse_loss(output, target):
    return F.mse_loss(output, target)


from tqdm import tqdm
from src.imagen_unet_and_diffusion import *
from einops import reduce

# +
kwargs = {
    "phase": "train",
    "dataloader": era5_hr_dl, 
    "metrics": ["mae"],
    "resume_state" : "/dss/dssfs04/lwp-dss-0002/pn49fu/pn49fu-dss-0002/ge74xuf2/diffusion/revision_palette/100", 
    "n_epochs" : 100, 
    "batch_size" : config["batch_size"],
    "save_checkpoint_epoch" : 5, 
    "save_path_base":"/dss/dssfs04/lwp-dss-0002/pn49fu/pn49fu-dss-0002/ge74xuf2/diffusion",
}


edm_diffusion = EDMPrecondSR(img_resolution=256,
                            img_channels=1,
                            img_in_channels=1,
                            img_out_channels=1,
                            #sigma_data=1,
                            model_type="SongUNet",  #"SongUNetPosEmbd",
                            scale_cond_input=False,)
edm_loss = EDMLossSR()
noise_sched = GaussianDiffusionContinuousTimes(noise_schedule="cosine",timesteps=100)


palette_model = Palette(
    networks=edm_diffusion,
    losses=mse_loss,
    sample_num=8,
    task="inpainting",
    optimizers={"lr": 1e-4, "weight_decay": 0},  # was 5e-5
    log_iter = 1000,                                                            
    model_path = "/dss/dssfs04/lwp-dss-0002/pn49fu/pn49fu-dss-0002/ge74xuf2/diffusion/revision_palette/100_EDMPrecondSR.pth",
    dataloader_circ_1 = era5_lr_dl,  
    dataloader_circ_2 = dataloader_bc_gfdl_val, #dataloader_bc_gfdl,
    scale=0.5,
    ema_scheduler = None,
    **kwargs
    )

res_ckpt_filename = "/dss/dssfs04/lwp-dss-0002/pn49fu/pn49fu-dss-0002/ge74xuf2/diffusion/revision_palette/100_EDMPrecondSR.pth"
net_res = palette_model.load_pretrain_diffusion(res_ckpt_filename)
net_res = net_res.eval().to("cuda").to(memory_format=torch.channels_last)

sampler_fn = partial(
            stochastic_sampler,
            img_shape= 256, # img_shape[1],
            patch_shape=256,  #patch_shape[1],
            boundary_pix=0,
            overlap_pix=0,
            )


# +
do_training = False


if do_training==True:
    palette_model_result = palette_model.train()

# +
do_testing = False

#if error - rmv: .unsqueeze(1) in test()
if do_testing==True:
    palette_model_result = palette_model.test(False)
# -

# # generate

res_ckpt_filename = "/dss/dssfs04/lwp-dss-0002/pn49fu/pn49fu-dss-0002/ge74xuf2/diffusion/revision_palette/100_EDMPrecondSR.pth"
net_res = palette_model.load_pretrain_diffusion(res_ckpt_filename)
net_res = net_res.eval().to("cuda").to(memory_format=torch.channels_last)

bc_gfld_noisy,_ ,_ ,_  = noise_sched.q_sample(bc_gfld_sample,torch.tensor(50))
bc_gfld_noisy = torch.clip(bc_gfld_noisy,-1,1)

reconstruction_n = generate_fn(bc_gfld_noisy.to("cuda"), net_res)
reconstruction_n.shape

print("QM noisy GFDL")
plot_images_no_lab(bc_gfld_noisy[:5])
print("BC GFDL")
plot_images_no_lab(reconstruction_n[:5])

# +
#stop here
# -

# # GFDL eval

do_eval_sr_dataset = True

bs_val = 1400

if do_eval_sr_dataset == True:
    bc_qm_gfdl_dataset = QM_GFDL_LR_Dataset_256('data/11_01_deltaQM_debiased_gfdl_valid_custom_dl.pth')

    dataloader_embed_gfdl = data.DataLoader(bc_qm_gfdl_dataset, batch_size=bs_val, shuffle=False,
                                      drop_last=True,num_workers=2)


    embed_gfdl = next(iter(dataloader_embed_gfdl))
    print("embedded gfdl shape:",embed_gfdl.shape)

if do_eval_sr_dataset == True:
    era5_hr_ds = era5_0_25d_256(stage='valid')
    era5_hr_dl = data.DataLoader(era5_hr_ds, batch_size=bs_val, shuffle=False, drop_last=True)
    era5_hr_val = next(iter(era5_hr_dl))
    print("HR ERA5", era5_hr_val.shape)

# +
#dm_loaded = palette_model.load_pretrain_diffusion("/dss/dssfs04/lwp-dss-0002/pn49fu/pn49fu-dss-0002/ge74xuf2/diffusion/sr_imagen/40_Imagen.pth")

# +
do_save_sr_dataset = False

res_ckpt_filename = "/dss/dssfs04/lwp-dss-0002/pn49fu/pn49fu-dss-0002/ge74xuf2/diffusion/revision_palette/100_EDMPrecondSR.pth"
net_res = palette_model.load_pretrain_diffusion(res_ckpt_filename)
net_res = net_res.eval().to("cuda").to(memory_format=torch.channels_last)

if do_save_sr_dataset==True:
    output_tensors = []
    
    
    for b, el in enumerate(dataloader_embed_gfdl):
        print(b)
        gfdl_qm_ = el.to("cuda").float()   # was .unsqueeze(1)
        
        gfdl_qm,_ ,_ ,_  = noise_sched.q_sample(gfdl_qm_.to("cpu"),torch.tensor(50))
        gfdl_qm = torch.clip(gfdl_qm,-1,1)
        
        
        sr_gfdl_output = generate_fn(image_lr=gfdl_qm.to("cuda"), net_res=net_res)
        plot_images_no_lab(gfdl_qm_[:7])
        plot_images_no_lab(sr_gfdl_output[:7])
        
        ssim_list = []
        for i in range(sr_gfdl_output.size(0)):  
            sr_image = sr_gfdl_output.cpu()[i]
            bc_image = gfdl_qm_.cpu()[i]
            ssim_val = ssim(sr_image[0].numpy(), bc_image[0].numpy(), data_range=2.0).item()
            ssim_list.append(ssim_val)  
        #print("SSIM values", ssim_list)
        print("average SSIM", np.sum(ssim_list)/len(ssim_list))
            
        output_tensors.append(sr_gfdl_output)
        
        
        ssd = SpatialSpectralDensity_diff_res( era5_hr_ds.inverse_dwd_trafo(era5_hr_val).numpy(), 
                                                 era5_hr_ds.inverse_dwd_trafo(sr_gfdl_output.cpu()).numpy(),
                                                 era5_lr_ds.inverse_dwd_trafo(gfdl_qm_.cpu()).numpy(),
                                                 new_labels = ["ear5 hr","sr bc gfdl","qm gfdl lr"])
        ssd.run(num_times=None)
        ssd.plot_psd(fname="",model_resolution=0.25, model_resolution_2 = 0.25)
        
        latitudinal_mean_three(era5_hr_ds.inverse_dwd_trafo(era5_hr_val)
                             ,era5_hr_ds.inverse_dwd_trafo(sr_gfdl_output.cpu())
                             ,era5_lr_ds.inverse_dwd_trafo(gfdl_qm_.cpu())
                             ,label_name = ["ear5 hr","sr bc gfdl","qm gfdl lr"])


    sr_gfdl_dataset = torch.cat(output_tensors, dim=0) 
    
    print(sr_gfdl_dataset.shape)
# -

if do_save_sr_dataset == True:
    sr_gfdl_dataset = torch.cat(output_tensors, dim=0) 
    print(sr_gfdl_dataset.shape)

if do_save_sr_dataset==True:
    
    save_path = 'data/edm_e100.pth'
    torch.save(sr_gfdl_dataset, save_path)
    print("saving to:",save_path)

if do_eval_sr_dataset == True:
    from src.dataloader_sr import SR_BC_GFDL_Dataset_256

    sr_gfdl_val_sr_dataset = SR_BC_GFDL_Dataset_256('data/edm_e100.pth') # 1000_bcgfd_sr_imagen_e40

    dataloader_sr_gfdl_val = data.DataLoader(sr_gfdl_val_sr_dataset, batch_size=1400, shuffle=False, drop_last=True)

    dm_hr_gfdl = next(iter(dataloader_sr_gfdl_val))#.unsqueeze(1)
    print("batch size:",dm_hr_gfdl.shape)

dm_hr_gfdl = torch.clip(dm_hr_gfdl,-1,1)

if do_eval_sr_dataset == True:
    
    ssd = SpatialSpectralDensity_diff_res( era5_hr_ds.inverse_dwd_trafo(era5_hr_val).numpy()
                                         ,era5_hr_ds.inverse_dwd_trafo(dm_hr_gfdl.cpu().numpy())
                                         ,bc_qm_gfdl_dataset.inverse_dwd_trafo(embed_gfdl).numpy()
                                         ,new_labels = ["ear5 hr"," dm_hr_era5","bc gfdl lr"])
    ssd.run(num_times=None)

    ssd.plot_psd(fname="",model_resolution=0.25, model_resolution_2 = 0.25)
    
    
    latitudinal_mean_three(era5_hr_ds.inverse_dwd_trafo(era5_hr_val)
                             ,era5_hr_ds.inverse_dwd_trafo(dm_hr_gfdl.cpu())
                             ,bc_qm_gfdl_dataset.inverse_dwd_trafo(embed_gfdl)
                             ,label_name=["ear5 hr"," dm_hr_era5","bc gfdl lr"],var="p")
    
    histograms_three_np(era5_hr_ds.inverse_dwd_trafo(era5_hr_val).numpy()#[:500]
                         ,era5_hr_ds.inverse_dwd_trafo(dm_hr_gfdl.cpu().numpy())#[:500]
                         ,bc_qm_gfdl_dataset.inverse_dwd_trafo(embed_gfdl).numpy()#[:500]
                         ,xlim_end=300, bins=300, label_name=["ear5 hr"," dm_hr_era5","bc gfdl lr"],var="p")

if do_eval_sr_dataset == True:
    
    ssd = SpatialSpectralDensity_diff_res(era5_hr_val.numpy()
                                         ,dm_hr_gfdl.cpu().numpy()
                                         ,embed_gfdl.numpy()
                                         ,new_labels = ["ear5 hr"," dm_hr_era5","bc gfdl lr"])
    ssd.run(num_times=None)

    ssd.plot_psd(fname="",model_resolution=0.25, model_resolution_2 = 0.25)
    
    
    latitudinal_mean_three(era5_hr_val
                             ,dm_hr_gfdl.cpu()
                             ,embed_gfdl
                             ,label_name=["ear5 hr"," dm_hr_era5","bc gfdl lr"],var="p")
    
    histograms_three_np(era5_hr_val.numpy()
                         ,dm_hr_gfdl.cpu().numpy()
                         ,embed_gfdl.numpy()
                         ,xlim_end=None, label_name=["ear5 hr"," dm_hr_era5","bc gfdl lr"],var="p")


