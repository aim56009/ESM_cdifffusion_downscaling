import torch
from typing import Any, Callable, Optional
from torch import Tensor
import numpy as np


def stochastic_sampler(
    net: Any,
    latents: Tensor,
    img_lr: Tensor,
    class_labels: Optional[Tensor] = None,
    randn_like: Callable[[Tensor], Tensor] = torch.randn_like,
    img_shape: int = 448,
    patch_shape: int = 448,
    overlap_pix: int = 4,
    boundary_pix: int = 2,
    mean_hr: Optional[Tensor] = None,
    num_steps: int = 18,
    sigma_min: float = 0.002,
    sigma_max: float = 800,
    rho: float = 7,
    S_churn: float = 0,
    S_min: float = 0,
    S_max: float = float("inf"),
    S_noise: float = 1,
) -> Tensor:
    """
    Proposed EDM sampler (Algorithm 2) with minor changes to enable super-resolution and patch-based diffusion.

    Parameters
    ----------
    net : Any
        The neural network model that generates denoised images from noisy inputs.
    latents : Tensor
        The latent variables (e.g., noise) used as the initial input for the sampler.
    img_lr : Tensor
        Low-resolution input image for conditioning the super-resolution process.
    class_labels : Optional[Tensor], optional
        Class labels for conditional generation, if required by the model. By default None.
    randn_like : Callable[[Tensor], Tensor]
        Function to generate random noise with the same shape as the input tensor.
        By default torch.randn_like.
    img_shape : int
        The height and width of the full image (assumed to be square). By default 448.
    patch_shape : int
        The height and width of each patch (assumed to be square). By default 448.
    overlap_pix : int
        Number of overlapping pixels between adjacent patches. By default 4.
    boundary_pix : int
        Number of pixels to be cropped as a boundary from each patch. By default 2.
    mean_hr : Optional[Tensor], optional
        Optional tensor containing mean high-resolution images for conditioning. By default None.
    num_steps : int
        Number of time steps for the sampler. By default 18.
    sigma_min : float
        Minimum noise level. By default 0.002.
    sigma_max : float
        Maximum noise level. By default 800.
    rho : float
        Exponent used in the time step discretization. By default 7.
    S_churn : float
        Churn parameter controlling the level of noise added in each step. By default 0.
    S_min : float
        Minimum time step for applying churn. By default 0.
    S_max : float
        Maximum time step for applying churn. By default float("inf").
    S_noise : float
        Noise scaling factor applied during the churn step. By default 1.

    Returns
    -------
    Tensor
        The final denoised image produced by the sampler.
    """

    # Adjust noise levels based on what's supported by the network.
    sigma_min = max(sigma_min, net.sigma_min)
    sigma_max = min(sigma_max, net.sigma_max)

    # Time step discretization.
    step_indices = torch.arange(num_steps, dtype=torch.float64, device=latents.device)
    t_steps = (
        sigma_max ** (1 / rho)
        + step_indices
        / (num_steps - 1)
        * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))
    ) ** rho
    t_steps = torch.cat(
        [net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])]
    )  # t_N = 0

    b = latents.shape[0]
    Nx = torch.arange(img_shape)
    Ny = torch.arange(img_shape)
    grid = torch.stack(torch.meshgrid(Nx, Ny, indexing="ij"), dim=0)[
        None,
    ].expand(b, -1, -1, -1)

    # conditioning = [mean_hr, img_lr, global_lr, pos_embd]
    batch_size = img_lr.shape[0]
    x_lr = img_lr
    if mean_hr is not None:
        x_lr = torch.cat((mean_hr.expand(x_lr.shape[0], -1, -1, -1), x_lr), dim=1)
    ###global_index = None

    # input and position padding + patching
    if patch_shape != img_shape:
        input_interp = torch.nn.functional.interpolate(
            img_lr, (patch_shape, patch_shape), mode="bilinear"
        )
        x_lr = image_batching(
            x_lr,
            img_shape,
            img_shape,
            patch_shape,
            patch_shape,
            batch_size,
            overlap_pix,
            boundary_pix,
            input_interp,
        )
        """
        
        global_index = image_batching(
            grid.float(),
            img_shape,
            img_shape,
            patch_shape,
            patch_shape,
            batch_size,
            overlap_pix,
            boundary_pix,
        ).int()
        """

    # Main sampling loop.
    x_next = latents.to(torch.float64) * t_steps[0]
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):  # 0, ..., N-1
        x_cur = x_next
        # Increase noise temporarily.
        gamma = (
            min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= t_cur <= S_max else 0
        )
        t_hat = net.round_sigma(t_cur + gamma * t_cur)
        x_hat = x_cur + (t_hat**2 - t_cur**2).sqrt() * S_noise * randn_like(x_cur)

        # Euler step. Perform patching operation on score tensor if patch-based generation is used
        if patch_shape != img_shape:
            x_hat_batch = image_batching(
                x_hat,
                img_shape,
                img_shape,
                patch_shape,
                patch_shape,
                batch_size,
                overlap_pix,
                boundary_pix,
            )
        else:
            x_hat_batch = x_hat
        #denoised = net(x_hat_batch, x_lr, t_hat, class_labels, global_index=global_index).to(torch.float64)
        denoised = net(x_hat_batch, x_lr, t_hat, class_labels).to(torch.float64)

        if patch_shape != img_shape:
            denoised = image_fuse(
                denoised,
                img_shape,
                img_shape,
                patch_shape,
                patch_shape,
                batch_size,
                overlap_pix,
                boundary_pix,
            )
        d_cur = (x_hat - denoised) / t_hat
        x_next = x_hat + (t_next - t_hat) * d_cur

        # Apply 2nd order correction.
        if i < num_steps - 1:
            if patch_shape != img_shape:
                x_next_batch = image_batching(
                    x_next,
                    img_shape,
                    img_shape,
                    patch_shape,
                    patch_shape,
                    batch_size,
                    overlap_pix,
                    boundary_pix,
                )
            else:
                x_next_batch = x_next
            denoised = net(x_next_batch, x_lr, t_next, class_labels).to(torch.float64)
            if patch_shape != img_shape:
                denoised = image_fuse(
                    denoised,
                    img_shape,
                    img_shape,
                    patch_shape,
                    patch_shape,
                    batch_size,
                    overlap_pix,
                    boundary_pix,
                )
            d_prime = (x_next - denoised) / t_next
            x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)
    return x_next


class RandomGenerator:  # pragma: no cover
    """
    Wrapper for torch.Generator that uses a single random seed for the entire minibatch.
    """

    def __init__(self, device, seed):
        super().__init__()
        # Initialize a single generator with a fixed seed
        self.generator = torch.Generator(device).manual_seed(int(seed) % (1 << 32))

    def randn(self, size, **kwargs):
        # Generate a single tensor with random values for the entire batch
        return torch.randn(size, generator=self.generator, **kwargs)

    def randn_like(self, input):
        # Generate a tensor with random values with the same shape and properties as the input
        return self.randn(
            input.shape, dtype=input.dtype, layout=input.layout, device=input.device
        )

    def randint(self, *args, size, **kwargs):
        # Generate random integers for the entire batch
        return torch.randint(*args, size=size, generator=self.generator, **kwargs)


def diffusion_step(  
    net: torch.nn.Module,
    sampler_fn: callable,
    seed_batch_size: int,
    img_shape: tuple,
    img_out_channels: int,
    img_lr: torch.Tensor,
    device: torch.device,
    hr_mean: torch.Tensor = None,
) -> torch.Tensor:
    """
    Generate images using diffusion techniques as described in the relevant paper.

    Args:
        net (torch.nn.Module): The diffusion model network.
        sampler_fn (callable): Function used to sample images from the diffusion model.
        seed_batch_size (int): Number of seeds per batch.
        img_shape (tuple): Shape of the images, (height, width).
        img_out_channels (int): Number of output channels for the image.
        img_lr (torch.Tensor): Low-resolution input image.
        device (torch.device): Device to perform computations.
        hr_mean (torch.Tensor, optional): High-resolution mean tensor, to be used as an additional input. By default None.

    Returns:
        torch.Tensor: Generated images concatenated across batches.
    """

    img_lr = img_lr.to(memory_format=torch.channels_last)

    # Handling of the high-res mean
    additional_args = {}
    if hr_mean is not None:
        additional_args["mean_hr"] = hr_mean

    # Initialize random generator and generate latents for the entire batch
    rnd = RandomGenerator(device, seed=42)
    latents = rnd.randn(
        [
            seed_batch_size,
            img_out_channels,
            img_shape,
            img_shape,
        ],
        device=device,
    ).to(memory_format=torch.channels_last)

    # Generate images using the sampler function
    with torch.inference_mode():
        images = sampler_fn(
            net, latents, img_lr, randn_like=rnd.randn_like, **additional_args
        )

    return images


