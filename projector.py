# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Project given image to the latent space of pretrained network pickle."""

import copy
import os
from time import perf_counter

import click
import imageio
import numpy as np
import PIL.Image
import torch
import torch.nn.functional as F

import dnnlib
import legacy

# ----------------------------------------------------------------------------

# Reference: https://github.com/PDillis/stylegan2-ada-pytorch/blob/dab4f40a4ec1031b55f2c4b8e59e530bd47e2cd1/projector.py


class VGG16FeaturesNVIDIA(torch.nn.Module):
    def __init__(self, vgg16):
        super(VGG16FeaturesNVIDIA, self).__init__()
        # ReLU is already included in the output of every conv output
        self.conv1_1 = vgg16.layers.conv1
        self.conv1_2 = vgg16.layers.conv2
        self.pool1 = vgg16.layers.pool1

        self.conv2_1 = vgg16.layers.conv3
        self.conv2_2 = vgg16.layers.conv4
        self.pool2 = vgg16.layers.pool2

        self.conv3_1 = vgg16.layers.conv5
        self.conv3_2 = vgg16.layers.conv6
        self.conv3_3 = vgg16.layers.conv7
        self.pool3 = vgg16.layers.pool3

        self.conv4_1 = vgg16.layers.conv8
        self.conv4_2 = vgg16.layers.conv9
        self.conv4_3 = vgg16.layers.conv10
        self.pool4 = vgg16.layers.pool4

        self.conv5_1 = vgg16.layers.conv11
        self.conv5_2 = vgg16.layers.conv12
        self.conv5_3 = vgg16.layers.conv13
        self.pool5 = vgg16.layers.pool5
        self.adavgpool = torch.nn.AdaptiveAvgPool2d(
            output_size=(7, 7)
        )  # We need this for 256x256 images (> 224x224)

        self.fc1 = vgg16.layers.fc1
        self.fc2 = vgg16.layers.fc2
        self.fc3 = vgg16.layers.fc3
        self.softmax = vgg16.layers.softmax

    def forward(self, x, layers, normed=True):
        """
        x is an image/tensor of shape [1, 3, 256, 256], and layers is a list of the names of the layers you wish
        to return in order to compare the activations with another image.
        Example:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            img1 = torch.randn(1, 3, 256, 256, device=device)
            img2 = torch.randn(1, 3, 256, 256, device=device)
            layers = ['conv1_1', 'conv1_2', 'conv3_3', 'conv3_3', 'fc3']  # Indeed, return twice conv3_3
            vgg16 = VGG16FeaturesNVIDIA(device=device)
            # Get the desired features from the layers list
            features1 = vgg16(img1, layers)
            features2 = vgg16(img2, layers)
            # Get, e.g., the MSE loss between the two features
            mse = torch.nn.MSELoss(reduction='mean')
            loss = sum(map(lambda x, y: mse(x, y), features1, features2))
        """
        # Legend: => conv2d, -> max pool 2d, ~> adaptive average pool 2d, ->> fc layer; shapes of input/output are shown
        assert layers is not None
        conv1_1 = self.conv1_1(x)  # [1, 3, 256, 256] => [1, 64, 256, 256]
        conv1_2 = self.conv1_2(conv1_1)  # [1, 64, 256, 256] => [1, 64, 256, 256]

        conv2_1 = self.conv2_1(
            self.pool1(conv1_2)
        )  # [1, 64, 256, 256] -> [1, 64, 128, 128] => [1, 128, 128, 128]
        conv2_2 = self.conv2_2(conv2_1)  # [1, 128, 128, 128] => [1, 128, 128, 128]

        conv3_1 = self.conv3_1(
            self.pool2(conv2_2)
        )  # [1, 128, 128, 128] -> [1, 128, 64, 64] => [1, 256, 64, 64]
        conv3_2 = self.conv3_2(conv3_1)  # [1, 256, 64, 64] => [1, 256, 64, 64]
        conv3_3 = self.conv3_3(conv3_2)  # [1, 256, 64, 64] => [1, 256, 64, 64]

        conv4_1 = self.conv4_1(
            self.pool3(conv3_3)
        )  # [1, 256, 64, 64] -> [1, 256, 32, 32] => [1, 512, 32, 32]
        conv4_2 = self.conv4_2(conv4_1)  # [1, 512, 32, 32] => [1, 512, 32, 32]
        conv4_3 = self.conv4_3(conv4_2)  # [1, 512, 32, 32] => [1, 512, 32, 32]

        conv5_1 = self.conv5_1(
            self.pool4(conv4_3)
        )  # [1, 512, 32, 32] -> [1, 512, 16, 16] => [1, 512, 16, 16]
        conv5_2 = self.conv5_2(conv5_1)  # [1, 512, 16, 16] => [1, 512, 16, 16]
        conv5_3 = self.conv5_3(conv5_2)  # [1, 512, 16, 16] => [1, 512, 16, 16]

        adavgpool = self.adavgpool(
            self.pool5(conv5_3)
        )  # [1, 512, 16, 16] -> [1, 512, 8, 8] ~> [1, 512, 7, 7]
        fc1 = self.fc1(adavgpool)  # [1, 512, 7, 7] ->> [1, 4096]; w/ReLU
        fc2 = self.fc2(fc1)  # [1, 4096] ->> [1, 4096]; w/ReLU
        fc3 = self.softmax(
            self.fc3(fc2)
        )  # [1, 4096] ->> [1, 1000]; w/o ReLU; apply softmax

        result_list = list()
        for layer in layers:
            if normed:
                result_list.append(eval(layer) / torch.numel(eval(layer)))
            else:
                result_list.append(eval(layer))
        return result_list


# ----------------------------------------------------------------------------
# Modified to project the image to the w* space instead of the w space (https://arxiv.org/abs/1904.03189)
def project(
    G,
    target: torch.Tensor,  # [C,H,W] and dynamic range [0,255], W & H must match G output resolution
    *,
    num_steps=1000,
    w_avg_samples=10000,
    initial_learning_rate=0.1,
    initial_noise_factor=0.05,
    lr_rampdown_length=0.25,
    lr_rampup_length=0.05,
    noise_ramp_length=0.75,
    regularize_noise_weight=1e5,
    verbose=False,
    device: torch.device,
):
    assert target.shape == (G.img_channels, G.img_resolution, G.img_resolution)

    def logprint(*args):
        if verbose:
            print(*args)

    G = copy.deepcopy(G).eval().requires_grad_(False).to(device)  # type: ignore

    # Compute w stats.
    logprint(f"Computing W midpoint and stddev using {w_avg_samples} samples...")
    z_samples = np.random.RandomState(123).randn(w_avg_samples, G.z_dim)
    w_samples = G.mapping(
        torch.from_numpy(z_samples).to(device), None
    )  # [N, L, C]    # from_mean = True
    w_samples = w_samples.cpu().numpy().astype(np.float32)  # [N, 1, C]
    w_avg = np.mean(w_samples, axis=0, keepdims=True)  # [1, L, C]
    w_std = (np.sum((w_samples - w_avg) ** 2) / w_avg_samples) ** 0.5

    # Setup noise inputs.
    noise_bufs = {
        name: buf
        for (name, buf) in G.synthesis.named_buffers()
        if "noise_const" in name
    }

    # Load VGG16 feature detector.
    url = "https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/vgg16.pt"
    with dnnlib.util.open_url(url) as f:
        vgg16 = torch.jit.load(f).eval().to(device)

    vgg16_features = VGG16FeaturesNVIDIA(vgg16)

    # Features for target image.
    target_images = target.unsqueeze(0).to(device).to(torch.float32)
    if target_images.shape[2] > 256:
        target_images = F.interpolate(target_images, size=(256, 256), mode="area")
    target_features = vgg16(target_images, resize_images=False, return_lpips=True)
    # This is too cumbersome to add as command-line input, so we leave it here; use whatever you need
    # layers = [ 'conv3_1', 'conv3_2', 'conv3_3', 'conv4_1', 'conv4_2',
    #             'conv4_3', 'conv5_1', 'conv5_2', 'conv5_3']
    # target_features = vgg16_features(target_images, layers)

    w_opt = torch.tensor(
        w_avg, dtype=torch.float32, device=device, requires_grad=True
    )  # pylint: disable=not-callable
    w_out = torch.zeros(
        [num_steps] + list(w_opt.shape[1:]), dtype=torch.float32, device=device
    )
    optimizer = torch.optim.Adam(
        [w_opt] + list(noise_bufs.values()),
        betas=(0.9, 0.999),
        lr=initial_learning_rate,
    )

    # Init noise.
    for buf in noise_bufs.values():
        buf[:] = torch.randn_like(buf)
        buf.requires_grad = True

    for step in range(num_steps):
        # Learning rate schedule.
        t = step / num_steps
        w_noise_scale = (
            w_std * initial_noise_factor * max(0.0, 1.0 - t / noise_ramp_length) ** 2
        )
        lr_ramp = min(1.0, (1.0 - t) / lr_rampdown_length)
        lr_ramp = 0.5 - 0.5 * np.cos(lr_ramp * np.pi)
        lr_ramp = lr_ramp * min(1.0, t / lr_rampup_length)
        lr = initial_learning_rate * lr_ramp
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # Synth images from opt_w.
        w_noise = torch.randn_like(w_opt) * w_noise_scale
        ws = w_opt + w_noise  # .repeat([1, G.mapping.num_ws, 1])
        synth_images = G.synthesis(ws, noise_mode="const")

        # Downsample image to 256x256 if it's larger than that. VGG was built for 224x224 images.
        synth_images = (synth_images + 1) * (255 / 2)
        if synth_images.shape[2] > 256:
            synth_images = F.interpolate(synth_images, size=(256, 256), mode="area")

        # Features for synth images.
        synth_features = vgg16(synth_images, resize_images=False, return_lpips=True)
        # synth_features = vgg16_features(synth_images, layers)
        mse = torch.nn.MSELoss(reduction="mean")
        # dist = sum(map(lambda x, y: mse(x, y), target_features, synth_features))
        dist = (target_features - synth_features).square().sum()

        # Noise regularization.
        reg_loss = 0.0
        for v in noise_bufs.values():
            noise = v[None, None, :, :]  # must be [1,1,H,W] for F.avg_pool2d()
            while True:
                reg_loss += (noise * torch.roll(noise, shifts=1, dims=3)).mean() ** 2
                reg_loss += (noise * torch.roll(noise, shifts=1, dims=2)).mean() ** 2
                if noise.shape[2] <= 8:
                    break
                noise = F.avg_pool2d(noise, kernel_size=2)
        loss = (
            dist + reg_loss * regularize_noise_weight
        )  # F.mse_loss(target_images, synth_images) ## Add MSELOSS(target,synth) here? or use HW5 code?

        # Step
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        logprint(
            f"step {step+1:>4d}/{num_steps}: dist {dist:<4.2f} loss {float(loss):<5.2f}"
        )

        # Save projected W for each optimization step.
        w_out[step] = w_opt.detach()[0]

        # Normalize noise.
        with torch.no_grad():
            for buf in noise_bufs.values():
                buf -= buf.mean()
                buf *= buf.square().mean().rsqrt()

    return w_out  # .repeat([1, G.mapping.num_ws, 1])


# ----------------------------------------------------------------------------
# default NVIDIA implementation
def project_orig(
    G,
    target: torch.Tensor,  # [C,H,W] and dynamic range [0,255], W & H must match G output resolution
    *,
    num_steps=1000,
    w_avg_samples=10000,
    initial_learning_rate=0.1,
    initial_noise_factor=0.05,
    lr_rampdown_length=0.25,
    lr_rampup_length=0.05,
    noise_ramp_length=0.75,
    regularize_noise_weight=1e5,
    verbose=False,
    device: torch.device,
):
    assert target.shape == (G.img_channels, G.img_resolution, G.img_resolution)

    def logprint(*args):
        if verbose:
            print(*args)

    G = copy.deepcopy(G).eval().requires_grad_(False).to(device)  # type: ignore

    # Compute w stats.
    logprint(f"Computing W midpoint and stddev using {w_avg_samples} samples...")
    z_samples = np.random.RandomState(123).randn(w_avg_samples, G.z_dim)
    w_samples = G.mapping(torch.from_numpy(z_samples).to(device), None)  # [N, L, C]
    w_samples = w_samples[:, :1, :].cpu().numpy().astype(np.float32)  # [N, 1, C]
    w_avg = np.mean(w_samples, axis=0, keepdims=True)  # [1, 1, C]
    w_std = (np.sum((w_samples - w_avg) ** 2) / w_avg_samples) ** 0.5

    # Setup noise inputs.
    noise_bufs = {
        name: buf
        for (name, buf) in G.synthesis.named_buffers()
        if "noise_const" in name
    }

    # Load VGG16 feature detector.
    url = "https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/vgg16.pt"
    with dnnlib.util.open_url(url) as f:
        vgg16 = torch.jit.load(f).eval().to(device)

    # Features for target image.
    target_images = target.unsqueeze(0).to(device).to(torch.float32)
    if target_images.shape[2] > 256:
        target_images = F.interpolate(target_images, size=(256, 256), mode="area")
    target_features = vgg16(target_images, resize_images=False, return_lpips=True)

    w_opt = torch.tensor(
        w_avg, dtype=torch.float32, device=device, requires_grad=True
    )  # pylint: disable=not-callable
    w_out = torch.zeros(
        [num_steps] + list(w_opt.shape[1:]), dtype=torch.float32, device=device
    )
    optimizer = torch.optim.Adam(
        [w_opt] + list(noise_bufs.values()),
        betas=(0.9, 0.999),
        lr=initial_learning_rate,
    )

    # Init noise.
    for buf in noise_bufs.values():
        buf[:] = torch.randn_like(buf)
        buf.requires_grad = True

    for step in range(num_steps):
        # Learning rate schedule.
        t = step / num_steps
        w_noise_scale = (
            w_std * initial_noise_factor * max(0.0, 1.0 - t / noise_ramp_length) ** 2
        )
        lr_ramp = min(1.0, (1.0 - t) / lr_rampdown_length)
        lr_ramp = 0.5 - 0.5 * np.cos(lr_ramp * np.pi)
        lr_ramp = lr_ramp * min(1.0, t / lr_rampup_length)
        lr = initial_learning_rate * lr_ramp
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # Synth images from opt_w.
        w_noise = torch.randn_like(w_opt) * w_noise_scale
        ws = (w_opt + w_noise).repeat([1, G.mapping.num_ws, 1])
        synth_images = G.synthesis(ws, noise_mode="const")

        # Downsample image to 256x256 if it's larger than that. VGG was built for 224x224 images.
        synth_images = (synth_images + 1) * (255 / 2)
        if synth_images.shape[2] > 256:
            synth_images = F.interpolate(synth_images, size=(256, 256), mode="area")

        # Features for synth images.
        synth_features = vgg16(synth_images, resize_images=False, return_lpips=True)
        dist = (target_features - synth_features).square().sum()

        # Noise regularization.
        reg_loss = 0.0
        for v in noise_bufs.values():
            noise = v[None, None, :, :]  # must be [1,1,H,W] for F.avg_pool2d()
            while True:
                reg_loss += (noise * torch.roll(noise, shifts=1, dims=3)).mean() ** 2
                reg_loss += (noise * torch.roll(noise, shifts=1, dims=2)).mean() ** 2
                if noise.shape[2] <= 8:
                    break
                noise = F.avg_pool2d(noise, kernel_size=2)
        loss = dist + reg_loss * regularize_noise_weight

        # Step
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        logprint(
            f"step {step+1:>4d}/{num_steps}: dist {dist:<4.2f} loss {float(loss):<5.2f}"
        )

        # Save projected W for each optimization step.
        w_out[step] = w_opt.detach()[0]

        # Normalize noise.
        with torch.no_grad():
            for buf in noise_bufs.values():
                buf -= buf.mean()
                buf *= buf.square().mean().rsqrt()

    return w_out.repeat([1, G.mapping.num_ws, 1])


# ----------------------------------------------------------------------------


@click.command()
@click.option("--network", "network_pkl", help="Network pickle filename", required=True)
@click.option(
    "--target",
    "target_fname",
    help="Target image file to project to",
    required=True,
    metavar="FILE",
)
@click.option(
    "--num-steps",
    help="Number of optimization steps",
    type=int,
    default=1000,
    show_default=True,
)
@click.option("--seed", help="Random seed", type=int, default=303, show_default=True)
@click.option(
    "--save-video",
    help="Save an mp4 video of optimization progress",
    type=bool,
    default=True,
    show_default=True,
)
@click.option(
    "--default-project",
    help="Use default NVIDIA projection method, if False, used w* project",
    type=bool,
    default=False,
    show_default=True,
)
@click.option(
    "--outdir", help="Where to save the output images", required=True, metavar="DIR"
)
def run_projection(
    network_pkl: str,
    target_fname: str,
    outdir: str,
    save_video: bool,
    seed: int,
    num_steps: int,
    default_project: bool,
):
    """Project given image to the latent space of pretrained network pickle.

    Examples:

    \b
    python projector.py --outdir=out --target=~/mytargetimg.png \\
        --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/ffhq.pkl
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Load networks.
    print('Loading networks from "%s"...' % network_pkl)
    device = torch.device("cuda")
    with dnnlib.util.open_url(network_pkl) as fp:
        G = legacy.load_network_pkl(fp)["G_ema"].requires_grad_(False).to(device)  # type: ignore

    # Load target image.
    target_pil = PIL.Image.open(target_fname).convert("RGB")
    w, h = target_pil.size
    s = min(w, h)
    target_pil = target_pil.crop(
        ((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2)
    )
    target_pil = target_pil.resize(
        (G.img_resolution, G.img_resolution), PIL.Image.LANCZOS
    )
    target_uint8 = np.array(target_pil, dtype=np.uint8)

    # Optimize projection.
    start_time = perf_counter()
    if default_project:
        projected_w_steps = project_orig(
            G,
            target=torch.tensor(
                target_uint8.transpose([2, 0, 1]), device=device
            ),  # pylint: disable=not-callable
            num_steps=num_steps,
            device=device,
            verbose=True,
        )
    else:
        projected_w_steps = project(
            G,
            target=torch.tensor(
                target_uint8.transpose([2, 0, 1]), device=device
            ),  # pylint: disable=not-callable
            num_steps=num_steps,
            device=device,
            verbose=True,
        )
    print(f"Elapsed: {(perf_counter()-start_time):.1f} s")

    # Render debug output: optional video and projected image and W vector.
    os.makedirs(outdir, exist_ok=True)
    if save_video:
        video = imageio.get_writer(
            f"{outdir}/proj.mp4", mode="I", fps=10, codec="libx264", bitrate="16M"
        )
        print(f'Saving optimization progress video "{outdir}/proj.mp4"')
        for projected_w in projected_w_steps:
            synth_image = G.synthesis(projected_w.unsqueeze(0), noise_mode="const")
            synth_image = (synth_image + 1) * (255 / 2)
            synth_image = (
                synth_image.permute(0, 2, 3, 1)
                .clamp(0, 255)
                .to(torch.uint8)[0]
                .cpu()
                .numpy()
            )
            video.append_data(np.concatenate([target_uint8, synth_image], axis=1))
        video.close()

    # Save final projected frame and W vector.
    target_pil.save(f"{outdir}/target.png")
    projected_w = projected_w_steps[-1]
    synth_image = G.synthesis(projected_w.unsqueeze(0), noise_mode="const")
    synth_image = (synth_image + 1) * (255 / 2)
    synth_image = (
        synth_image.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()
    )
    PIL.Image.fromarray(synth_image, "RGB").save(f"{outdir}/proj.png")
    np.savez(f"{outdir}/projected_w.npz", w=projected_w.unsqueeze(0).cpu().numpy())


# ----------------------------------------------------------------------------

if __name__ == "__main__":
    run_projection()  # pylint: disable=no-value-for-parameter

# ----------------------------------------------------------------------------

