import os
import re
from typing import List, Optional

import click
import dnnlib
import numpy as np
import PIL.Image
import torch
import copy
import math
import legacy


from stylegan_blending import get_blended_model
from projector import project

import imageio


def blend_model(
    network_pkl1: str,
    network_pkl2: str,
    resolution: int = 16,
    network_size: int = 512,
    blend_width: float = None,
    verbose: bool = False,
):
    """Generate images using pretrained network pickle.

    Examples:
    # Generate curated MetFaces images without truncation (Fig.10 left)
    python generate.py --outdir=out --trunc=1 --seeds=85,265,297,849 \\
        --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metfaces.pkl
    """

    print(f"Loading networks from {network_pkl1} and {network_pkl2} ...")
    device = torch.device("cuda")
    with dnnlib.util.open_url(network_pkl1) as f:
        G1 = legacy.load_network_pkl(f)["G_ema"].to(device)  # type: ignore
    with dnnlib.util.open_url(network_pkl2) as f:
        G2 = legacy.load_network_pkl(f)["G_ema"].to(device)  # type: ignore
    # None = hard switch, float = smooth switch (logistic) with given width
    blend_width = None
    level = 0
    resolution = f"b{resolution}"  # blend at layer

    blended_model = get_blended_model(
        G1,
        G2,
        resolution,
        level,
        blend_width,
        network_size=network_size,
        verbose=verbose,
    )
    return blended_model, G1


@click.command()
@click.pass_context
@click.option(
    "--network1", "network_pkl1", help="Network pickle filename", required=True
)
@click.option(
    "--network2", "network_pkl2", help="Network pickle filename", required=True
)
@click.option(
    "--input_image", "input_image", help="Input Image filename", required=True
)
@click.option(
    "--dim", "network_size", type=int, help="Network max dimension", default=512
)
@click.option(
    "--blend_layer",
    "blend_layer",
    type=int,
    help="Layer at which we should blend at",
    default=32,
)
@click.option(
    "--trunc",
    "truncation_psi",
    type=float,
    help="Truncation psi",
    default=1,
    show_default=True,
)
@click.option(
    "--noise-mode",
    help="Noise mode",
    type=click.Choice(["const", "random", "none"]),
    default="const",
    show_default=True,
)
@click.option(
    "--outdir",
    help="Where to save the output images",
    type=str,
    required=True,
    metavar="DIR",
)
@click.option(
    "--verbose",
    "verbose",
    type=bool,
    help="Verbose printing",
    default=False,
    show_default=True,
)
def main(
    ctx: click.Context,
    network_pkl1: str,
    network_pkl2: str,
    network_size: int,
    input_image: str,
    blend_layer: int,
    truncation_psi: float,
    noise_mode: str,
    outdir: str,
    verbose: bool,
):
    device = "cuda"
    # Take input image
    # Get embedding for the image after K iterations on G1 (K=500?)
    # TODO. may be use a projector NN that approximates the transform and use that vector as initialization and only do 20 steps instead of 500
    # Use the embedding and run it through both the initial and blended model

    input_image_name, ext = os.path.splitext((os.path.split(input_image)[-1]))

    blended_model, G1 = blend_model(
        network_pkl1, network_pkl2, resolution=blend_layer, network_size=network_size
    )
    target_pil = PIL.Image.open(input_image).convert("RGB")
    w, h = target_pil.size
    s = min(w, h)
    target_pil = target_pil.crop(
        ((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2)
    )
    target_pil = target_pil.resize(
        (G1.img_resolution, G1.img_resolution), PIL.Image.LANCZOS
    )
    target_uint8 = np.array(target_pil, dtype=np.uint8)
    target_torch = torch.tensor(target_uint8.transpose([2, 0, 1]), device=device)

    # since the project function returns all the w's, we only want the last one
    w_plus = project(G1, target_torch, num_steps=500, device=device, verbose=True)

    # generate and save the normal image
    normal_img = G1.synthesis(w_plus[-1].unsqueeze(0), noise_mode="const")
    normal_img = (normal_img + 1) * (255 / 2)
    normal_img = (
        normal_img.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()
    )
    normal_img_pil = PIL.Image.fromarray(normal_img, "RGB")
    normal_img_pil.save(f"{outdir}/{input_image_name}_synthesized{ext}")

    # generate and save the blended image
    blended_img = blended_model.synthesis(w_plus[-1].unsqueeze(0), noise_mode="const")
    blended_img = (blended_img + 1) * (255 / 2)
    blended_img = (
        blended_img.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()
    )
    blended_img_pil = PIL.Image.fromarray(blended_img, "RGB")
    blended_img_pil.save(f"{outdir}/{input_image_name}_blended{ext}")

    video = imageio.get_writer(
        f"{outdir}/proj_blended.mp4", mode="I", fps=10, codec="libx264", bitrate="16M"
    )
    print(f'Saving optimization progress video "{outdir}/proj_blended.mp4"')
    for projected_w in w_plus:
        unblended_image = G1.synthesis(projected_w.unsqueeze(0), noise_mode="const")
        unblended_image = (unblended_image + 1) * (255 / 2)
        unblended_image = (
            unblended_image.permute(0, 2, 3, 1)
            .clamp(0, 255)
            .to(torch.uint8)[0]
            .cpu()
            .numpy()
        )
        synth_image = blended_model.synthesis(
            projected_w.unsqueeze(0), noise_mode="const"
        )
        synth_image = (synth_image + 1) * (255 / 2)
        synth_image = (
            synth_image.permute(0, 2, 3, 1)
            .clamp(0, 255)
            .to(torch.uint8)[0]
            .cpu()
            .numpy()
        )
        video.append_data(
            np.concatenate([target_uint8, unblended_image, synth_image], axis=1)
        )
    video.close()


if __name__ == "__main__":
    main()
