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

# Based on the Original Tensorflow implementation here https://github.com/justinpinkney/stylegan2/blob/master/blend_models.py
# modified by Tarang Shah


def num_range(s: str) -> List[int]:
    """Accept either a comma separated list of numbers 'a,b,c' or a range 'a-c' and return as a list of ints."""

    range_re = re.compile(r"^(\d+)-(\d+)$")
    m = range_re.match(s)
    if m:
        return list(range(int(m.group(1)), int(m.group(2)) + 1))
    vals = s.split(",")
    return [int(x) for x in vals]


def get_conv_names(model, max_dim=1024):
    val = 1 + int(np.log2(max_dim / 4))  # 9 for 1024, 8 for 512
    resolutions = [4 * 2 ** x for x in range(val)]
    names = [x[0] for x in list(model.named_parameters())]
    level_names = [["conv0", "const"], ["conv1", "torgb"]]
    position = 0
    conv_names = []
    for res in resolutions:
        rootname = f"synthesis.b{res}."
        for level, level_suffixes in enumerate(level_names):
            for suffix in level_suffixes:
                searchname = rootname + suffix
                matches = [x for x in names if x.startswith(searchname)]
                info_tuples = [(name, f"b{res}", level, position) for name in matches]
                conv_names.extend(info_tuples)
            position += 1
    return conv_names


def get_blended_model(G1, G2, resolution, level, blend_width=None, verbose=False):
    model1_names = get_conv_names(G1, 512)
    model2_names = get_conv_names(G2, 512)
    assert all((x == y for x, y in zip(model1_names, model2_names)))

    output_model = copy.deepcopy(G1)

    short_names = [(x[1:3]) for x in model1_names]
    full_names = [(x[0]) for x in model1_names]
    mid_point_idx = short_names.index((resolution, level))
    mid_point_pos = model1_names[mid_point_idx][3]

    ys = []

    for name, resolution, level, position in model1_names:
        x = position - mid_point_pos
        if blend_width:
            exponent = -x / blend_width
            y = 1 / (1 + math.exp(exponent))
        else:
            y = 1 if x > 1 else 0
        ys.append(y)
        if verbose:
            print(f"Blending {name} by {y}")

    new_model_state_dict = output_model.state_dict()
    for name, y in zip(full_names, ys):
        new_model_state_dict[name] = G2.state_dict()[name] * y + G2.state_dict()[
            name
        ] * (1 - y)
    output_model.load_state_dict(new_model_state_dict)

    return output_model


def get_image(model, z, label=0, truncation_psi=0.7, noise_mode="const"):
    img = model(z, label, truncation_psi=truncation_psi, noise_mode=noise_mode)
    img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
    imgfile = PIL.Image.fromarray(img[0].cpu().numpy(), "RGB")
    return imgfile
    # imgfile.save(f"Blended_seed{seed:04d}.png")


def run_blend_images(
    network_pkl1: str,
    network_pkl2: str,
    seeds: Optional[List[int]] = [700, 701, 702],
    outdir: str = "./out_blend",
    truncation_psi: float = 0.7,
    noise_mode: str = "const",
    blending_layers: List[int] = [4, 8, 256, 512],
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

    os.makedirs(outdir, exist_ok=True)

    blend_width = (
        0.9  # # None = hard switch, float = smooth switch (logistic) with given width
    )
    level = 1

    for blending_layer in blending_layers:
        resolution = f"b{blending_layer}"  # blend at layer

        blended_model = get_blended_model(
            G1, G2, resolution, level, blend_width, verbose=verbose
        )

        # seed = 279
        for seed in seeds:
            z_vector = z = torch.from_numpy(
                np.random.RandomState(seed).randn(1, G1.z_dim)
            ).to(device)
            orig1 = get_image(G1, z_vector)
            orig2 = get_image(G2, z_vector)
            blended = get_image(blended_model, z_vector)
            fprefix = f"seed_{seed}_layer_{resolution}"
            orig1.save(f"{outdir}/seed_{seed}_G1.png")
            orig2.save(f"{outdir}/seed_{seed}_G2.png")
            blended.save(f"{outdir}/{fprefix}_blended.png")


@click.command()
@click.pass_context
@click.option(
    "--network1", "network_pkl1", help="Network pickle filename", required=True
)
@click.option(
    "--network2", "network_pkl2", help="Network pickle filename", required=True
)
@click.option("--seeds", type=num_range, help="List of random seeds")
@click.option(
    "--trunc",
    "truncation_psi",
    type=float,
    help="Truncation psi",
    default=1,
    show_default=True,
)
@click.option(
    "--class",
    "class_idx",
    type=int,
    help="Class label (unconditional if not specified)",
)
@click.option(
    "--noise-mode",
    help="Noise mode",
    type=click.Choice(["const", "random", "none"]),
    default="const",
    show_default=True,
)
@click.option("--projected-w", help="Projection result file", type=str, metavar="FILE")
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
def generate_images(
    ctx: click.Context,
    network_pkl1: str,
    network_pkl2: str,
    seeds: Optional[List[int]],
    truncation_psi: float,
    noise_mode: str,
    outdir: str,
    class_idx: Optional[int],
    projected_w: Optional[str],
    verbose: bool,
):
    """Generate images using pretrained network pickle.

    Examples:
    # Generate curated MetFaces images without truncation (Fig.10 left)
    python generate.py --outdir=out --trunc=1 --seeds=85,265,297,849 \\
        --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metfaces.pkl
    """
    if seeds is None:
        seeds = [701, 702, 703]
    run_blend_images(
        network_pkl1, network_pkl2, seeds, outdir, truncation_psi, noise_mode
    )
    # print(f"Loading networks from {network_pkl1} and {network_pkl2} ...")
    # device = torch.device("cuda")
    # with dnnlib.util.open_url(network_pkl1) as f:
    #     G1 = legacy.load_network_pkl(f)["G_ema"].to(device)  # type: ignore
    # with dnnlib.util.open_url(network_pkl2) as f:
    #     G2 = legacy.load_network_pkl(f)["G_ema"].to(device)  # type: ignore

    # os.makedirs(outdir, exist_ok=True)

    # # Hyperparams
    # BLENDING_LAYER = "b64"  # anything from b4 to b512 (doubles)
    # blend_width = (
    #     None  # # None = hard switch, float = smooth switch (logistic) with given width
    # )
    # resolution = BLENDING_LAYER  # blend at layer
    # level = 0
    # verbose = True

    # blended_model = get_blended_model(
    #     G1, G2, resolution, level, blend_width, verbose=True
    # )

    # seed = 279
    # z_vector = z = torch.from_numpy(np.random.RandomState(seed).randn(1, G1.z_dim)).to(
    #     device
    # )
    # orig1 = get_image(G1, z_vector)
    # orig2 = get_image(G2, z_vector)
    # blended = get_image(blended_model, z_vector)

    # # Synthesize the result of a W projection.
    # if projected_w is not None:
    #     if seeds is not None:
    #         print("warn: --seeds is ignored when using --projected-w")
    #     print(f'Generating images from projected W "{projected_w}"')
    #     ws = np.load(projected_w)["w"]
    #     ws = torch.tensor(ws, device=device)  # pylint: disable=not-callable
    #     assert ws.shape[1:] == (G.num_ws, G.w_dim)
    #     for idx, w in enumerate(ws):
    #         img = G.synthesis(w.unsqueeze(0), noise_mode=noise_mode)
    #         img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
    #         img = PIL.Image.fromarray(img[0].cpu().numpy(), "RGB").save(
    #             f"{outdir}/proj{idx:02d}.png"
    #         )
    #     return

    # if seeds is None:
    #     ctx.fail("--seeds option is required when not using --projected-w")

    # # Labels.
    # label = torch.zeros([1, G.c_dim], device=device)
    # if G.c_dim != 0:
    #     if class_idx is None:
    #         ctx.fail(
    #             "Must specify class label with --class when using a conditional network"
    #         )
    #     label[:, class_idx] = 1
    # else:
    #     if class_idx is not None:
    #         print("warn: --class=lbl ignored when running on an unconditional network")

    # # Generate images.
    # for seed_idx, seed in enumerate(seeds):
    #     print("Generating image for seed %d (%d/%d) ..." % (seed, seed_idx, len(seeds)))
    #     z = torch.from_numpy(np.random.RandomState(seed).randn(1, G.z_dim)).to(device)
    #     img = G(z, label, truncation_psi=truncation_psi, noise_mode=noise_mode)
    #     img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
    #     PIL.Image.fromarray(img[0].cpu().numpy(), "RGB").save(
    #         f"{outdir}/seed{seed:04d}.png"
    #     )


# ----------------------------------------------------------------------------

if __name__ == "__main__":
    generate_images()  # pylint: disable=no-value-for-parameter

# ----------------------------------------------------------------------------
