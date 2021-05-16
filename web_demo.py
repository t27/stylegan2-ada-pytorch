import streamlit as st  # pip install streamlit
import os
import use_blended_model
from align_face import align_face
import numpy as np
import torch
from streamlit.report_thread import get_report_ctx
import shutil
import subprocess

### RISKY!, but solves the problem for tomorrow's demo
# os.system(
#     "find `realpath streamlit_temp` -mmin +120 -delete"
# )  # delete all files in the session older than the past two hours


@st.cache(allow_output_mutation=True)
def load_network(file):
    return torch.load(file)


@st.cache(
    allow_output_mutation=True,
    hash_funcs={
        torch.nn.parameter.Parameter: lambda _: None,
        torch.Tensor: lambda _: None,
    },
)
def get_models(network_name1, network_name2, blend_layer):
    blended_model, G1 = use_blended_model.blend_model_simple(
        network_loaded[network_name1],
        network_loaded[network_name2],
        resolution=blend_layer,
        network_size=256,  # fixed for the above 4 chkpts
        blend_width=0.7,  # chosen for best results
    )
    return blended_model, G1


@st.cache(
    allow_output_mutation=True,
    hash_funcs={
        torch.nn.parameter.Parameter: lambda _: None,
        torch.Tensor: lambda _: None,
    },
)
def get_image_projection(image, G1):
    # if not os.path.exists(f"streamlit_temp/{session_id}/vector.npz"):
    w_plus = use_blended_model.project_image(image, G1, "cuda", pil=True)
    np.savez(
        f"streamlit_temp/{session_id}/vector.npz", w=w_plus.unsqueeze(0).cpu().numpy(),
    )
    # else:
    #     ws = np.load(f"streamlit_temp/{session_id}/vector.npz")["w"]
    #     w_plus = torch.tensor(ws, device="cuda").squeeze(
    #         0
    #     )  # pylint: disable=not-callable
    #     # print(ws.shape, (G1.num_ws, G1.w_dim))
    #     # assert ws.shape[1:] == (G1.num_ws, G1.w_dim)
    #     # # for idx, w in enumerate(ws):
    # images = []
    # w = ws[-1]
    return w_plus


@st.cache
def align_face_image(content_file):
    img = align_face(content_file)
    return img


ctx = get_report_ctx()
print("session id", ctx.session_id)
session_id = ctx.session_id


networks = {
    "faces": "checkpoints/faces.pt",
    "cats": "checkpoints/cats.pt",
    "dogs": "checkpoints/dogs.pt",
    "wildlife": "checkpoints/wildlife.pt",
    "cartoons": "checkpoints/cartoons.pt",
}

network_loaded = {k: load_network(v) for (k, v) in networks.items()}

blend_layers = [4, 8, 16, 32, 64]


st.write(
    f"""
# StyleGAN Blending
 [Tarang Shah(tarangs)](github.com/t27/), [Rohan Rao(rgrao)](github.com/themathgeek13)

## CMU 16726 Learning Based Image Synthesis

## Final Project - Spring 2021
--------------------------------

This website is the demo for our final project done at CMU in May 2021 for the above course.
Our project report describing our work is available [here](to_add_link).

-------------------
_Hello_, your session id is `{session_id}`.

> The first time you load an image, it will take about a minute to do some configurations, 
> after that you can choose between different Blend Levels and Models and see the results. You can also choose different models

Note: All images are deleted after you close the tab, cached data cleared 2 hours after you open this webpage


------------
"""
)

"Choose an input image(Remember to have only one face in the image)"

## Add option to choose from predefined images

content_file = st.file_uploader("Choose an Image", type=["png", "jpg", "jpeg"],)


if not content_file:
    st.warning("Please upload or select an image")
    st.stop()


"""


Tips on choosing a _blend level_

> For *Cats*, *Dogs* and *Wildlife*, you can get the best results with blend layer **4, 8, 16**

> For *Cartoons* for best results choose **32, 64**

_P.S. For scary results, choose 32, 64 with Cats, Dogs, Wildlife **(Strictly at your own risk)**_
"""

blend_layer = st.radio("Choose the Blend Level", blend_layers)

# network1 = st.selectbox(
#     "Choose the base network(Choose faces if you are uploading a face image)",
#     list(networks.keys()),
#     format_func=str.title,
# )
network_name1 = "faces"
# print(network1)
st.write("Base Network:", network_name1)

other_keys = list(networks.keys())
other_keys.remove(network_name1)

network_name2 = st.selectbox(
    "Choose the blending network", list(other_keys), format_func=str.title,
)
# print(network2)


network1 = load_network(networks[network_name1])
network2 = load_network(networks[network_name2])
blended_model, G1 = get_models(network_name1, network_name2, blend_layer)

aligned_image = align_face_image(content_file)
# reset_image_cache(content_file, session_id)
os.makedirs(f"./streamlit_temp/{session_id}/", exist_ok=True)
st.write("Image alignment done!")

col1, col2, col3 = st.beta_columns(3)
col1.image(aligned_image.resize((256, 256)), caption="Aligned Image")

with st.spinner(
    text="Processing image(Projecting to StyleGAN's latent space)(this may take around 60seconds)..."
):
    w_plus = get_image_projection(aligned_image, G1)
st.write("Image projection done!")


# generate and save the normal image
normal_img_pil = use_blended_model.generate_image(G1, w_plus)

col2.image(normal_img_pil, caption="Synthesized Image")

# generate and save the blended image
blended_img_pil = use_blended_model.generate_image(blended_model, w_plus)

col3.image(blended_img_pil, caption="Blended Image")

# get_video = st.checkbox("See Video")

if st.button("See Video"):
    with st.spinner(text="Generating Video, this may take around 20-30 seconds"):
        uint8img = use_blended_model.get_target_transformed_img(
            aligned_image, G1.img_resolution, pil=True
        )

        use_blended_model.make_video(
            G1, blended_model, w_plus, uint8img, f"streamlit_temp/{session_id}.mp4"
        )
    video_bytes = open(f"streamlit_temp/{session_id}.mp4", "rb").read()
    st.video(video_bytes)


"Project source available at https://github.com/t27/stylegan2-blending"
