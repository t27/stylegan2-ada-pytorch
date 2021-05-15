import streamlit as st  # pip install streamlit
import os
import use_blended_model
from align_face import align_face
import numpy as np
import torch
from streamlit.report_thread import get_report_ctx
import shutil

ctx = get_report_ctx()
print("session id", ctx.session_id)
session_id = ctx.session_id


networks = {
    "faces": "checkpoints/ffhq_chkpt_256.pkl",
    "cats": "checkpoints/cat_chkpt_256.pkl",
    "dogs": "checkpoints/dog_chkpt_256.pkl",
    "wildlife": "checkpoints/wild_chkpt_256.pkl",
    "cartoons": "checkpoints/cartoon_chkpt_256.pkl",
}

blend_layers = [4, 8, 16, 32, 64]


st.write(
    f"""
# Img Synth Project
-------------------
_Hello_, your session id is {session_id}.

The first time you load an image, it will take about a minute to do some configurations, after that you can choose different Blend Levels and see the results. You can also choose different models

------------
"""
)

"Choose an input image"

## Add option to choose from predefined images

content_file = st.file_uploader("Choose an Image", type=["png", "jpg", "jpeg"])


if not content_file:
    st.warning("Please upload or select an image")
    st.stop()


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


# @st.cache(allow_output_mutation=True)
def get_models(network_pkl1, network_pkl2, blend_layer):
    blended_model, G1 = use_blended_model.blend_model(
        network_pkl1,
        network_pkl2,
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
    if not os.path.exists(f"streamlit_temp/{session_id}/vector.npz"):
        w_plus = use_blended_model.project_image(image, G1, "cuda", pil=True)
        np.savez(
            f"streamlit_temp/{session_id}/vector.npz",
            w=w_plus.unsqueeze(0).cpu().numpy(),
        )
    else:
        ws = np.load(f"streamlit_temp/{session_id}/vector.npz")["w"]
        w_plus = torch.tensor(ws, device="cuda").squeeze(
            0
        )  # pylint: disable=not-callable
        # print(ws.shape, (G1.num_ws, G1.w_dim))
        # assert ws.shape[1:] == (G1.num_ws, G1.w_dim)
        # # for idx, w in enumerate(ws):
        # images = []
        # w = ws[-1]
    return w_plus


blended_model, G1 = get_models(
    networks[network_name1], networks[network_name2], blend_layer
)


@st.cache
def align_face_image(content_file):
    img = align_face(content_file)
    return img


@st.cache
def reset_image_cache(
    content_file, session_id
):  # hack to clear the session cache whenever we have a new image
    shutil.rmtree(f"streamlit_temp/{session_id}")
    return True


aligned_image = align_face_image(content_file)
# reset_image_cache(content_file, session_id)
os.makedirs(f"./streamlit_temp/{session_id}/", exist_ok=True)
st.write("Image alignment done!")

col1, col2, col3 = st.beta_columns(3)
col1.image(aligned_image.resize((256, 256)), caption="Aligned Image")

with st.spinner(text="Projecting image(this may take around 60seconds)..."):
    w_plus = get_image_projection(aligned_image, G1)
st.write("Image projection done!")


# generate and save the normal image
normal_img_pil = use_blended_model.generate_image(G1, w_plus)

col2.image(normal_img_pil, caption="Synthesized Image")

# generate and save the blended image
blended_img_pil = use_blended_model.generate_image(blended_model, w_plus)

col3.image(blended_img_pil, caption="Blended Image")

get_video = st.checkbox("See Video")
if get_video:
    uint8img = use_blended_model.get_target_transformed_img(
        aligned_image, G1.img_resolution, pil=True
    )

    use_blended_model.make_video(
        G1, blended_model, w_plus, uint8img, f"streamlit_temp/{session_id}.mp4"
    )
    video_bytes = open(f"streamlit_temp/{session_id}.mp4", "rb").read()
    st.video(video_bytes)

