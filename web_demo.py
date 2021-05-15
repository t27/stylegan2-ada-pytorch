import streamlit as st  # pip install streamlit
import use_blended_model
from align_face import align_face

networks = {
    "faces": "checkpoints/ffhq_chkpt_256.pkl",
    "cats": "checkpoints/cat_chkpt_256.pkl",
    "dogs": "checkpoints/dog_chkpt_256.pkl",
    "wildlife": "checkpoints/wild_chkpt_256.pkl",
    "cartoons": "checkpoints/cartoon_chkpt_256.pkl",
}

blend_layers = [4, 8, 16, 32, 64]


st.write(
    """
# Img Synth Project
-------------------
_Hello_
"""
)

blend_layer = st.radio("Choose the Level", blend_layers)
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

"Choose an input image"

## Add option to choose from predefined images

content_file = st.file_uploader("Choose an Image", type=["png", "jpg", "jpeg"])

if not content_file:
    st.warning("Please upload or select an image")
    st.stop()


# @st.cache
def get_models(network_pkl1, network_pkl2, blend_layer):
    blended_model, G1 = use_blended_model.blend_model(
        network_pkl1,
        network_pkl2,
        resolution=blend_layer,
        network_size=256,  # fixed for the above 4 chkpts
        blend_width=0.7,  # chosen for best results
    )
    return blended_model, G1


# @st.cache(allow_output_mutation=True)
def get_image_projection(image_name, G1):
    w_plus = use_blended_model.project_image(image_name, G1, "cuda", pil=True)
    return w_plus


blended_model, G1 = get_models(
    networks[network_name1], networks[network_name2], blend_layer
)


@st.cache
def align_face_image(content_file):
    img = align_face(content_file)
    return img


aligned_image = align_face_image(content_file)
st.write("Image alignment done!")

st.image(aligned_image, caption="Synthesized Image")

with st.spinner(text="Projecting image(this could take around 60seconds)..."):
    w_plus = get_image_projection(aligned_image, G1)
st.write("Image projection done!")


# generate and save the normal image
normal_img_pil = use_blended_model.generate_image(G1, w_plus)

st.image(normal_img_pil, caption="Synthesized Image")

# generate and save the blended image
blended_img_pil = use_blended_model.generate_image(blended_model, w_plus)

st.image(blended_img_pil, caption="Blended Image")

#

