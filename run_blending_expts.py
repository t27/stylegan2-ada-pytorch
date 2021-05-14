import subprocess
import sys
import os
from tqdm import tqdm

images = [
    "individual/rohan.jpg",
    "individual/tarang.png",
    "individual/viraj.jpg",
    "individual/junyan.jpg",
    "individual/yufei.jpg",
    "individual/sanil.jpg",
]
networks = [
    "checkpoints/ffhq_chkpt_256.pkl",
    "checkpoints/cat_chkpt_256.pkl",
    "checkpoints/dog_chkpt_256.pkl",
    "checkpoints/wild_chkpt_256.pkl",
    "checkpoints/cartoon_chkpt_256.pkl",
]

blend_layers = [4, 16, 32]


total = len(blend_layers) * len(images) * (len(networks) - 1)

pbar = tqdm(total=total)
for img in images:
    for blend in blend_layers:
        N1 = networks[0]
        for N2 in networks:
            if N1 != N2:
                imgname = os.path.splitext(os.path.split(img)[-1])[0]
                fromname = N1.split("-")[1]
                toname = N2.split("-")[1]
                cmd = "python use_blended_model.py --network1 " + N1
                cmd += " --network2 " + N2
                cmd += " --input_image " + img
                cmd += f" --dim 256 --outdir experiment_0p7blending_{imgname}_{fromname}_{toname}_{blend}"
                cmd += " --blend_layer " + str(blend)
                cmd += " --blend_width 0.7"
                print("Running cmd: ", cmd)
                subprocess.check_call(
                    cmd, shell=True, stdout=sys.stdout, stderr=subprocess.STDOUT
                )
                pbar.update(1)
pbar.close()
