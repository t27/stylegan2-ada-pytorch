import subprocess
import sys

images = ["rohan.jpg", "tarang.png", "viraj.jpg", "junyan.jpg", "yufei.jpg", "sanil.jpg"]
networks = ["ffhq-res256-mirror-paper256-noaug.pkl", "cat-network-snapshot-000560.pkl", "dog-network-snapshot-000400.pkl", "wild-network-snapshot-000400.pkl"]
blend_layers = [4, 16, 32]

for img in images:
    for blend in blend_layers:
        for N1 in networks:
            for N2 in networks:
                if N1!=N2:
                    cmd = "python use_blended_model.py --network1 " + N1
                    cmd += " --network2 " + N2
                    cmd += " --input_image " + img
                    cmd += " --dim 256 --outdir experiment_blending_" + img.split(".")[0]
                    cmd += " --blend_layer " + str(blend)
                    print("Running cmd: ", cmd)
                    subprocess.check_call(cmd, shell=True, stdout=sys.stdout, stderr=subprocess.STDOUT)

