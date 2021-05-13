
#pretrained ffhq and ahqcats
python stylegan_blending.py --network1 pretrained/ffhq-res512-mirror-stylegan2-noaug.pkl --network2 network-snapshot-000080.pkl --outdir out_blend

# 256
python stylegan_blending.py --network1 pretrained/ffhq-res256-mirror-paper256-noaug.pkl --network2 00004-afhqcat256-mirror-auto2-resumeffhq256/network-snapshot-000560.pkl --outdir out_custom --dim 256



# TRaining

CUDA_VISIBLE_DEVICES=0 python train.py --outdir ./training_runs --data ./datasets/afhqdog256.zip --gpus 1 --mirror 1 --resume=ffhq256 --snap=10

CUDA_VISIBLE_DEVICES=3 python train.py --outdir ./training_runs --data ./datasets/cartoon256.zip --gpus 1 --mirror 1 --resume=ffhq256 --snap=10

CUDA_VISIBLE_DEVICES=1 python train.py --outdir ./training_runs --data ./datasets/afhqwild256.zip --gpus 1 --mirror 1 --resume=ffhq256 --snap=10



