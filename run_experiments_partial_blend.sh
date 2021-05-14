echo "Running Blending Experiments"
CUDA_VISIBLE_DEVICES=0
# CAT_CHKPT_PATH="training_runs/00004-afhqcat256-mirror-auto2-resumeffhq256/network-snapshot-000560.pkl"
# DOG_CHKPT_PATH="training_runs/00007-afhqdog256-mirror-auto1-resumeffhq256/network-snapshot-000640.pkl"
# WILD_CHKPT_PATH="training_runs/00009-afhqwild256-mirror-auto1-resumeffhq256/network-snapshot-000600.pkl"
# CARTOON_CHKPT_PATH="training_runs/00008-cartoon256-mirror-auto1-resumeffhq256/network-snapshot-000640.pkl"

# PRETRAINED_FFHQ="https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/transfer-learning-source-nets/ffhq-res256-mirror-paper256-noaug.pkl"


PRETRAINED_FFHQ="checkpoints/ffhq_chkpt_256.pkl"
CAT_CHKPT_PATH="checkpoints/cat_chkpt_256.pkl"
DOG_CHKPT_PATH="checkpoints/dog_chkpt_256.pkl"
WILD_CHKPT_PATH="checkpoints/wild_chkpt_256.pkl"
CARTOON_CHKPT_PATH="checkpoints/cartoon_chkpt_256.pkl"


echo "Faces -> Cats"
python stylegan_blending.py \
--network1 $PRETRAINED_FFHQ \
--network2 $CAT_CHKPT_PATH \
--outdir experiments_0p2_faces_to_cats --dim 256 --blend_width 0.2

# echo "Cats -> Faces"
# python stylegan_blending.py \
# --network2 $PRETRAINED_FFHQ \
# --network1 $CAT_CHKPT_PATH \
# --outdir experiments_0p5_cats_to_faces --dim 256 --blend_width 0.5


# echo "Faces -> Dogs"
# python stylegan_blending.py \
# --network1 $PRETRAINED_FFHQ \
# --network2 $DOG_CHKPT_PATH \
# --outdir experiments_0p5_faces_to_dogs --dim 256 --blend_width 0.5

# echo "Dogs -> Faces"
# python stylegan_blending.py \
# --network2 $PRETRAINED_FFHQ \
# --network1 $DOG_CHKPT_PATH \
# --outdir experiments_0p5_dogs_to_faces --dim 256 --blend_width 0.5


# echo "Faces -> Wildlife"
# python stylegan_blending.py \
# --network1 $PRETRAINED_FFHQ \
# --network2 $WILD_CHKPT_PATH \
# --outdir experiments_0p5_faces_to_wild --dim 256 --blend_width 0.5

# echo "Wildlife -> Faces"
# python stylegan_blending.py \
# --network2 $PRETRAINED_FFHQ \
# --network1 $WILD_CHKPT_PATH \
# --outdir experiments_0p5_wild_to_faces --dim 256 --blend_width 0.5

# echo "Faces -> CartoonFace"
# python stylegan_blending.py \
# --network1 $PRETRAINED_FFHQ \
# --network2 $CARTOON_CHKPT_PATH \
# --outdir experiments_0p5_faces_to_cartoon --dim 256 --blend_width 0.5

# echo "CartoonFace -> Faces"
# python stylegan_blending.py \
# --network2 $PRETRAINED_FFHQ \
# --network1 $CARTOON_CHKPT_PATH \
# --outdir experiments_0p5_cartoon_to_faces --dim 256 --blend_width 0.5


# echo "Dogs -> Cats"
# python stylegan_blending.py \
# --network1 $DOG_CHKPT_PATH \
# --network2 $CAT_CHKPT_PATH \
# --outdir experiments_0p5_dogs_to_cats --dim 256 --blend_width 0.5


# echo "Wildlife -> Cats"
# python stylegan_blending.py \
# --network1 $WILD_CHKPT_PATH \
# --network2 $CAT_CHKPT_PATH \
# --outdir experiments_0p5_wild_to_cats --dim 256 --blend_width 0.5

# echo "Cartoons -> Cats"
# python stylegan_blending.py \
# --network1 $CARTOON_CHKPT_PATH \
# --network2 $CAT_CHKPT_PATH \
# --outdir experiments_0p5_cartoons_to_cats --dim 256 --blend_width 0.5
