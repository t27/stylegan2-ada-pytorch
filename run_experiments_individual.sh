echo "Running Blending Experiments"
PRETRAINED_FFHQ="checkpoints/ffhq_chkpt_256.pkl"
CAT_CHKPT_PATH="checkpoints/cat_chkpt_256.pkl"
DOG_CHKPT_PATH="checkpoints/dog_chkpt_256.pkl"
WILD_CHKPT_PATH="checkpoints/wild_chkpt_256.pkl"
CARTOON_CHKPT_PATH="checkpoints/cartoon_chkpt_256.pkl"


echo "Rohan"
python stylegan_blending.py \
--network1 $PRETRAINED_FFHQ \
--network2 $DOG_CHKPT_PATH \
--outdir experiments_rohan --dim 256 --projected_w ./individuals/rohan_projected_w.npz

echo "junyan"
python stylegan_blending.py \
--network1 $PRETRAINED_FFHQ \
--network2 $DOG_CHKPT_PATH \
--outdir experiments_junyan --dim 256 --projected_w ./individuals/junyan_projected_w.npz


echo "tarang"
python stylegan_blending.py \
--network1 $PRETRAINED_FFHQ \
--network2 $DOG_CHKPT_PATH \
--outdir experiments_tarang --dim 256 --projected_w ./individuals/tarang_projected_w.npz

echo "sanil"
python stylegan_blending.py \
--network1 $PRETRAINED_FFHQ \
--network2 $DOG_CHKPT_PATH \
--outdir experiments_sanil --dim 256 --projected_w ./individuals/sanil_projected_w.npz

echo "viraj"
python stylegan_blending.py \
--network1 $PRETRAINED_FFHQ \
--network2 $DOG_CHKPT_PATH \
--outdir experiments_viraj --dim 256 --projected_w ./individuals/viraj_projected_w.npz

echo "yufei"
python stylegan_blending.py \
--network1 $PRETRAINED_FFHQ \
--network2 $DOG_CHKPT_PATH \
--outdir experiments_yufei --dim 256 --projected_w ./individuals/yufei_projected_w.npz

