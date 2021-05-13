echo "Running Blending Experiments"

CAT_CHKPT_PATH="00004-afhqcat256-mirror-auto2-resumeffhq256/network-snapshot-000560.pkl"
DOG_CHKPT_PATH="00007-afhqdog256-mirror-auto1-resumeffhq256/network-snapshot-000400.pkl"
WILD_CHKPT_PATH="00009-afhqwild256-mirror-auto1-resumeffhq256/network-snapshot-000400.pkl"
CARTOON_CHKPT_PATH="00008-cartoon256-mirror-auto1-resumeffhq256/network-snapshot-000400.pkl"


echo "Faces -> Cats"
python stylegan_blending.py \
--network1 pretrained/ffhq-res256-mirror-paper256-noaug.pkl \
--network2 $CAT_CHKPT_PATH \
--outdir experiments_faces_to_cats --dim 256

echo "Cats -> Faces"
python stylegan_blending.py \
--network2 pretrained/ffhq-res256-mirror-paper256-noaug.pkl \
--network1 $CAT_CHKPT_PATH \
--outdir experiments_cats_to_faces --dim 256


echo "Faces -> Dogs"
python stylegan_blending.py \
--network1 pretrained/ffhq-res256-mirror-paper256-noaug.pkl \
--network2 $DOG_CHKPT_PATH \
--outdir experiments_faces_to_dogs --dim 256

echo "Dogs -> Faces"
python stylegan_blending.py \
--network2 pretrained/ffhq-res256-mirror-paper256-noaug.pkl \
--network1 $DOG_CHKPT_PATH \
--outdir experiments_dogs_to_faces --dim 256


echo "Faces -> Wildlife"
python stylegan_blending.py \
--network1 pretrained/ffhq-res256-mirror-paper256-noaug.pkl \
--network2 $WILD_CHKPT_PATH \
--outdir experiments_faces_to_wild --dim 256

echo "Wildlife -> Faces"
python stylegan_blending.py \
--network1 pretrained/ffhq-res256-mirror-paper256-noaug.pkl \
--network2 $WILD_CHKPT_PATH \
--outdir experiments_wild_to_faces --dim 256

echo "Faces -> CartoonFace"
python stylegan_blending.py \
--network1 pretrained/ffhq-res256-mirror-paper256-noaug.pkl \
--network2 $CARTOON_CHKPT_PATH \
--outdir experiments_faces_to_cartoon --dim 256

echo "CartoonFace -> Faces"
python stylegan_blending.py \
--network1 pretrained/ffhq-res256-mirror-paper256-noaug.pkl \
--network2 $CARTOON_CHKPT_PATH \
--outdir experiments_cartoon_to_faces --dim 256