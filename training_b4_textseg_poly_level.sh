export CUDA_VISIBLE_DEVICES=1
source activate segformer

./tools/dist_train.sh local_configs/segformer/B4/textseg_poly_level.py 1



