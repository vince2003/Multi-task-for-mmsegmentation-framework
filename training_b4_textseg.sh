export CUDA_VISIBLE_DEVICES=0
source activate segformer

./tools/dist_train.sh local_configs/segformer/B4/textseg.py 1



