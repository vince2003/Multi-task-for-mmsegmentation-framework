export CUDA_VISIBLE_DEVICES=0
source activate segformer
python tools/print_config.py local_configs/segformer/B4/segformer.b4.64x64.DRIVE.20k.py #--options model.norm_cfg = 'BN'
