export CUDA_VISIBLE_DEVICES=1
source activate segformer
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python tools/test.py\
 local_configs/segformer/B4/segformer.b4.512x512.Textseg.160k_testmulscale_v4.py\
  ./work_dirs/segformer.b4.512x512.Textseg.160k_testmulscale_v4/latest.pth\
   --eval mIoU\
    --eval-options efficient_test=True\
     --show-dir Textseg_results\

