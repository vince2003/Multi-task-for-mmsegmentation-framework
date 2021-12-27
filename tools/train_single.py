import argparse

import os
import pdb

# Cur_work_dir_moi='/media/vinh/17616758-c225-41d8-8023-4e795947f4ad/segmentation_by_transformer/SegFormer/SegFormer'
# os.chdir(Cur_work_dir_moi)

from mmseg.datasets import build_dataset
from mmseg.models import build_segmentor
from mmseg.apis import train_segmentor
from mmcv import Config




def main():

    #args = parse_args()
    
    pdb.set_trace()

    cfg = Config.fromfile('local_configs/segformer/B4/segformer.b4.64x64.DRIVE.20k.py')

    cfg.seed = 0
    cfg.norm_cfg= dict(type='BN', requires_grad=True)
   

    cfg.work_dir = './work_dirs/thu'
    
    





    model = build_segmentor(
        cfg.model,
        train_cfg=cfg.get('train_cfg'),
        test_cfg=cfg.get('test_cfg'))



    datasets = [build_dataset(cfg.data.train)]


    # add an attribute for visualization convenience
    model.CLASSES = datasets[0].CLASSES
    train_segmentor(
        model,
        datasets,
        cfg,
        distributed=False,
        validate=True)


if __name__ == '__main__':
    main()
