_base_ = [
    '../../_base_/models/segformer.py',
    #'../../_base_/datasets/textseg_repeat.py',
    '../../_base_/default_runtime.py',
    '../../_base_/schedules/schedule_160k_adamw.py'
]

#-----------model--------------------
# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)
find_unused_parameters = True
model = dict(
    type='EncoderDecoder',
    pretrained='pretrained/mit_b5.pth',
    backbone=dict(
        type='mit_b5',
        style='pytorch'),
    decode_head=dict(
        type='SegFormerHead',
        #in_channels=[64, 128, 320, 512],
        #in_index=[0, 1, 2, 3],
        #feature_strides=[4, 8, 16, 32],
        channels=128,
        #dropout_ratio=0.1,
        num_classes=2,
        #norm_cfg=norm_cfg,
        #align_corners=False,
        #decoder_params=dict(embed_dim=768),
        loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=False,loss_weight=1.0),
        sampler=dict(type='OHEMPixelSampler', thresh=0.7, min_kept=100000)),

    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))#(mode="slide", crop_size=(512, 512), stride=(256, 256))

#-------dataset-------
# dataset settings
dataset_type = 'TextSegDataset'
data_root = 'data/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size = (512, 512)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=(512, 512), ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg1', 'gt_semantic_seg2']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(512, 512),
        img_ratios=[0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5],
        flip=False,
        transforms=[
            dict(type='AlignedResize', keep_ratio=True, size_divisor=32), # Ensure the long and short sides are divisible by 32
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=0,
    train=dict(
        type='RepeatDataset',
        times=50,
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            img_dir=['text_seg_debug/images/training'],
            ann_dir1='text_seg_debug/annotations/training_convert/segmentation',
            ann_dir2='text_seg_debug/annotations/training_convert/skeleton',
            pipeline=train_pipeline)),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='text_seg_debug/images/validation',
        ann_dir1='text_seg_debug/annotations/validation_convert/segmentation',
        ann_dir2='text_seg_debug/annotations/validation_convert/skeleton',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='text_seg_debug/images/validation',
        ann_dir1='text_seg_debug/annotations/validation_convert/segmentation',
        ann_dir2='text_seg_debug/annotations/validation_convert/skeleton',
        pipeline=test_pipeline))
#---------------------    
    


#-------Schedule----------
optimizer_config = dict()
# runtime settings
runner = dict(type='IterBasedRunner', max_iters=500)
checkpoint_config = dict(by_epoch=False, interval=500)

# optimizer
optimizer = dict(_delete_=True, type='AdamW', lr=0.000005, betas=(0.9, 0.999), weight_decay=0.01,
                 paramwise_cfg=dict(custom_keys={'pos_block': dict(decay_mult=0.),
                                                 'norm': dict(decay_mult=0.),
                                                 'head': dict(lr_mult=10.)
                                                 }))

lr_config = dict(_delete_=True, policy='poly',
                 warmup='linear',
                 warmup_iters=1500,
                 warmup_ratio=1e-6,
                 power=1.0, min_lr=0.0, by_epoch=False)

evaluation = dict(interval=100, metric=['mIoU', 'mDice'])
#------------------------

#------runtime---------
# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
        #dict(type='WandbLoggerHook', cfg_name='./local_configs/segformer/B4/{{ fileBasename }}' ,init_kwargs=dict(project='segformer_skeleton_FST', name='train_dung_b5_co_sampler', save_code=True))
    ])
# yapf:enable
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None#
resume_from = None#'./work_dirs/v9/latest.pth'
workflow = [('train', 1)]
cudnn_benchmark = True

#-------------------------
#from mmcv import Config
#cfg=Config.fromfile('./local_configs/segformer/B4/segformer.b4.512x512.Textseg.160k_testmulscale_v3.py')


#import pdb
#pdb.set_trace()


