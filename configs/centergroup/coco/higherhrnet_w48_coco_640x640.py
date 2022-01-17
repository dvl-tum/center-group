_base_ = '../base.py'

# Accounts for larger resolution
data_cfg = dict(
    image_size=640,
    base_size=320,
    base_sigma=2,
    heatmap_size=[160, 320, 640],
    num_joints=18)

# Accounts for larger backbone
bu_model = dict(
    pretrained='https://download.openmmlab.com/mmpose/'
    'pretrain_models/hrnet_w48-8ef0771d.pth',
    backbone=dict(
        extra=dict(
            stage1=dict(
                num_modules=1,
                num_branches=1,
                block='BOTTLENECK',
                num_blocks=(4, ),
                num_channels=(64, )),
            stage2=dict(
                num_modules=1,
                num_branches=2,
                block='BASIC',
                num_blocks=(4, 4),
                num_channels=(48, 96)),
            stage3=dict(
                num_modules=4,
                num_branches=3,
                block='BASIC',
                num_blocks=(4, 4, 4),
                num_channels=(48, 96, 192)),
            stage4=dict(
                num_modules=3,
                num_branches=4,
                block='BASIC',
                num_blocks=(4, 4, 4, 4),
                num_channels=(48, 96, 192, 384))),
    ),
    keypoint_head=dict(in_channels=48,
                       num_deconv_filters=[48]),    
)

model = dict(bu_model=bu_model,
             bu_ckpt='models/higherhrnet_w_root_w48_coco_640x640.pth',
             kp_embed_net=dict(in_c=48), 
             person_embed_net=dict(in_c=48)
             )

data_root = 'data/coco'
data = dict(
    samples_per_gpu=24,
    workers_per_gpu=4,
    train=dict(
        data_cfg=data_cfg,
        type='BottomUpCocoDatasetWithCentersAndBoxes',
        ann_file=f'{data_root}/annotations/person_keypoints_train2017.json',
        img_prefix=f'{data_root}/train2017/'),
    
    val=dict(
        data_cfg=data_cfg,
        type='BottomUpCocoDatasetWithCenters',
        ann_file=f'{data_root}/annotations/person_keypoints_val2017.json',
        img_prefix=f'{data_root}/val2017/'),

    test=dict(
        data_cfg=data_cfg,
        type='BottomUpCocoDatasetWithCenters',
        ann_file=f'{data_root}/annotations/person_keypoints_val2017.json',
        img_prefix=f'{data_root}/val2017/')
)
