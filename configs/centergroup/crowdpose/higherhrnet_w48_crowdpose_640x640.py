_base_ = '../base.py'

NUM_JOINTS = 14
WITH_CENTER= 1

# Accounts HRNet w48 backbone
bu_model = dict(
    pretrained=None,
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
                       loss_keypoint=dict(num_joints=NUM_JOINTS+WITH_CENTER),
                       num_joints=NUM_JOINTS + WITH_CENTER,
                       num_deconv_filters=[48]),    
)

# Changes due to having a different number of joints, and using a larger backbone
model = dict(group_module=dict (head_cfg = dict(num_joints=NUM_JOINTS),
                        kp_encoder_cfg= dict(num_joints=NUM_JOINTS + WITH_CENTER)),
             bu_ckpt='models/higherhrnet_w_root_w48_crowdpose_640x640.pth',                               
             kp_embed_net=dict(in_c=48), 
             person_embed_net=dict(in_c=48),
             bu_model=bu_model
)
channel_cfg = dict(
    num_output_channels=NUM_JOINTS + WITH_CENTER,
    dataset_joints=NUM_JOINTS + WITH_CENTER,
    dataset_channel=[
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
    ],
    inference_channel=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14])


data_cfg = dict(
    num_joints=NUM_JOINTS+WITH_CENTER,
    flip_index=[1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 12, 13],
    dataset_channel=channel_cfg['dataset_channel'],
    inference_channel=channel_cfg['inference_channel'],
    image_size=640,
    base_size=320,
    base_sigma=2,
    heatmap_size=[160, 320, 640],

    )

data_root = 'data/crowdpose'
data = dict(
    samples_per_gpu=24,
    workers_per_gpu=8,
    train=dict(
        type='BottomUpCrowdPoseDatasetWithCentersAndBoxes',
        data_cfg = data_cfg,
        ann_file=f'{data_root}/annotations/mmpose_crowdpose_trainval.json',
        img_prefix=f'{data_root}/images/'),

    val=dict(
        type='BottomUpCrowdPoseDatasetWithCenters',
        data_cfg = data_cfg,
        ann_file=f'{data_root}/annotations/mmpose_crowdpose_test.json',
        img_prefix=f'{data_root}/images/'),
    test=dict(
        type='BottomUpCrowdPoseDatasetWithCenters',
        data_cfg = data_cfg,
        ann_file=f'{data_root}/annotations/mmpose_crowdpose_test.json',
        img_prefix=f'{data_root}/images/')
)
