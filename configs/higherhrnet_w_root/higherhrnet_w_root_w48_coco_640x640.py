_base_ = '../higherhrnet/higherhrnet_w48_coco_640x640.py'

checkpoint_config = dict(interval=2)
evaluation = dict(interval=15, metric='mAP', key_indicator='AP')
log_config = dict(interval=50)



WITH_CENTER = 1
NUM_JOINTS = 17
channel_cfg = dict(
    num_output_channels=NUM_JOINTS + WITH_CENTER,
    dataset_joints=NUM_JOINTS + WITH_CENTER)

data_cfg = dict(
    image_size=640,
    num_joints=NUM_JOINTS+WITH_CENTER,
    with_center = bool(WITH_CENTER))

#optimizer = dict(lr=0.0015*(20/24))
optimizer = dict(lr=0.0015)

model=dict(
    keypoint_head=dict(
        #type='BottomUpHigherResolutionHead',
        type='AEHigherResolutionHeadWithRoot',
        remove_center_test=True,
        num_joints=NUM_JOINTS + WITH_CENTER,
        loss_keypoint=dict( num_joints=NUM_JOINTS + WITH_CENTER)
),

    train_cfg=dict(
        num_joints=channel_cfg['dataset_joints'],
        img_size=data_cfg['image_size']),
    test_cfg=dict(
        num_joints=channel_cfg['dataset_joints'] - 1) # Very important
)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='BottomUpRandomAffine',
        rot_factor=30,
        scale_factor=[0.75, 1.5],
        scale_type='short',
        trans_factor=40),
    dict(type='BottomUpRandomFlip', flip_prob=0.5),
    dict(type='ToTensor'),
    dict(
        type='NormalizeTensor',
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]),
    dict(
        type='GenerateRootNode',
    ),
    dict(
        type='BottomUpGenerateTarget',
        sigma=2,
        max_num_people=30,
    ),
    dict(
        type='Collect',
        keys=['img', 'joints', 'targets', 'masks'],
        meta_keys=[]),
]


#data_root = '/storage/user/brasoand/coco'
data_root = '/storage/slurm/brasoand/coco'

data = dict(
    samples_per_gpu=24,
    #samples_per_gpu=20,
    workers_per_gpu=4,
    train=dict(
        type='BottomUpCocoDatasetWithCenters',
        data_cfg = data_cfg,
        ann_file=f'{data_root}/annotations/person_keypoints_train2017.json',
        img_prefix=f'{data_root}/train2017/',
        pipeline=train_pipeline),
    val=dict(
        type='BottomUpCocoDatasetWithCenters',
        data_cfg=data_cfg,
        ann_file=f'{data_root}/annotations/person_keypoints_val2017.json',
        img_prefix=f'{data_root}/val2017/'),
        
    test=dict(
        type='BottomUpCocoDatasetWithCenters',
        data_cfg=data_cfg,
        ann_file=f'{data_root}/annotations/person_keypoints_val2017.json',
        #ann_file=f'{data_root}/annotations/person_keypoints_val2017_tiny.json',

        img_prefix=f'{data_root}/val2017/'),
)