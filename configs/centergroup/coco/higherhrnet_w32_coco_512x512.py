_base_ = '../base.py'
    
model = dict(bu_ckpt='models/higherhrnet_w_root_w32_coco_512x512.pth')


data_root = 'data/coco'
data = dict(
    samples_per_gpu=42,
    workers_per_gpu=4,
    train=dict(
        #type='BottomUpCocoDatasetWithCenters',
        type='BottomUpCocoDatasetWithCentersAndBoxes',
        ann_file=f'{data_root}/annotations/person_keypoints_train2017.json',
        img_prefix=f'{data_root}/train2017/'),
    
    val=dict(
        type='BottomUpCocoDatasetWithCenters',
        ann_file=f'{data_root}/annotations/person_keypoints_val2017.json',
        img_prefix=f'{data_root}/val2017/'),

    test=dict(
        type='BottomUpCocoDatasetWithCenters',
        ann_file=f'{data_root}/annotations/person_keypoints_val2017.json',
        img_prefix=f'{data_root}/val2017/')
)
