
dist_params = dict(backend='nccl')

log_level = 'INFO'
load_from = None
resume_from = None
dist_params = dict(backend='nccl')
workflow = [('train', 1)]
checkpoint_config = dict(interval=2)
evaluation = dict(interval=1, metric='mAP', save_best='AP')

channel_cfg = dict(
    dataset_joints=18,
    dataset_channel=[[
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16
    ]],
    inference_channel=[
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16
    ],
    num_output_channels=18)

data_cfg = dict(
    image_size=512,
    base_size=256,
    base_sigma=2,
    heatmap_size=[128, 256, 512],
    num_joints=18,
    dataset_channel=[[
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16
    ]],
    inference_channel=[
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16
    ],
    num_scales=3,
    scale_aware_sigma=False,
    unbiased_encoding=False,
    max_num_people=30,
    with_center=True,
    flip_index=[0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15])


bu_model = dict(
    type='AssociativeEmbedding_',
    pretrained=
    'https://download.openmmlab.com/mmpose/pretrain_models/hrnet_w32-36af842e.pth',
    backbone=dict(
        type='HRNet',
        in_channels=3,
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
                num_channels=(32, 64)),
            stage3=dict(
                num_modules=4,
                num_branches=3,
                block='BASIC',
                num_blocks=(4, 4, 4),
                num_channels=(32, 64, 128)),
            stage4=dict(
                num_modules=3,
                num_branches=4,
                block='BASIC',
                num_blocks=(4, 4, 4, 4),
                num_channels=(32, 64, 128, 256))),
        norm_eval=True),
    keypoint_head=dict(
        type='AEHigherResolutionHeadWithRoot',
        in_channels=32,
        remove_center_test=True,
        num_joints=18,
        tag_per_joint=True,
        extra=dict(final_conv_kernel=1, ),
        num_deconv_layers=1,
        num_deconv_filters=[32],
        num_deconv_kernels=[4],
        num_basic_blocks=4,
        cat_output=[True],
        with_ae_loss=[True, False],
        loss_keypoint=dict(
            type='MultiLossFactory',
            num_joints=18,
            num_stages=2,
            ae_loss_type='exp',
            #with_ae_loss=[True, False],
            with_ae_loss=[False, False],
            push_loss_factor=[0.001, 0.001],
            pull_loss_factor=[0.001, 0.001],
            with_heatmaps_loss=[True, True],
            heatmaps_loss_factor=[1.0, 1.0])),
    train_cfg=dict(
        num_joints=18,
        img_size=512),
    test_cfg=dict(
        num_joints=17,
        max_num_people=30,
        scale_factor=[1],
        with_heatmaps=[True, True],
        with_ae=[True, False],
        project2image=True,
        nms_kernel=5,
        nms_padding=2,
        tag_per_joint=True,
        detection_threshold=0.1,
        tag_threshold=1,
        use_detection_val=True,
        ignore_too_much=False,
        adjust=True,
        refine=True,
        flip_test=True))

test_cfg = dict(flip_test=True,
                multiscale=False, # If it's set to true, then test_scale_factor in BottomUpGetImgSize needs to be changed accordingly
                conf_thresh=0.5,
                tformer_layer=-1,
                adjust=True)

heatmap_cfg = dict(kps=dict(parsing_cfg=dict(
                            nms_kernel=5,
                            nms_padding=2,
                            top_k_per_type=30,
                            min_score=0.005,
                            min_top_k_per_type=3),
                            upsample=True,
                            res_ix=0),
                   persons=dict(parsing_cfg=dict(
                                nms_kernel=17,
                                nms_padding=8,
                                top_k_per_type=30,
                                min_score=0.005,
                                min_top_k_per_type=3),
                                upsample=True,
                                res_ix=0))

kp_embed_net = dict(in_c=32, hidden_c=128, out_c=128, num_blocks=2)
person_embed_net = dict(in_c=32, hidden_c=128, out_c=128, num_blocks=2)

group_module = dict(
    kp_encoder_cfg=dict(
        mlp_cfg=dict(
            input_dim=128, hidden_dim=128, output_dim=128, num_layers=1),
        w_mlp=True,
        w_type=True,
        num_joints=18,
        dim_size=128),
    transformer_cfg=dict(
        layer_cfg=dict(
            d_model=128,
            nhead=4,
            dim_feedforward=512,
            dropout=0.1,
            activation='relu',
            normalize_before=False),
        num_layers=3,
        skip_conns=True),
    head_cfg=dict(
        dim_size=128,
        hidden_dim=64,
        num_joints=17,
        mlp_viz_cfg=dict(num_layers=3),
        mlp_person_tp_cfg=dict(num_layers=3),
        share_weights=False,
        w_pos=True,
        vis_from_attn=False,
        use_type_mask=False,
        vis_attn=True,
        share_loc_vis_attn=True,
        predict_first=True),
    initial_mlp_cfg=dict(
        input_dim=128,
        hidden_dim=128,
        output_dim=128,
        num_layers=2,
        relu_last=True))

model = dict(type='CenterGroup',
             bu_model=bu_model,
             group_module=group_module,
             #bu_ckpt='/storage/user/brasoand/pose/mmpose/work_dirs/higher_hrnet32_coco_512x512_w_root/latest.pth',
             kp_embed_net=kp_embed_net,
             person_embed_net=person_embed_net,
             train_cfg=dict(freeze_bn=True,
                            optimizer_ = {'type': 'Adam', 'bu_lr': 1e-05, 'lr': 0.0001},
                            lr_scheduler_ = {'milestones': [20, 40]},
                            train_cnn=True,
                            lr_warmup =  {'do_warmup': True,
                                            'warmup_iters': 1000,
                                            'warmup_ratio': 0.001},
                            bu_loss_factor = 10,                                                                        
                            group_loss = {'loc_weight': 0.02,
                                            'vis_weight': 1.0,
                                            'person_weight': 1.0,
                                            'with_sigma': True,
                                            'mask_vis_loss': True
                                            },
                            gt_assign = {'assign_method': 'greedy',
                                         'max_match_dist': 0.8}),
             test_cfg=test_cfg,
             heatmap_cfg=heatmap_cfg)
             
train_pipeline = [
    dict(type='AddHasKPAnns'), # Needs to be called BEFORE BotttomUpGenerateTarget
    dict(type='LoadImageFromFile'),
    dict(
        type='BottomUpRandomAffineWPadMask',
        #rot_factor=30,
        rot_factor=0,
        scale_factor=[0.75, 1.5],
        scale_type='short',
        trans_factor=40),
    dict(type='BottomUpRandomFlipWPadMaskAndBoxes', flip_prob=0.5),
    dict(type='ToTensor'),
    dict(
        type='NormalizeTensor',
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]),
    dict(type='SeparateJointsAndBoxes'), # Needs to be called before GenerateTarget  and GenerateRootNode
    dict(type='AddAndUpdateVisibility'), # Needs to be called AFTER RandomAffine
    dict(type='GenerateRootNode'), # Needs to be called before GenerateTarget
    dict(type='CloneJoints'), # Needs to be called before GenerateTarget
    dict(type='BottomUpGenerateTarget', sigma=2, max_num_people=30),
    dict(
        type='PadArrays',
        max_num_people=30,
        keys_to_update=['joints_', 'obj_vis', 'has_kp_anns', 'boxes']),
    dict(
        type='Collect',
        keys=[
            'img', 'boxes', 'masks',  'pad_mask', 'joints', 'joints_', 'targets',
            'obj_vis', 'has_kp_anns'
        ],
        meta_keys=['flip_index'])
]

val_pipeline = [dict(type= 'LoadImageFromFile'),
                  #dict(type='BottomUpGetImgSize', test_scale_factor=[0.5, 1, 2]), # For Multi-scale test
                dict(type='BottomUpGetImgSize', test_scale_factor=[1, 1, 1]),
                dict(
                     type='BottomUpResizeAlignWPadMask',
                     transforms=[
                         dict(type='ToTensor'),
                         dict(
                              type='NormalizeTensor',
                              mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225]),
                      ]),
                 #dict(type='RenameStuffMultiScale'),
                 dict(type='Collect',
                      keys=['img', 'pad_mask'],
                      meta_keys=['image_file', 'center', 'scale', 'aug_data', 'test_scale_factor', 'base_size',
                                 'center', 'scale', 'flip_index'])]

data = dict(
    samples_per_gpu=10,
    workers_per_gpu=4,
    train=dict(
        data_cfg=data_cfg,
        pipeline=train_pipeline),
    
    val=dict(
        data_cfg=data_cfg,
        pipeline=val_pipeline),

    test=dict(
        data_cfg=data_cfg,
        pipeline=val_pipeline)
)