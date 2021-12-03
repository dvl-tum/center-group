# CenterGroup Checkpoints
The following checkpoints correspond to our final models:

| Train Data | Method| Detector|Input size | Val/Test AP | Config | Checkpoint |
|----------|----------|------------|------------|-------|-------|--------|
| COCO Train 2017 | CenterGroup| HigherHRNet-w32  | 512 | 69.0  | [cfg](../configs/centergroup/coco/higherhrnet_w32_coco_512x512.py) | [ckpt](https://vision.in.tum.de/webshare/u/brasoand/centergroup/models/centergroup/centergroup_higherhrnet_w32_coco_512x512.pth) |
| COCO Train 2017 | CenterGroup| HigherHRNet-w48  | 640 | 71.0  | [cfg](../configs/centergroup/coco/higherhrnet_w48_coco_640x640.py) | [ckpt](https://vision.in.tum.de/webshare/u/brasoand/centergroup/models/centergroup/centergroup_higherhrnet_w48_coco_640x640.pth) |
| CrowdPose train+val | CenterGroup| HigherHRNet-w48  | 640 | 67.6  | [cfg](../configs/centergroup2/crowdpose/higherhrnet_w48_crowdpose_640x640.py) | [ckpt](https://vision.in.tum.de/webshare/u/brasoand/centergroup/models/centergroup/centergroup_higherhrnet_w48_crowdpose_640x640.pth) |

Note that we report single-scale AP on the respective val/test dataset (val for COCO, test for CrowdPose).

# HigherHRNet with Centers
As explained in the paper, before training CenterGroup, we first pretrain our Keypoint Detector, HigherHRNet, by adding an additional keypoint prediction corresponding to person centers. When training CenterGroup, these checkpoints are needed to initialize our detector.

| Train Data | Method| Detector|Input size | Test AP | Config | Checkpoint |
|----------|----------|------------|------------|-------|-------|--------|
| COCO Train 2017 | CenterGroup| HigherHRNet-w32  | 512 | 67.2  | [cfg](../configs/higherhrnet_w_root/higherhrnet_w_root_w32_coco_512x512.py) | [ckpt](https://vision.in.tum.de/webshare/u/brasoand/centergroup/models/higherhrnet_w_root/higherhrnet_w_root_w32_coco_512x512.pth) |
| COCO Train 2017 | CenterGroup| HigherHRNet-w48  | 640 | 69.7  | [cfg](../configs/higherhrnet_w_root/higherhrnet_w_root_w48_coco_640x640.py) | [ckpt](https://vision.in.tum.de/webshare/u/brasoand/centergroup/models/higherhrnet_w_root/higherhrnet_w_root_w48_coco_640x640.pth) |
| CrowdPose train+val | CenterGroup| HigherHRNet-w48  | 640 | ??  | [cfg](../configs/higherhrnet_w_root/higherhrnet_w_root_w48_crowdpose_640x640.py) | [ckpt](https://vision.in.tum.de/webshare/u/brasoand/centergroup/models/higherhrnet_w_root/higherhrnet_w_root_w48_crowdpose_640x640.pth) |

Note that we report single-scale AP on the respective val/test dataset (val for COCO, test for CrowdPose).

Our config files expect to find these checkpoints under `$CENTERGROUP_ROOT/models`. Please update the config files if you download them somewhere else.
