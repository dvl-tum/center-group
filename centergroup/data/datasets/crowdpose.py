import json_tricks as json
import numpy as np
from xtcocotools.coco import COCO
from xtcocotools.cocoeval import COCOeval

from mmpose.datasets.builder import DATASETS
from .coco import BottomUpCocoDatasetWithCenters, BottomUpCocoDataset


@DATASETS.register_module()
class BottomUpCrowdPoseDatasetWithCenters(BottomUpCocoDatasetWithCenters):
    """CrowdPose dataset for bottom-up pose estimation.

    The dataset loads raw features and apply specified transforms
    to return a dict containing the image tensors and other information.

    CrowdPose keypoint indexes::

        0: 'left_shoulder',
        1: 'right_shoulder',
        2: 'left_elbow',
        3: 'right_elbow',
        4: 'left_wrist',
        5: 'right_wrist',
        6: 'left_hip',
        7: 'right_hip',
        8: 'left_knee',
        9: 'right_knee',
        10: 'left_ankle',
        11: 'right_ankle',
        12: 'top_head',
        13: 'neck'

    Args:
        ann_file (str): Path to the annotation file.
        img_prefix (str): Path to a directory where images are held.
            Default: None.
        data_cfg (dict): config
        pipeline (list[dict | callable]): A sequence of data transforms.
        test_mode (bool): Store True when building test or
            validation dataset. Default: False.
    """

    def __init__(self,
                 ann_file,
                 img_prefix,
                 data_cfg,
                 pipeline,
                 test_mode=False):
        super(BottomUpCocoDataset, self).__init__(
            ann_file, img_prefix, data_cfg, pipeline, test_mode=test_mode)
        
        self.ann_info['with_center'] = data_cfg['with_center'] if 'with_center' in data_cfg else False

        self.ann_info['flip_index'] = [
            1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 12, 13
        ]
        
        self.ann_info['flip_index_with_c'] = [
            1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 12, 13, 14
        ]        
        

        self.ann_info['use_different_joint_weights'] = False
        self.ann_info['joint_weights'] = np.array(
            [
                0.2, 0.2, 0.2, 1.3, 1.5, 0.2, 1.3, 1.5, 0.2, 0.2, 0.5, 0.2,
                0.2, 0.5
            ],
            #dtype=np.float32).reshape((self.ann_info['num_joints'], 1))
            dtype=np.float32).reshape((-1, 1))

        # 'https://github.com/Jeff-sjtu/CrowdPose/blob/master/crowdpose-api/'
        # 'PythonAPI/crowdposetools/cocoeval.py#L224'
        self.sigmas = np.array([
            .79, .79, .72, .72, .62, .62, 1.07, 1.07, .87, .87, .89, .89, .79,
            .79
        ]) / 10.0

        self.coco = COCO(ann_file)

        cats = [
            cat['name'] for cat in self.coco.loadCats(self.coco.getCatIds())
        ]
        self.classes = ['__background__'] + cats
        self.num_classes = len(self.classes)
        self._class_to_ind = dict(zip(self.classes, range(self.num_classes)))
        self._class_to_coco_ind = dict(zip(cats, self.coco.getCatIds()))
        self._coco_ind_to_class_ind = dict(
            (self._class_to_coco_ind[cls], self._class_to_ind[cls])
            for cls in self.classes[1:])
        self.img_ids = self.coco.getImgIds()
        if not test_mode:
            self.img_ids = [
                img_id for img_id in self.img_ids
                if len(self.coco.getAnnIds(imgIds=img_id, iscrowd=None)) > 0
            ]
        self.num_images = len(self.img_ids)
        self.id2name, self.name2id = self._get_mapping_id_name(self.coco.imgs)
        self.dataset_name = 'crowdpose'

        print(f'=> num_images: {self.num_images}')

    def _do_python_keypoint_eval(self, res_file):
        """Keypoint evaluation using COCOAPI."""

        stats_names = [
            'AP', 'AP .5', 'AP .75', 'AR', 'AR .5', 'AR .75', 'AP(E)', 'AP(M)',
            'AP(H)'
        ]

        with open(res_file, 'r') as file:
            res_json = json.load(file)
            if not res_json:
                info_str = list(zip(stats_names, [
                    0,
                ] * len(stats_names)))
                return info_str

        coco_det = self.coco.loadRes(res_file)
        coco_eval = COCOeval(
            self.coco,
            coco_det,
            'keypoints_crowd',
            self.sigmas,
            use_area=False)
        coco_eval.params.useSegm = None
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        info_str = list(zip(stats_names, coco_eval.stats))

        return info_str
    def evaluate(self, *args, **kwargs):
        assert self.ann_info['num_joints'] == 15
        self.ann_info['num_joints'] = 14
        out = super(BottomUpCocoDatasetWithCenters, self).evaluate(*args, **kwargs)
        self.ann_info['num_joints'] = 15
        return out

@DATASETS.register_module()
class BottomUpCrowdPoseDatasetWithCentersAndBoxes(BottomUpCrowdPoseDatasetWithCenters):
    """Copied from coco.py file"""
    def _get_clean_box(self, obj, height, width):
        x, y, w, h = obj['bbox']

        # Top left
        x1 = max(0, x)
        y1 = max(0, y)

        # Bottom right
        x2 = x1 + max(0, w)
        x2 = min(width - 1, x2)

        y2 = y1 + max(0, h)
        y2 = min(height - 1, y2)

        if ('area' not in obj or obj['area'] > 0) and x2 > x1 and y2 > y1:
            return np.array([[x1, y1],
                             [x2, y1], # Top-right
                             [x1, y2], # Bottom-Left
                             [x2, y2]
                             ])
        else:
            print(obj['bbox'])
            raise RuntimeError("Image does not have proper box")
    
    def _filter_anno(self, anno):
        ixs_to_keep = []
        for i, obj in enumerate(anno):
            try:
                if 'bbox' not in obj:
                    seg_mask =self.coco.annToMask(anno[i])
                    mask_y, mask_x = np.where(seg_mask)
                    x = min(mask_x)
                    y = min(mask_y)
                    w = max(mask_x) - x
                    h = max(mask_y) - y
                    #print("Old:", anno[i]['bbox'], "New", np.array([x, y, w, h]))
                    anno[i]['bbox'] = np.array([x, y, w, h])
                    anno[i]['area'] = w*h

                valid_bbox = anno[i]['bbox'][-1] > 0 and anno[i]['bbox'][-2]>0
                #valid_kps = (np.array(obj['keypoints']) != 0).any()
                if not valid_bbox:
                    raise RuntimeError
                ixs_to_keep.append(i)
            except:
                pass

        return [anno[i] for i in ixs_to_keep]

    def _get_joints(self, anno):
        """Modify get joints function to also include box corners, and discard objects without proper boxes"""
        anno = self._filter_anno(anno)
        joints = super()._get_joints(anno)
        

        #joints = self._get_joints(anno)
        
        # Add boxes corners as additional keypoints
        if anno:
            img_ann = self.coco.loadImgs(anno[0]['image_id'])[0]
            height, width = img_ann['height'], img_ann['width']
            
            assert len(anno) == joints.shape[0]
            boxes = np.zeros((joints.shape[0], 4, 3))
            boxes[..., 2] = 2
            for i, obj in enumerate(anno):
                boxes[i, ..., :2] = self._get_clean_box(obj, height, width)

            joints = np.concatenate((joints, boxes), axis = 1)
        
        return joints

