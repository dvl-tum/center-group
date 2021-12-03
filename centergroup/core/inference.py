import torch
import numpy as np
from mmpose.core.post_processing.post_transforms import get_affine_transform, affine_transform
from mmpose.core.post_processing.group import HeatmapParser

def transform_keypoints(kps, tform_mat):
    tform_kps = np.zeros_like(kps)
    for i, kp in enumerate(kps):
        tform_kps[i] = affine_transform(kp, tform_mat[:2])

    return tform_kps

def assign_pose_conf(vis_conf_, person_conf, kp_thresh, person_thresh):
    joint_vis = (vis_conf_ > kp_thresh)
    topk_score_per_pose, _ = torch.topk(vis_conf_, k=5, dim=1)
    joint_vis  = torch.maximum(joint_vis, vis_conf_ >= topk_score_per_pose[:, -1:])
    vis_conf = (vis_conf_ * joint_vis).sum(dim=1)  / (joint_vis.sum(dim=1) + 1e-9)
    pose_conf = person_conf.clone().view(-1)
    mask = (person_conf >person_thresh).view(-1)
    pose_conf[mask]=vis_conf[mask].view(-1).type(pose_conf.dtype)
    return pose_conf

def get_pose_output(model, preds, person_batch, bottom_up_out, test_cfgs, return_heatmaps=False, return_for_dflt=False):
    batch = bottom_up_out['batch']
    b_size, _, h, w = batch['img'].shape
    assert b_size == 1
    hmap_w, hmap_h = person_batch['kp_res']
    assert (h %hmap_h) == 0 and (h / hmap_h) == (w / hmap_w) 
    upsample_factor = (h / hmap_h)

    results = {}
    for cfg_name, test_cfg in test_cfgs.items():    
        conf_thresh=test_cfg['conf_thresh']
        tformer_layer = test_cfg['tformer_layer']

        loc_preds = upsample_factor * preds['hard_loc_pred'][tformer_layer]
        person_conf =torch.sigmoid(preds['person_tp_pred'][tformer_layer])[..., 0]
        vis_conf= torch.sigmoid(preds['viz_pred'][tformer_layer])

        pose_conf= assign_pose_conf(vis_conf_=vis_conf,
                                    person_conf=person_conf,
                                    kp_thresh=conf_thresh,
                                    person_thresh=conf_thresh)

        kp_preds = torch.cat((loc_preds, vis_conf), -1).detach().cpu().numpy()
        pose_scores = pose_conf.view(-1).detach().cpu().numpy()
        
        if test_cfg['adjust']:
            try:
                parser = HeatmapParser(model.bu_model.test_cfg)
                kp_preds = parser.adjust([kp_preds.copy()], person_batch['kp_heatmaps'])[0]
            except:
                #print("Adjust did not work!!!")
                pass

        # Project keypoint coords back to original image coordinates
        img_metas = batch['img_metas']
        tform_matrix = get_affine_transform(center=img_metas['center'], 
                                            scale=img_metas['scale'], 
                                            rot=0., 
                                            output_size=(w, h), 
                                            inv=True)
        tformed_kp_preds = np.zeros_like(kp_preds)
        tformed_kp_preds[..., 2:] = kp_preds[..., 2:]
        for i, kps in enumerate(kp_preds):
            tformed_kp_preds[i,..., :2] = transform_keypoints(kps[..., :2], tform_matrix)

        # Add one useless 4th column
        tformed_kp_preds = np.concatenate((tformed_kp_preds, tformed_kp_preds[..., -1:]), axis=-1)
        final_out= [[tformed_kp_preds, pose_scores, list(batch['img_metas']['image_file']), None]]

        if return_heatmaps: 
            return final_out, person_batch

        else:
            results[cfg_name] = final_out
            if return_for_dflt and cfg_name=='dflt':
                return final_out

    return results