import torch
import numpy as np
from torch import nn
from torchvision.ops import sigmoid_focal_loss
from torch.nn import functional as F

def _get_per_person_tensors(num_persons, vis_mask, det_ix, gt_ix, batch, res_ix, keys):
    per_person_targets = {}
    for key in keys:
        if key in batch:
            target = batch[key]
            if isinstance(target, list):
                target = target[res_ix]
            per_person_target = torch.zeros([num_persons]+[ix for i, ix in enumerate(target.shape) if i >1], 
                                            device=target.device,
                                            dtype = target.dtype)

            per_person_target[det_ix] = target[vis_mask][gt_ix]
            per_person_targets[key] = per_person_target
    
    return per_person_targets

import numpy as np

def oks_norm_dist_mat(dist_mat, boxes,  sigma = 0.15):
    """Normalizes values in distance matrix from pixels to object keypoint
    similarity scores (OKS) as used for COCO evaluation.
    Distances are normalized by the size of the ground truth box, and transformed
    to the (0, 1) interval.
    See: https://github.com/cocodataset/cocoapi/blob/8c9bcc3cf640524c4c20a9c40e89cb6a2f2fa0e9/PythonAPI/pycocotools/cocoeval.py#L192
    """
    num_det, num_gt = dist_mat.shape
    assert boxes.shape[0] == num_gt and boxes.shape[1]==2 and boxes.shape[2] ==3

    var = (sigma * 2)**2
    var = torch.as_tensor(var, device = dist_mat.device)

    hwidths = (boxes[..., 1, :2] - boxes[..., 0, :2])
    areas =  hwidths[..., 0] * hwidths[..., 1]
    tmparea = areas* 0.53
    norm_factor = 1 / var/(tmparea+np.spacing(1)) / 2
    norm_dist_mat = torch.einsum("dg,g->dg", dist_mat**2, norm_factor)
    oks = torch.exp(-norm_dist_mat)

    return oks

@torch.no_grad()
def generate_person_targets(dt_centers, 
                            person_batch_ix, batch,
                            person_mask,
                            assign_method = 'greedy',
                            max_match_dist=0.8,
                            joint_pred_res=-1):
    """Match detected centers to the ground truth ones to determine the joint coordinates and
    additional targets of each detected center """

    vis_mask = batch['obj_vis'] & (batch['has_kp_anns'] == 1)

    b_size=vis_mask.size(0)
    batch_ix = torch.arange(b_size, device=vis_mask.device).view(b_size, 1)
    batch_ix = batch_ix.expand(-1, vis_mask.size(1))

    assert len(batch['joints_']) ==3, "We assume we have access to full resolution heatmaps"
    gt_poses = batch['joints_'][-1][vis_mask]#[:, -1:]
    gt_batch = batch_ix[vis_mask]
    
    # Compute distances between all ground truth and detected centers 
    gt_centers = gt_poses[:, -1, :2]
    dist_mat = torch.cdist(dt_centers.type(gt_centers.dtype), gt_centers)
    dist_mat= oks_norm_dist_mat(dist_mat, boxes=batch['boxes'][-1][vis_mask])

    cost_matrix = 1- dist_mat    

    # Make persons in different batch elements unmatchable and with score too low
    diff_batch = person_batch_ix.reshape(-1, 1) != gt_batch.view(1, -1)
    cost_matrix[diff_batch] = np.nan
    cost_matrix[cost_matrix > max_match_dist] = np.nan

    # Obtain a matching
    if assign_method == 'greedy':
        det_ix, gt_ix = [], []
        cost_matrix[torch.isnan(cost_matrix)] = np.inf
        while not (cost_matrix == np.inf).all():
            min_idx = torch.argmin(cost_matrix)
            det_ix_ = min_idx //  cost_matrix.shape[1]
            gt_ix_ = min_idx %  cost_matrix.shape[1]

            assert cost_matrix[det_ix_, gt_ix_] == cost_matrix.min()

            cost_matrix[det_ix_] = np.inf
            cost_matrix[:, gt_ix_] = np.inf

            det_ix.append(det_ix_.item())
            gt_ix.append(gt_ix_.item())
        
        det_ix = np.array(det_ix)
        gt_ix = np.array(gt_ix)
    else:
        raise RuntimeError

    # Given this matching, assign GT keypoints/boxes to each detected center
    num_persons = cost_matrix.shape[0]
    per_person_batch = _get_per_person_tensors(num_persons=num_persons,
                                                vis_mask = vis_mask,
                                                det_ix=det_ix,
                                                gt_ix=gt_ix,
                                                res_ix=joint_pred_res,
                                                batch=batch,
                                                keys=['boxes', 'joints_'])


    # Generate additional targets
    is_tp = torch.zeros(num_persons, device = dist_mat.device)
    is_tp[det_ix] = 1

    no_anns = torch.ones(num_persons, device = dist_mat.device)
    is_in_ignore_region = person_mask == 1
    no_anns[det_ix] = batch['has_kp_anns'][vis_mask][gt_ix].type(is_tp.dtype)
    ignore = torch.minimum(no_anns, is_in_ignore_region)

    per_person_batch['is_tp'] = is_tp
    per_person_batch['ignore'] = ignore

    per_person_batch['vis_target'] = per_person_batch['joints_'][..., -1] > 0
    per_person_batch['loc_target'] = per_person_batch['joints_'][..., :2]

    return per_person_batch

def compute_oks(loc_pred, loc_target, viz_mask, boxes, upsample_factor=1):
    """Compute OKS scores between pairs of predicted and ground truth keypoint coordinates
    See: https://github.com/cocodataset/cocoapi/blob/8c9bcc3cf640524c4c20a9c40e89cb6a2f2fa0e9/PythonAPI/pycocotools/cocoeval.py#L192
    """
    if loc_pred.shape[-2] == 17: # COCO
        sigmas = torch.as_tensor([
                .26, .25, .25, .35, .35, .79, .79, .72, .72, .62, .62, 1.07, 1.07,
                .87, .87, .89, .89
            ]) / 10.0

    else: # CrowdPose
        sigmas = torch.as_tensor([
            .79, .79, .72, .72, .62, .62, 1.07, 1.07, .87, .87, .89, .89, .79,
            .79
        ]) / 10.0

    vars = (sigmas * 2)**2
    vars = vars.cuda()

    x_dist = loc_target[..., 0] - loc_pred[..., 0]
    y_dist = loc_target[..., 1] - loc_pred[..., 1]
    dist = x_dist**2 + y_dist**2
    
    hwidths = upsample_factor *(boxes[..., 1, :2] - boxes[..., 0, :2])
    areas =  hwidths[..., 0] * hwidths[..., 1]
    tmparea = areas.unsqueeze(-1).unsqueeze(0) * 0.53
    e = dist / vars/(tmparea+np.spacing(1)) / 2
    oks = torch.exp(-e)

    return oks * viz_mask

def compute_group_loss(preds, vis_target, loc_target, boxes, person_tp_target, ignore_person, loss_cfg):
    losses = dict()
    vis_pred = preds['viz_pred']
    assert vis_pred.size(-1) == 1
    vis_pred = vis_pred[..., 0]
    vis_target = vis_target.unsqueeze(0).expand(vis_pred.size(0), -1, -1).type(vis_pred.dtype)
    vis_target = vis_target[..., :-1]
    assert vis_target.shape == vis_pred.shape

    person_tp_pred = preds['person_tp_pred']
    person_tp_target = person_tp_target.unsqueeze(0).expand(person_tp_pred.size(0), -1)
    
    assert person_tp_pred.size(-1) == 1
    
    person_tp_loss = sigmoid_focal_loss(person_tp_pred[..., -1], person_tp_target, gamma = 2, alpha=-1) 
    #person_tp_loss = FocalLoss(logits=True)(person_tp_pred[..., -1], person_tp_target) 

    losses['person_loss'] = person_tp_loss[:, ignore_person ==1].mean()
    assert not torch.isnan(person_tp_loss).any() and not torch.isnan(losses['person_loss'])

    # Only compute keypoint based losses over persons that are TP and have annotation keypoints
    joints_loss_mask = torch.minimum(person_tp_target[0], ignore_person) == 1
    vis_loss = sigmoid_focal_loss(vis_pred, vis_target, gamma = 2, alpha=-1) 
    #vis_loss = FocalLoss(logits=True)(vis_pred, vis_target) 

    viz_mask = (vis_target >0)[:, joints_loss_mask]
    loc_pred = preds['loc_pred'][:, joints_loss_mask]
    loc_target = loc_target.unsqueeze(0).expand(loc_pred.size(0), -1, -1, -1).type(loc_pred.dtype)
    loc_target = loc_target[:, joints_loss_mask, :-1] 
    #loc_loss = F.l1_loss(loc_pred[viz_mask], loc_target[viz_mask])
    loc_loss = F.l1_loss(loc_pred, loc_target, reduce=False)

    losses['loc_loss']= loc_loss[viz_mask].mean()
    vis_loss_pre_mean = vis_loss[:, joints_loss_mask]
    if loss_cfg['mask_vis_loss']: # Mask visibility prediction if locataion prediction is not good
        oks_vals = compute_oks(loc_pred=loc_pred, loc_target=loc_target, viz_mask=viz_mask, 
                                boxes=boxes[joints_loss_mask])

        mask_vis_ = torch.ones_like(vis_loss_pre_mean)
        mask_vis_[viz_mask] = (oks_vals[viz_mask]>=0.5).type(mask_vis_.dtype)
        losses['vis_loss'] = (vis_loss_pre_mean*mask_vis_).sum()/mask_vis_.sum()

    else:
        losses['vis_loss'] = vis_loss_pre_mean.mean()

    losses['overall'] = loss_cfg['loc_weight'] * losses['loc_loss']
    losses['overall'] += loss_cfg['vis_weight'] * losses['vis_loss']
    losses['overall'] += loss_cfg['person_weight'] * losses['person_loss']

    return losses