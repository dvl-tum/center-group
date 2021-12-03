import torch
import torch
import torch.nn.functional as F
from mmpose.core.evaluation import get_multi_stage_outputs
import numpy as np

def nms(heatmaps, nms_kernel, nms_padding):    
    nms_pool = torch.nn.MaxPool2d(nms_kernel, 1, nms_padding)
    maxm = nms_pool(heatmaps)
    maxm = torch.eq(maxm, heatmaps).float()
    heatmaps = heatmaps * maxm
    return heatmaps

def get_kps(heatmaps, mask=None, rm_kp_in_mask=True, return_dense=False, cfg=None):        
    assert isinstance(heatmaps, torch.Tensor)
    assert heatmaps.dim() == 4 # Batch size, C, H, W
    device = heatmaps.device

    if mask is not None and rm_kp_in_mask:
        assert isinstance(mask, torch.Tensor)
        assert mask.dim() == 3 # Batch size, H, W
        assert mask.size(0) == heatmaps.size(0) and mask.size(1) == heatmaps.size(2) and mask.size(2) == heatmaps.size(3), f"Mask size: {mask.shape}. Heatmaps {heatmaps.shape}"
        heatmaps = heatmaps * mask.unsqueeze(1)

    # Select top k locations per KP type
    heatmaps = nms(heatmaps, nms_kernel=cfg['nms_kernel'], nms_padding=cfg['nms_padding'])
    N, K, H, W = heatmaps.size()
    heatmaps = heatmaps.view(N, K, -1)
    val_k, ind = heatmaps.topk(cfg['top_k_per_type'], dim=2)

    # Get their spatial coordinates
    x = ind % W
    y = ind // W
            
    # Add Batch index as an additional coordinate            
    b_size=ind.size(0)
    batch_ix = torch.arange(b_size, device=ind.device).view(b_size, 1, 1)
    batch_ix = batch_ix.expand(-1, ind.size(1), ind.size(2))

    ind_k = torch.stack((batch_ix, x, y), dim=3)
    pred_joint_type = torch.arange(K, device = device).view(1, K, 1).expand(N, -1, cfg['top_k_per_type'])

    # We may want to keep at least min_top_k_per_type kps. For that, we use the following tensor
    in_top_k = torch.zeros_like(val_k, dtype=torch.bool)
    in_top_k[..., :cfg['min_top_k_per_type']] = True
    keep = (val_k >=cfg['min_score']) | in_top_k

    return {'kp_score': val_k,
            'kp_pos': ind_k,
            'kp_ind': ind,
            'kp_type': pred_joint_type,
            'h': H,
            'w': W,
            'keep_mask': keep}

def get_kp_feats(features, kps):
    def extract_feat_at_inds(fmap, ind):
        N, C, H, W = fmap.size()
        _, K, _ = ind.size() # Number of Keypoints dimension
        fmap_ = fmap.view(N,1,C, -1).permute(0, 1, 3, 2) # flatten spatial dims and put channels last
        fmap_ = fmap_.expand(-1, K, -1, -1)
        ind_ = ind.unsqueeze(3).expand(-1, -1, -1, C)
        feats = torch.gather(fmap_, dim = 2, index=ind_ )
        return feats
    assert isinstance(features, torch.Tensor) and features.dim() ==4 
    assert (kps['h']/float(features.size(2))) == (kps['w']/float(features.size(3)))
    assert kps['kp_ind'].size(0) == features.size(0) # Same batch size

    # If needed, upsample feature maps to match the resolution at which keypoint coordinates  were extracted
    scale_factor = kps['h']/float(features.size(2))
    features= torch.nn.functional.interpolate(features, 
            scale_factor=scale_factor, mode="bilinear", align_corners=False, recompute_scale_factor=False)
    assert features.size(2) ==kps['h'] and features.size(3) ==kps['w']
    assert (kps['kp_ind'] < kps['w']*kps['h']).all()

    # Select feature vectors at KP locations from feature maps
    kp_feats = extract_feat_at_inds(features, kps['kp_ind'])

    return torch.flatten(kp_feats, 1, 2) # Flatten topK and KP  type dims


def kp_locs_from_heatmaps(bu_pred, bu_pred_flip, mask, test_cfg, node_cfg, size_projected, num_joints, flip_index, heatmaps=None):
    if not isinstance(bu_pred[0], list):
        if bu_pred_flip:
            assert not isinstance(bu_pred_flip[0], list)
        bu_pred  = [bu_pred]
        bu_pred_flip  = [bu_pred_flip]

    heatmaps_combined= None
    for bu_pred_, bu_pred_flip_ in zip(bu_pred, bu_pred_flip):
        _, heatmaps, _ = get_multi_stage_outputs(bu_pred_, outputs_flip=bu_pred_flip_,
                                                    num_joints=num_joints,
                                                    with_heatmaps=test_cfg['with_heatmaps'],
                                                    with_ae=test_cfg['with_ae'],
                                                    tag_per_joint=False,  # Only relevant if doing flipping
                                                    flip_index = flip_index,
                                                    project2image=test_cfg['project2image'],
                                                    size_projected=size_projected)
        if bu_pred_flip_ is not None:
            assert len(heatmaps) == 2
            heatmaps = (heatmaps[0] + heatmaps[1]) / 2.0 

        else:    
            heatmaps= heatmaps[0]

        if heatmaps_combined is None:
            heatmaps_combined = heatmaps
        else:
            heatmaps_combined += heatmaps

    heatmaps = heatmaps_combined / len(bu_pred)

    # Parse Heatmap to get Top scoring keypoint locations
    kps = get_kps(heatmaps, 
                  return_dense=True, 
                  mask=mask,
                  rm_kp_in_mask=True,
                  cfg = node_cfg)

    kps['heatmap'] = heatmaps
    return kps, heatmaps

def extract_feats_at_loc(fmaps, locs, h, w, upsample=False):
    if isinstance(fmaps, (list, tuple)):
        assert upsample
        avg_fmaps = []
        for fmap in fmaps:
            avg_fmaps.append(extract_feats_at_loc(fmap, locs, h, w, upsample=upsample))

        return 0.2* avg_fmaps[0] + 0.6* avg_fmaps[1] + 0.2* avg_fmaps[2] # Simple averaging gives ~0.5 AP worse

    else:
        assert locs.dim() == 4 and fmaps.dim()== 4
        assert locs.size(-1) == 3
        scale_factor = h/float(fmaps.size(2))
        assert scale_factor == w/float(fmaps.size(3))

        if upsample:
            # Upsample feature maps to match the resolution at which Keypoint coordinates  were extracted
            fmaps_= torch.nn.functional.interpolate(fmaps, 
                    scale_factor=scale_factor, mode="bilinear", align_corners=False, recompute_scale_factor=False)

            assert fmaps_.size(2) ==h and fmaps_.size(3) ==w, f"Fmaps shape: {fmaps_.shape}. H W: {(h, w)}"
            locs_ = locs    
        
        else:
            raise NotImplementedError

    return fmaps_[locs_[..., 0], :, locs_[..., 2], locs_[..., 1]]

def gather_all_kp_feats(embed_fmaps, mask, pos_fmaps, locs, h, w, upsample):
    if not isinstance(embed_fmaps, list):
        assert embed_fmaps.size(0) == mask.size(0) and  mask.size(0) == pos_fmaps.size(0) # Same batch size

    else:
        assert embed_fmaps[0].size(0) == mask.size(0) and  mask.size(0) == pos_fmaps.size(0) # Same batch size

    mask_ = mask.clone()
    if mask_.dim() == 3: #no Channels
        mask_ = mask_.unsqueeze(1)
    
    assert mask_.size(-1) == w and mask_.size(-2)==h, "Mask should be provided at full resolution"
        
    # Get embedding features, position embeddings and masked regions for every keypoint
    embed= extract_feats_at_loc(fmaps=embed_fmaps, locs=locs, h=h, w=w, upsample=upsample)
    mask=extract_feats_at_loc(fmaps=mask_, locs=locs, h=h, w=w, upsample=upsample)
    pos=extract_feats_at_loc(fmaps=pos_fmaps, locs=locs, h=h, w=w, upsample=upsample)

    return embed, mask, pos


def get_kp_graph(bu_pred, bu_pred_flip, kp_embed_fmaps, pos_embed_fmaps, batch, upsample, test_cfg, parsing_cfg, num_joints, flip_index, train=True, heatmaps=None,
                mask_crowd_kps=False):
    
    _, _, im_h, im_w = batch['img'].shape
    kps, heatmaps_ = kp_locs_from_heatmaps(bu_pred = bu_pred,  
                              num_joints=num_joints,
                              flip_index=flip_index,
                              bu_pred_flip =bu_pred_flip,
                              mask = batch['pad_mask' if (not train or not mask_crowd_kps) else 'masks'][-1], 
                              test_cfg = test_cfg,
                              size_projected = (im_w, im_h),
                              node_cfg=parsing_cfg,
                              heatmaps=heatmaps)
        
    kps['kp_embed'], kps['kp_mask'], kps['kp_pos_embed']= gather_all_kp_feats(embed_fmaps= kp_embed_fmaps, 
                                                    mask=batch['pad_mask' if not train else 'masks'][-1] , 
                                                    pos_fmaps=pos_embed_fmaps, 
                                                    locs=kps['kp_pos'], 
                                                    h=kps['h'],
                                                    w=kps['w'],
                                                    upsample=upsample)

    kps['coords'] = kps['kp_pos'][..., 1:]
    kps['batch'] = kps['kp_pos'][..., :1]
    kps['score'] = kps['kp_score']
    kps['heatmaps'] = heatmaps_

    return kps
    
def get_person_graph(bu_pred, bu_pred_flip, batch, person_embed_fmaps, pos_embed_fmaps, parsing_cfg, test_cfg,upsample=True,train=True,
                     heatmaps=None, mask_crowd_kps=False):
    _, _, im_h, im_w = batch['img'].shape
    person_kps, heatmaps_= kp_locs_from_heatmaps(bu_pred = bu_pred,  
                                                bu_pred_flip =bu_pred_flip,
                                                num_joints = 1,
                                                flip_index =[0],
                                                mask = batch['pad_mask' if (not train or not mask_crowd_kps) else 'masks'][-1] , 
                                                test_cfg = test_cfg,
                                                size_projected = (im_w, im_h),
                                                node_cfg=parsing_cfg,
                                                heatmaps=heatmaps)
    
    person_kps['person_embed'], person_kps['person_mask'], person_kps['person_pos']= gather_all_kp_feats(embed_fmaps= person_embed_fmaps, 
                                                                mask=batch['pad_mask' if not train else 'masks'][-1], 
                                                                pos_fmaps=pos_embed_fmaps, 
                                                                locs=person_kps['kp_pos'], 
                                                                h=person_kps['h'],
                                                                w=person_kps['w'],
                                                                upsample=upsample)
    person_kps['score'] = person_kps['kp_score']
    person_kps['coords'] = person_kps['kp_pos'][..., 1:]
    person_kps['batch'] = person_kps['kp_pos'][..., :1]
    person_kps['heatmaps'] = heatmaps_

    return person_kps