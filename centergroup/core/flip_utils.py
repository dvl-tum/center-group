import torch
import copy

def _split_bottom_up_outputs(bottom_up_outs, batch, batch_flip):
    """Retrieve flipped heatmaps and feature maps from Bottom-Up keypoint detector forward pass"""
    bottom_up_outs_flip = copy.deepcopy(bottom_up_outs)

    for key in ('kp_pred', 'p_pred'):
        bottom_up_outs_flip[key], bottom_up_outs_flip[key+'_flip'] = bottom_up_outs_flip[key+'_flip'], bottom_up_outs_flip[key]
        
    for key in ('kp_embed_fmaps', 'person_embed_fmaps'):
        bottom_up_outs_flip[key]=bottom_up_outs_flip.pop(key + '_flip')

    for key in ('kp_embed_fmaps', 'person_embed_fmaps'):
        bottom_up_outs.pop(key + '_flip')

    bottom_up_outs['batch'] = batch
    bottom_up_outs_flip['batch'] = batch_flip
    
    return bottom_up_outs, bottom_up_outs_flip


def _merge_bu_flip_preds(bottom_up_outs, bottom_up_outs_flip):
    for key in bottom_up_outs.keys():
        if isinstance(bottom_up_outs[key], (tuple, list)):
            for scale_idx, out in enumerate(bottom_up_outs[key]):
                if isinstance(out, (tuple, list)):
                    for i, out_ in enumerate(out):
                        assert out_.shape[0] == 1
                        bottom_up_outs[key][scale_idx][i] = torch.cat((out_, bottom_up_outs_flip[key][scale_idx][i]),dim=0)
        
                else:
                    assert out.shape[0] == 1
                    bottom_up_outs[key][scale_idx] = torch.cat((out, bottom_up_outs_flip[key][scale_idx]), dim=0)
        
        elif isinstance(bottom_up_outs[key], (torch.Tensor)):
            assert bottom_up_outs[key].shape[0] == 1
            bottom_up_outs[key] = torch.cat((bottom_up_outs[key], 
                                            bottom_up_outs_flip[key]), dim=0)

    for i, mask in enumerate(bottom_up_outs['batch']['pad_mask']):
        bottom_up_outs['batch']['pad_mask'][i] = torch.cat((mask, bottom_up_outs_flip['batch']['pad_mask'][i]), 
                                                            dim=0)

    return bottom_up_outs


def _split_flip_group_preds(preds, batch_idx):
    assert (torch.unique(batch_idx) == torch.as_tensor([0, 1], device=batch_idx.device)).all(), "Predictions contain more than 1 image!"
    preds_flip = {}
    keys_to_pop = []
    for key, pred in preds.items():
        if pred.shape[1] == batch_idx.nelement():
            preds_flip[key] = pred[:, batch_idx == 1]
            preds[key] = pred[:, batch_idx == 0]

        elif pred.shape[1] == 2:
            preds_flip[key] = pred[:, 1: ]
            preds[key] = pred[:, :1]

        else:
            keys_to_pop.append(key)

    for key in keys_to_pop:
        preds.pop(key)
    
    return preds, preds_flip


def _flip_back_preds(preds_, flip_index, w):
    num_joints = len(flip_index)

    for key in ['hard_loc_pred', 'loc_pred']:
        assert preds_[key].dim() == 4 and preds_[key].shape[-1] == 2 and preds_[key].shape[-2] == num_joints
        preds_[key]  = preds_[key][:,:, flip_index ]
        preds_[key][..., 0] = w - preds_[key][..., 0] - 1

    assert preds_['viz_pred'].dim() == 4 and preds_['viz_pred'].shape[-1] == 1 and preds_['viz_pred'].shape[-2] == num_joints
    preds_['viz_pred'] = preds_['viz_pred'][:, :, flip_index]

    for key in ('loc_attn_weights', 'loc_attn_weights_raw'):
        assert preds_[key].dim() == 5 and preds_[key].shape[-1] == num_joints * 30
        preds_[key].shape
        preds_[key] = preds_[key][:, :, :, flip_index]

    return preds_


def _merge_stacked_preds(stacked_preds, person_comb= 'avg', joint_comb='avg'):
    unique_pred = {'loc_attn_weights': stacked_preds['loc_attn_weights'][0]}
    
    if person_comb == 'avg':
        unique_pred['person_tp_pred'] = stacked_preds['person_tp_pred'].mean(0)

    elif person_comb == 'max':
        unique_pred['person_tp_pred'] = torch.maximum(*stacked_preds['person_tp_pred'])

    ##########
    _, indices =torch.max(stacked_preds['viz_pred'], dim = 0)

    h_ix, p_ix, j_ix, _  = torch.where(indices != 0)
    pred_ix = indices[h_ix, p_ix, j_ix, _]

    unique_pred['viz_pred'] = stacked_preds['viz_pred'][0]
    unique_pred['hard_loc_pred'] = stacked_preds['hard_loc_pred'][0]
    unique_pred['loc_pred'] = stacked_preds['loc_pred'][0]

    if joint_comb == 'avg':
        unique_pred['viz_pred'] = stacked_preds['viz_pred'].mean(0)

    elif joint_comb == 'max':
        unique_pred['viz_pred'][h_ix, p_ix, j_ix] = stacked_preds['viz_pred'][pred_ix, h_ix, p_ix, j_ix]
    unique_pred['hard_loc_pred'][h_ix, p_ix, j_ix] = stacked_preds['hard_loc_pred'][pred_ix, h_ix, p_ix, j_ix]
    unique_pred['loc_pred'][h_ix, p_ix, j_ix] = stacked_preds['loc_pred'][pred_ix, h_ix, p_ix, j_ix]

    return unique_pred