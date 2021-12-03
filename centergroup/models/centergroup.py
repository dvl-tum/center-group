import os.path as osp
import copy
from copy import deepcopy     


from time import time
from torch.nn import BatchNorm2d as _BatchNorm
import socket

import torch
from torch import nn
from torch.nn import functional as F
from torchvision import ops

import matplotlib.pyplot as plt
import numpy as np

from mmpose.models import builder, POSENETS
from mmpose.models.detectors.base import BasePose
from mmcv.runner import load_checkpoint

from ..core.heatmap_parsing import get_kp_graph, get_person_graph
from ..core.inference import get_pose_output
from ..core.flip_utils import _split_bottom_up_outputs, _merge_bu_flip_preds, _merge_stacked_preds, _split_flip_group_preds, _flip_back_preds

from .grouping_attention import GroupingModel
from .position_encoding import PositionEmbeddingSine
from .utils import build_basicblock_cnn, split_kp_and_person_preds


@POSENETS.register_module()
class CenterGroup(BasePose):
    def __init__(self, bu_model, bu_ckpt, group_module, kp_embed_net, person_embed_net, train_cfg, test_cfg, heatmap_cfg, pretrained=None):
        super().__init__()
        self.fp16_enabled = False
        self.bu_model = builder.build_posenet(bu_model)
        if bu_ckpt is not None and bu_ckpt != 'None':
            load_checkpoint(self.bu_model, bu_ckpt, strict=True)
        
        # TODO: Revisit
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.heatmap_cfg = heatmap_cfg

        self._freeze_bn()

        self.kp_embed_cnn=build_basicblock_cnn(kp_embed_net)
        self.person_embed_cnn=build_basicblock_cnn(person_embed_net)
        
        num_pos_embeds = group_module['kp_encoder_cfg']['dim_size'] / 2
        self.pos_embed = PositionEmbeddingSine(num_pos_feats=num_pos_embeds)
        self.group_model = GroupingModel(**group_module)

        # TODO: Revisit this
        if False:    
            self.bu_loss_fn = build_loss(self.hparams['bu_loss'])

    def train(self, mode=True):
        super().train(mode)
        self._freeze_bn()

    def _freeze_bn(self):
        if self.train_cfg['freeze_bn']:
            for m in self.bu_model.modules():
                if isinstance(m, _BatchNorm):
                    m.eval()
                    m.requires_grad_(False)


    def forward_test(self, batch, multiscale):
        flip_test = self.test_cfg['flip_test']
        batch_flip = copy.deepcopy(batch)
        #batch_flip['img'] = torch.flip(batch['img'], dims=(-1,))    
        batch_flip['pad_mask'] = [torch.flip(mask, dims=(-1,)) for mask in batch_flip['pad_mask']]

        if multiscale:
            bottom_up_outs = self.forward_bottom_up(batch, train=False)
        else:
            batch['img_metas']['aug_data'] = batch['img_metas']['aug_data'][:1]
            bottom_up_outs = self.forward_bottom_up(batch, train=False)
            for key, val in bottom_up_outs.items():
                if val:
                    assert len(val) == 1 
                    bottom_up_outs[key] = val[0]
        
        bottom_up_outs, bottom_up_outs_flip = _split_bottom_up_outputs(bottom_up_outs, batch, batch_flip)        
        if flip_test:
            bottom_up_outs = _merge_bu_flip_preds(bottom_up_outs, bottom_up_outs_flip)
        _, _, preds, person_batch = self.forward_group(**bottom_up_outs, train=False)
        
        if flip_test:
            preds, preds_flip = _split_flip_group_preds(preds, batch_idx = person_batch['batch'])        
            _, _, _, w = batch['img'].shape
            flip_index = batch['img_metas']['flip_index']
            preds_flip = _flip_back_preds(preds_flip, flip_index, w)
            preds_list = [preds, preds_flip]

            try:
                stacked_preds = {}    
                for key in preds.keys():
                    stacked_preds[key] = torch.stack([preds_[key] for preds_ in preds_list])

                # Merge Preds
                final_preds = _merge_stacked_preds(stacked_preds, person_comb= 'avg', joint_comb='avg')
                out = get_pose_output(self, final_preds, person_batch, bottom_up_outs, {'dflt': self.test_cfg}, return_for_dflt=True)

            except:
                print("COULD NOT MERGE!!!")
                out = get_pose_output(self, preds, person_batch, bottom_up_outs, {'dflt': self.test_cfg}, return_for_dflt=True)
        
        else:
            out = get_pose_output(self, preds, person_batch, bottom_up_outs, {'dflt': self.test_cfg}, return_for_dflt=True)



        return {'preds':out[0][0],
                'scores': out[0][1],
                'image_paths': [batch['img_metas']['image_file']],
                'output_heatmap': None}

    def forward(self,
                img=None,
                pad_mask=None,
                img_metas=None,
                return_loss=True,
                **kwargs):
        
        batch = {'img': img,
                 'pad_mask': pad_mask,
                 'img_metas': img_metas[0]}

        if not return_loss:
            multiscale = 'multiscale' in self.test_cfg and self.test_cfg['multiscale']            
            return self.forward_test(batch, multiscale=multiscale)
        
        else:
            return self.forward_train(batch)

    def forward_train(*args, **kwargs):
        pass

    def forward_bottom_up(self, batch, train=False):
        flip_test = not train and self.test_cfg['flip_test']
        imgs = [img_.to(batch['img'].device) for img_ in batch['img_metas']['aug_data']]
        if flip_test:
            assert batch['img'].shape[0] == 1, 'Flip test only admits batch size 1'
            imgs = [torch.cat((img, torch.flip(img, dims=(-1,))), dim=0) for img in imgs]                    

        bu_outputs = [self.bu_model(img) for img in imgs]

        fmaps_list, bu_preds_list = [], []
        for bu_output in bu_outputs:
            assert len(bu_output) == 3
            _, fmaps, bu_pred = bu_output
            fmaps_list.append(fmaps)
            bu_preds_list.append(bu_pred)

        kp_pred_list, p_pred_list = [], []
        for bu_pred in bu_preds_list:
            kp_pred, p_pred = split_kp_and_person_preds(bu_pred, num_joints=self.bu_model.keypoint_head.num_joints)
            kp_pred_list.append(kp_pred)
            p_pred_list.append(p_pred)
        
        # Parse Keypoint Nodes and extract features for them
        cnn1 = self.kp_embed_cnn
        cnn2 = self.person_embed_cnn

        kp_embed_fmaps_list = [cnn1(fmap[self.heatmap_cfg['kps']['res_ix']]) for fmap in fmaps_list]
        person_embed_fmaps_list = [cnn2(fmap[self.heatmap_cfg['persons']['res_ix']]) for fmap in fmaps_list]
        
        if flip_test:
            def split_orig_and_flip(outs):
                assert isinstance(outs, (list, tuple))
                if isinstance(outs[0], (list, tuple)):
                    return [[out_[:1] for out_ in out] for out in outs], [[out_[1:] for out_ in out] for out in outs]

                else:
                    return [out[:1] for out in outs], [out[1:] for out in outs]
            
            kp_pred_list, kp_pred_flip_list  = split_orig_and_flip(kp_pred_list)
            p_pred_list, p_pred_flip_list  = split_orig_and_flip(p_pred_list)
            kp_embed_fmaps_list, kp_embed_fmaps_flip_list  = split_orig_and_flip(kp_embed_fmaps_list)
            person_embed_fmaps_list, person_embed_fmaps_flip_list  = split_orig_and_flip(person_embed_fmaps_list)
        
        else:
            kp_pred_flip_list, p_pred_flip_list, kp_embed_fmaps_flip_list, person_embed_fmaps_flip_list = None, None, None, None

        return {'kp_pred':kp_pred_list, 
                'p_pred': p_pred_list, 
                'kp_pred_flip': kp_pred_flip_list, 
                'p_pred_flip':p_pred_flip_list, 
                'kp_embed_fmaps': kp_embed_fmaps_list, 
                'person_embed_fmaps': person_embed_fmaps_list, 
                'kp_embed_fmaps_flip':kp_embed_fmaps_flip_list, 
                'person_embed_fmaps_flip': person_embed_fmaps_flip_list}

    def forward_group(self, batch, kp_pred, p_pred, kp_pred_flip, p_pred_flip, kp_embed_fmaps, person_embed_fmaps, train=True):
        pad_mask = batch['pad_mask']
        assert len(pad_mask) ==3

        if isinstance(kp_embed_fmaps, (tuple, list)):
            positions = self.pos_embed(kp_embed_fmaps[1], (1-batch['pad_mask'][self.heatmap_cfg['kps']['res_ix']]).bool())    
        
        else:
            positions = self.pos_embed(kp_embed_fmaps, (1-pad_mask[self.heatmap_cfg['kps']['res_ix']]).bool())    
        
        kp_feats= get_kp_graph(bu_pred = kp_pred, 
                               bu_pred_flip = kp_pred_flip,
                                kp_embed_fmaps = kp_embed_fmaps, 
                                pos_embed_fmaps = positions, 
                                num_joints = self.bu_model.keypoint_head.num_joints - 1,
                                flip_index=batch['img_metas']['flip_index'],
                                batch = batch, 
                                upsample= True, 
                                test_cfg = self.bu_model.test_cfg, 
                                parsing_cfg = self.heatmap_cfg['kps']['parsing_cfg'], 
                                train=train,
                                mask_crowd_kps=False)
        
        person_feats= get_person_graph(#bu_pred = bu_pred, # Using GT for Person Nodes
                                       bu_pred = p_pred, 
                                       bu_pred_flip = p_pred_flip,
                                       person_embed_fmaps = person_embed_fmaps, 
                                       pos_embed_fmaps = positions, 
                                       batch = batch, 
                                       test_cfg = self.bu_model.test_cfg, 
                                       upsample= True, 
                                       parsing_cfg = self.heatmap_cfg['persons']['parsing_cfg'],
                                       mask_crowd_kps=False, # This will mess up keypoint extraction
                                       train=train)

        #print("Masking crowd regions out")
        kp_feats['keep_mask'] = torch.minimum(kp_feats['keep_mask'], kp_feats['kp_mask'][..., 0]).bool()
        person_feats['keep_mask'] = torch.minimum(person_feats['keep_mask'], person_feats['person_mask'][..., 0]).bool()
        
        preds = self.group_model(kp_feats=kp_feats, person_feats=person_feats)

        # Keep only non-padded predictions
        keep_mask = person_feats['keep_mask']
        #preds = {key: val[:, keep_mask[:, 0]] for key, val in preds.items()}
        preds = {key: val[:, keep_mask[:, 0]] if key.endswith('pred') or key =='vis_raw_attn_weights' else val for key, val in preds.items() }
        
        if train:
            raise NotImplementedError
            """
            assert self.bu_model.test_cfg['project2image'] #and joint_pred_res == -1
            assert not 'use_pose_iter_start' in  self.train_cfg['gt_assign']
            person_batch = get_per_person_batch(person_coords=person_feats['coords'][keep_mask], 
                                                person_batch_ix=person_feats['batch'][keep_mask].view(-1), 
                                                batch=batch, 
                                                pred_poses = preds['hard_loc_pred'][-1],
                                                person_mask =person_feats['person_mask'][person_feats['keep_mask']].view(-1),
                                                person_res_ix = self.heatmap_cfg['persons']['res_ix'],
                                                num_joints = self.bu_model.keypoint_head.num_joints,
                                                assign_method=self.train_cfg['gt_assign']['assign_method'],
                                                max_match_dist=self.train_cfg['gt_assign']['max_match_dist'],
                                                joint_pred_res=-1,
                                                joints_match_weight=0)
                                                #joints_match_weight=0 if self.global_step < self.train_cfg['gt_assign']['use_pose_iter_start'] else (self.hparams['data_cfg']['num_joints']-1)/(self.hparams['data_cfg']['num_joints']))
            """
        else:
            person_batch = {'kp_heatmaps': kp_feats['heatmaps'],
                            'person_heatmaps': person_feats['heatmaps']}

        person_batch['batch'] = person_feats['batch'][keep_mask].view(-1)
        person_batch['kp_res'] =(kp_feats['w'], kp_feats['h'])
        
        return None, None, preds, person_batch

    def show_result(self):
        raise NotImplementedError