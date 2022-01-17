import os.path as osp
import copy
from copy import deepcopy     


from time import time
from torch.nn import BatchNorm2d as _BatchNorm
import socket

import torch


from mmpose.models import builder, POSENETS
from mmpose.models.detectors.base import BasePose
from mmpose.apis.test import collect_results_gpu
from mmcv.runner import load_checkpoint
from pytorch_lightning import LightningModule

from ..core.heatmap_parsing import get_kp_graph, get_person_graph
from ..core.train_utils import generate_person_targets, compute_group_loss
from ..core.inference import get_pose_output
from ..core.flip_utils import _split_bottom_up_outputs, _merge_bu_flip_preds, _merge_stacked_preds, _split_flip_group_preds, _flip_back_preds

from .grouping_attention import GroupingModel
from .position_encoding import PositionEmbeddingSine
from .utils import build_basicblock_cnn, split_kp_and_person_preds

list2fp32 = lambda x: [val_.float() for val_ in x]
dict2fp32 =lambda x: {key: list2fp32(val) if isinstance(val, list) and isinstance(val[0], torch.Tensor) and val[0].dtype== torch.half else val for key, val in x.items()}

@POSENETS.register_module()
class CenterGroup(BasePose, LightningModule):
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

        self.save_hyperparameters()


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
        preds, person_batch = self.forward_group(**bottom_up_outs, train=False)
        
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
                 'pad_mask': pad_mask}
        try:
            batch['img_metas']= img_metas[0]

        except:
            batch['img_metas']= img_metas.data[0][0]

        if not return_loss: # Force use of FP32 for validation
            batch['img'] = batch['img'].float()
            batch['img_metas']['aug_data'] =  list2fp32(batch['img_metas']['aug_data'])
            batch = dict2fp32(batch)
            multiscale = 'multiscale' in self.test_cfg and self.test_cfg['multiscale']            
            return self.forward_test(batch, multiscale=multiscale)
        
        else:
            return self.forward_train(batch) 

    def forward_train(*args, **kwargs): 
        # See 'training_step'
        # MMPose requires this method, but we use PyTorch Lightning for training.
        pass
    
    def validation_step(self, batch, batch_idx):
        return self.forward(**batch, return_loss=False)
    
    def validation_epoch_end(self, outputs):
        if self.trainer.test_dataloaders:
            dataset = self.trainer.test_dataloaders[0].dataset

        else:
            dataset =  self.trainer.val_dataloaders[0].dataset

        all_out = collect_results_gpu(outputs, len(dataset))

        if self.global_rank == 0:
            results = dataset.evaluate(all_out, self.logger.root_dir)
            self._log_metrics(results, 'val', prefix = None)

            return results

    def test_step(self, *args, **kwargs):
        return self.validation_step(*args, **kwargs)
    
    def test_epoch_end(self, *args, **kwargs):
        return self.validation_epoch_end(*args, **kwargs)

    def forward_bottom_up(self, batch, train=False):
        flip_test = not train and self.test_cfg['flip_test']
        if 'aug_data' in batch['img_metas']:
            imgs = [img_.to(batch['img'].device) for img_ in batch['img_metas']['aug_data']]
        
        else:
            imgs = [batch['img']]

        if flip_test:
            assert batch['img'].shape[0] == 1, 'Flip test only admits batch size 1'
            imgs = [torch.cat((img, torch.flip(img, dims=(-1,))), dim=0) for img in imgs]                    

        bu_outputs = [self.bu_model(img) for img in imgs]

        fmaps_list, bu_pred_list = [], []
        for bu_output in bu_outputs:
            assert len(bu_output) == 3
            _, fmaps, bu_pred = bu_output
            fmaps_list.append(fmaps if train else list2fp32(fmaps)) # Force using fp32 for evaluation
            bu_pred_list.append(bu_pred if train else list2fp32(bu_pred))

        kp_pred_list, p_pred_list = [], []
        for bu_pred in bu_pred_list:
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

        out = {'kp_pred':kp_pred_list, 
                'p_pred': p_pred_list, 
                'kp_pred_flip': kp_pred_flip_list, 
                'p_pred_flip':p_pred_flip_list, 
                'kp_embed_fmaps': kp_embed_fmaps_list, 
                'person_embed_fmaps': person_embed_fmaps_list, 
                'kp_embed_fmaps_flip':kp_embed_fmaps_flip_list, 
                'person_embed_fmaps_flip': person_embed_fmaps_flip_list}
        
        if train:
            out['bu_pred'] = bu_pred_list
        
        else:
            out = dict2fp32(out) # FP32 for evaluation
        
        return out

    def forward_group(self, batch, kp_pred, p_pred, kp_pred_flip, p_pred_flip, kp_embed_fmaps, person_embed_fmaps, train=True, **kwargs):
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
            person_batch = generate_person_targets(dt_centers=person_feats['coords'][keep_mask], 
                                                   person_batch_ix=person_feats['batch'][keep_mask].view(-1), 
                                                   batch=batch, 
                                                   person_mask =person_feats['person_mask'][person_feats['keep_mask']].view(-1),
                                                   assign_method=self.train_cfg['gt_assign']['assign_method'], # greedy
                                                   max_match_dist=self.train_cfg['gt_assign']['max_match_dist'], # 0.8
                                                   joint_pred_res=-1)

            # Not sure if needed:
            person_batch['coords'] = person_feats['coords'][keep_mask]
            person_batch['batch'] = person_feats['batch'][keep_mask].view(-1)
            person_batch['kp_heatmaps'] = kp_feats['heatmap']
            person_batch['kp_res'] =(kp_feats['w'], kp_feats['h'])
            person_batch['kp_score'] = kp_feats['score']
            person_batch['person_hmap_score'] = person_feats['score']
            person_batch['kp_keep_mask'] = kp_feats['keep_mask']
            person_batch['person_keep_mask']  = keep_mask
            person_batch['kp_coords'] = kp_feats['coords']
        else:
            person_batch = {'kp_heatmaps': kp_feats['heatmaps'],
                            'person_heatmaps': person_feats['heatmaps']}

        person_batch['batch'] = person_feats['batch'][keep_mask].view(-1)
        person_batch['kp_res'] =(kp_feats['w'], kp_feats['h'])
        
        return preds, person_batch

    def show_result(self):
        raise NotImplementedError

    def training_step(self, batch, batch_idx):
        batch['img_metas'] = batch['img_metas'].data[0][0]
        # Forward Pass
        bu_outs = self.forward_bottom_up(batch, train = True)
        bu_outs = {key: val[0] if val else val for key, val in bu_outs.items()}
        preds, person_batch = self.forward_group(batch = batch, **bu_outs, train = True)

        # Loss computation
        losses = compute_group_loss(preds, person_batch['vis_target'], person_batch['loc_target'],
                                    person_tp_target = person_batch['is_tp'], 
                                    ignore_person=person_batch['ignore'],
                                    boxes = person_batch['boxes'],
                                    loss_cfg=self.train_cfg['group_loss'])        
                                        
        if self.train_cfg['bu_loss_factor'] > 0:
            bu_pred = bu_outs['bu_pred']
            bu_losses = self.bu_model.keypoint_head.get_loss(bu_outs['bu_pred'], batch['targets'][:len(bu_pred)], batch['masks'][:len(bu_pred)], None)
            losses['heatmap_loss'] = bu_losses['heatmap_loss']
            losses['overall'] += self.train_cfg['bu_loss_factor'] * losses['heatmap_loss']
        
        self._log_metrics(losses, train_val = 'train', prefix = 'loss')

        return losses['overall']

    def configure_optimizers(self):
        optim_class = getattr(torch.optim, self.train_cfg['optimizer_']['type'])
        optimizer = optim_class([{'params': self.bu_model.parameters(),
                                  'lr': self.train_cfg['optimizer_']['bu_lr']},
                                 {'params': self.kp_embed_cnn.parameters(),
                                  'lr': self.train_cfg['optimizer_']['lr']},
                                 {'params': self.person_embed_cnn.parameters(),
                                  'lr': self.train_cfg['optimizer_']['lr']},
                                 {'params': self.group_model.parameters(),
                                  'lr': self.train_cfg['optimizer_']['lr']}
                                  ] )
        #print("LR MILESTONES", self.train_cfg['lr_scheduler_']['milestones'])
        lr_scheduler =  torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.train_cfg['lr_scheduler_']['milestones'])
        return [optimizer], [lr_scheduler]

    def _log_metrics(self, loss_dict, train_val, prefix = 'loss'):
        for loss_name, loss_val in  loss_dict.items():
            log_str = f"{loss_name}/{train_val}"
            if prefix:
                log_str = f"{prefix}/{log_str}"
            self.log(log_str, loss_val)

    def optimizer_step(
        self,
        epoch: int,
        batch_idx,
        optimizer,
        optimizer_idx: int,
        optimizer_closure,
        on_tpu: bool,
        using_native_amp: bool,
        using_lbfgs: bool,
    ) :
        if self.train_cfg['train_cnn'] and self.train_cfg['lr_warmup']['do_warmup'] and self.trainer.global_step < self.train_cfg['lr_warmup']['warmup_iters']:
            k = (1 - self.trainer.global_step / self.train_cfg['lr_warmup']['warmup_iters']) * (1 - self.train_cfg['lr_warmup']['warmup_ratio'])

            for _, pg in enumerate(optimizer.param_groups):
                if _ == 0 and self.train_cfg['lr_warmup']['do_warmup']:
                    pg['lr'] = (1 - k) * self.train_cfg['optimizer_']['bu_lr']
                elif _ == 1 and self.train_cfg['lr_warmup']['do_warmup']:
                    pg['lr'] = (1 - k) * self.train_cfg['optimizer_']['lr']

        super().optimizer_step(
                epoch,
                batch_idx,
                optimizer,
                optimizer_idx,
                optimizer_closure,
                on_tpu,
                using_native_amp,
                using_lbfgs
        )