import re

import numpy as np

import torch
from torch import nn

from mmpose.models.backbones.resnet import BasicBlock
from mmcv.runner import  _load_checkpoint, load_state_dict

def build_basicblock_cnn(cfg_):
    if cfg_['num_blocks'] == 0:
        assert cfg_['in_c'] == cfg_['out_c']
        return nn.Identity()

    elif cfg_['hidden_c'] != cfg_['in_c']:
        layers = [nn.Conv2d(cfg_['in_c'], cfg_['hidden_c'],1), 
                  nn.BatchNorm2d(cfg_['hidden_c']),
                  nn.ReLU(inplace=True)]
    else:
        layers = []

    layers += [BasicBlock(cfg_['hidden_c'], 
                            cfg_['out_c'] if l == cfg_['num_blocks']-1 else cfg_['hidden_c'])
                                                for l in range(cfg_['num_blocks'])]
    return nn.Sequential(*layers)

def split_kp_and_person_preds(outputs, num_joints):
    """Given a list of tensor with shape (B, K, *), we, with K being the number of keypoint and B the batch size, we assume that
    keypoint in position K - 1 corresponds to 'person predictions, and we return two lists of tensors, one with
    shapes (B, K-1, *), and the other one with shape (B, 1, *)
    """
    kp_outputs = []
    p_outputs = []
    for out in outputs:
        if out.shape[1] > num_joints:
            kp_outputs.append(torch.cat((out[:, :num_joints - 1], out[:, num_joints: - 1]), dim=1))
            p_outputs.append(torch.cat((out[:, num_joints-1:num_joints], out[:, -1:]), dim=1))

        else:
            kp_outputs.append(out[:, :num_joints - 1])
            p_outputs.append(out[:, -1:])

    assert len(outputs) == len(kp_outputs)    and len(outputs) == len(p_outputs)
    #assert len(outputs) == len(updated_outputs)    
    for out1, out2, out3 in zip(outputs, kp_outputs, p_outputs):
        assert out1.dim() == out2.dim() and out1.size(0) == out2.size(0) and out1.size(2) == out2.size(2) and out1.size(3) == out2.size(3)
        assert out1.dim() == out3.dim() and out1.size(0) == out3.size(0) and out1.size(2) == out3.size(2) and out1.size(3) == out3.size(3)

    return kp_outputs, p_outputs



def load_checkpoint(model,
                    filename,
                    map_location=None,
                    strict=False,
                    logger=None,
                    revise_keys=[(r'^module\.', '')]):
    """Modified MMCV's load_checkpoint to change some state_dict keys names"""

    checkpoint = _load_checkpoint(filename, map_location, logger)
    # OrderedDict is a subclass of dict
    if not isinstance(checkpoint, dict):
        raise RuntimeError(
            f'No state_dict found in checkpoint file {filename}')
    # get state_dict from checkpoint
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    # strip prefix of state_dict
    for p, r in revise_keys:
        state_dict = {re.sub(p, r, k): v for k, v in state_dict.items()}

    state_dict = {k.replace('graph_net', 'tr_encoder').replace('graph_model', 'group_model'): v  for k, v in state_dict.items()}

    # load state_dict
    load_state_dict(model, state_dict, strict, logger)
    return checkpoint

