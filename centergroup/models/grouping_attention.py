import torch
from torch import nn
from torch.nn import functional as F
from collections import defaultdict

from typing import Optional
import copy

import torch
import torch.nn.functional as F
from torch import nn, Tensor

class MLP(nn.Module):
    """ 
    Originally from https://github.com/facebookresearch/detr/blob/main/models/detr.py
    modified to not optionally apply ReLU to output
    """
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, relu_last=False):
        super().__init__()
        self.relu_last=relu_last
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 or self.relu_last else layer(x)
        
        return x

class KeypointEncoder(nn.Module):
    """Module used to add positional and type encodings to keypoint and center features"""
    def __init__(self, w_mlp, w_type, num_joints, dim_size,mlp_cfg=dict(input_dim=128, hidden_dim=128, output_dim=128, num_layers=1)):
        super().__init__()
        self.w_mlp = w_mlp
        if self.w_mlp:
            assert dim_size == mlp_cfg['output_dim']
            self.mlp = MLP(**mlp_cfg) 
        
        self.w_type = w_type
        if self.w_type:
            self.type_embed = nn.Embedding(num_embeddings = num_joints, embedding_dim=dim_size)

    def forward(self, pos_feats, joint_types):
        if self.w_mlp:
            pos_embed = self.mlp(pos_feats)

        else:
            pos_embed = pos_feats

        type_embed = self.type_embed(joint_types)
        return type_embed, pos_embed

def sanitize_mask(mask):
    if mask is None:
        return mask
    
    assert mask.dim()==2 and mask.dtype == torch.bool
    mask_ = mask.clone()
    fill_w_false = (~mask_).sum(dim=1) == 0
    mask_[fill_w_false] = False

    return mask_

class MHAttentionMap(nn.Module):
    """
    Module used to compute center-keypoint grouping (cross-)attention weights and visibility scores.
    Reused code from https://github.com/facebookresearch/detr/blob/main/models/segmentation.py
    """
    def __init__(self, query_dim, hidden_dim, num_heads, dropout=0.0, bias=True, w_value=True,
                w_pos=True):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(dropout)

        self.q_linear = nn.Linear(query_dim, hidden_dim, bias=bias)
        self.k_linear = nn.Linear(query_dim, hidden_dim, bias=bias)
        
        nn.init.zeros_(self.k_linear.bias)
        nn.init.zeros_(self.q_linear.bias)
        nn.init.xavier_uniform_(self.k_linear.weight)
        nn.init.xavier_uniform_(self.q_linear.weight)

        self.w_pos = w_pos
        self.w_value= w_value
        if self.w_value:
            self.v_linear = nn.Linear(query_dim, hidden_dim, bias=bias)
            nn.init.zeros_(self.v_linear.bias)
            nn.init.xavier_uniform_(self.v_linear.weight)

        self.normalize_fact = float(hidden_dim / self.num_heads) ** -0.5

    def forward(self, q, k, v=None, q_embed=None, kv_embed=None, mask= None):
        #with torch.no_grad():

        q = self.q_linear(q + q_embed.view(q.shape) if self.w_pos else q)
        k = self.k_linear(k + kv_embed.view(k.shape) if self.w_pos else k)

        qh = q.view(q.shape[0], q.shape[1], self.num_heads, self.hidden_dim // self.num_heads)
        kh = k.view(k.shape[0], k.shape[1], self.num_heads, self.hidden_dim // self.num_heads)
        weights = torch.einsum("bqnc,bknc->bqnk", qh * self.normalize_fact, kh)

        if mask is not None:
            weights.masked_fill_(mask.unsqueeze(1).unsqueeze(1), float("-inf"))

        weights_raw = weights.clone()
        
        weights = F.softmax(weights, dim=-1)
        weights = self.dropout(weights)

        if self.w_value:
            v = self.v_linear(v)
            vh = v.view(v.shape[0], v.shape[1], self.num_heads, self.hidden_dim // self.num_heads)
            result = torch.einsum('bqnk,bknc->bqnc', weights, vh)

            return weights, result, weights_raw
        else:
            return weights,  None, weights_raw


class PredictionHead(nn.Module):
    """Head used to predict 1) Center-Keypoint Attention Weights 2) Keypoint visibility scores 3) Personness scores
    from center and keypoint embeddings"""
    def __init__(self, dim_size, hidden_dim, num_joints, mlp_viz_cfg, mlp_person_tp_cfg, w_pos, vis_from_attn=False, use_type_mask=False,vis_attn=False, share_loc_vis_attn=True):
        super().__init__()
        self.vis_from_attn=vis_from_attn
        self.use_type_mask=use_type_mask
        self.loc_attn  = MHAttentionMap(dim_size, hidden_dim*num_joints, num_heads=num_joints, w_value=vis_attn and share_loc_vis_attn, w_pos=w_pos)
        
        self.vis_attn=vis_attn
        self.share_loc_vis_attn=share_loc_vis_attn
        if self.vis_attn:
            if not self.share_loc_vis_attn:
                raise RuntimeError("No loc/vis attention weight sharing is not supported")

            self.mlp_vis =MLP(input_dim=dim_size + hidden_dim, hidden_dim=dim_size,
                              output_dim=1, num_layers=mlp_viz_cfg['num_layers'])
        else:
            self.mlp_vis = MLP(input_dim=dim_size, hidden_dim=num_joints,
                                output_dim=num_joints, num_layers=mlp_viz_cfg['num_layers'])

        self.mlp_person_tp = MLP(input_dim=dim_size, hidden_dim=dim_size,
                                 output_dim=1, num_layers=mlp_person_tp_cfg['num_layers'])
        
    def forward(self, kp_nodes, p_nodes, kp_coords, mask=None, p_embed=None, kp_embed=None, type_mask=None):
        flatten = lambda x: torch.flatten(x, 1, 2)                        

        if kp_nodes.dim() ==4:
            kp_nodes =flatten(kp_nodes)

        if p_nodes.dim() ==4:
            p_nodes =flatten(p_nodes)

        if kp_coords is not None and kp_coords.dim() == 4:
            kp_coords = flatten(kp_coords)

        assert kp_coords.dim() == 3 and kp_nodes.shape[0] == kp_coords.shape[0] and  kp_nodes.shape[1] == kp_coords.shape[1]

        if mask.dim() ==3:
            mask =flatten(mask)

        mask_ = sanitize_mask(mask)
        person_tp_pred = self.mlp_person_tp(p_nodes)
         
        _, result_w_val, loc_attn_weights_raw = self.loc_attn(q=p_nodes, k=kp_nodes, v=kp_nodes, q_embed = p_embed, kv_embed=kp_embed, mask=mask_)

        if self.use_type_mask:
            raise RuntimeError("Type mask not supported")

        if self.vis_attn:
            vis_mssgs = result_w_val
            p_nodes_ = p_nodes.unsqueeze(2).expand(-1, -1, vis_mssgs.size(-2), -1)
            p_nodes_ = torch.cat((p_nodes_, vis_mssgs), dim=-1)    
            vis_pred = self.mlp_vis(p_nodes_)

        else:                
            vis_pred = self.mlp_vis(p_nodes).unsqueeze(-1)
        
        loc_attn_weights = loc_attn_weights_raw.softmax(-1)

        loc_pred = torch.einsum('bqnk,bkc->bqnc', loc_attn_weights, kp_coords.type(loc_attn_weights.dtype))
        with torch.no_grad(): # Use argmax instead of softmax
            hard_attn = (loc_attn_weights == torch.max(loc_attn_weights, -1)[0].unsqueeze(-1)).type(loc_attn_weights.dtype)
            hard_attn = hard_attn/hard_attn.sum(-1, keepdim=True)
            hard_loc_pred = torch.einsum('bqnk,bkc->bqnc', hard_attn, kp_coords.type(hard_attn.dtype))
                
        return {'loc_pred': loc_pred,
                'person_tp_pred': person_tp_pred,
                'hard_loc_pred': hard_loc_pred,
                'viz_pred': vis_pred,
                'loc_attn_weights': loc_attn_weights,
                'loc_attn_weights_raw': loc_attn_weights_raw}


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):

        def _get_activation_fn(activation):
            """Return an activation function given a string"""
            if activation == "relu":
                return F.relu
            if activation == "gelu":
                return F.gelu
            if activation == "glu":
                return F.glu
            raise RuntimeError(F"activation should be relu/gelu, not {activation}.")
        super().__init__()
        print("NHEADS", nhead)
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     src,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def forward(self, src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class TransformerEncoder_(nn.Module):
    """Modified TransformerEncoder with additional skip connections. They are most likely not needed but they are kept for
    reproducibility."""
    def __init__(self, layer_cfg, num_layers, skip_conns=False):
        super().__init__()
        encoder_layer = TransformerEncoderLayer(**layer_cfg)

        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.skip_conns = skip_conns

    def prepare_inputs(self, kp_nodes, kp_embed, kp_mask):
        if kp_embed is not None:
            assert kp_nodes.shape == kp_embed.shape
        if kp_nodes.dim() == 4:
            kp_nodes_ = kp_nodes.flatten(1, 2)
            if kp_embed is not None:
                kp_embed_ = kp_embed.flatten(1, 2)

            else: 
                kp_embed_=None

        else:
            assert kp_nodes.dim() ==3
            kp_nodes_ = kp_nodes.clone()
            if kp_embed is not None:
                kp_embed_ = kp_embed.clone()
            else: 
                kp_embed_=None

        if kp_mask.dim()== 3:
            kp_mask_ = kp_mask.flatten(1, 2)

        else:
            assert kp_mask.dim() == 2
            kp_mask_ = kp_mask.clone()

        assert kp_mask_.shape[0] == kp_nodes_.shape[0], "mask and keypoints do not have same Batch size"
        assert kp_mask_.shape[1] == kp_nodes_.shape[1], "mask and keypoints do not have same number of keypoints"

        kp_nodes_ = kp_nodes_.permute(1, 0, 2) # (B, N keypoints, Dim) --> (N keypints, B, Dim)
        if kp_embed is not None:
            kp_embed_ = kp_embed_.permute(1, 0, 2)
            assert kp_embed_.shape == kp_nodes_.shape
            assert (kp_embed_ != kp_nodes_).any()
        
        kp_mask_ = sanitize_mask(kp_mask_)

        return kp_nodes_, kp_embed_, kp_mask_


    def forward(self, kp_nodes, kp_mask = None, kp_embed = None):        
        original_node_shape = kp_nodes.shape
        kp_nodes_, kp_embed_, kp_mask_ = self.prepare_inputs(kp_nodes, kp_embed, kp_mask)
        output = kp_nodes_

        layer_outputs = []
        for layer in self.layers:
            layer_output = layer(output, src_mask=None,
                           src_key_padding_mask=kp_mask_, pos=kp_embed_)
            
            if self.skip_conns:
                output = layer_output + output
            
            else:
                output = layer_output

            layer_outputs.append(output.permute(1, 0, 2).view(original_node_shape))

        return torch.stack(layer_outputs)


class GroupingModel(nn.Module):
    """
    Main class implementing the grouping pipeline.
    It receives keypoint and center features, applies MLP and adds positional + type encodings, applies a transformer to them, and finally computes
    cross-attention between center and keypoint fetures to determine the grouping coefficients.
    """
    def __init__(self, kp_encoder_cfg, head_cfg, transformer_cfg=None, initial_mlp_cfg=None):
        super().__init__()
        self.initial_mlp=MLP(**initial_mlp_cfg)        
        self.kp_encoder = KeypointEncoder(**kp_encoder_cfg)

        if transformer_cfg is not None and transformer_cfg['num_layers'] >0:
            self.tr_encoder = TransformerEncoder_(**transformer_cfg)
        
        else:
            self.tr_encoder = None
        
        if head_cfg is not None:
            num_head_layers = transformer_cfg['num_layers'] + 1
            share_weights = head_cfg.pop('share_weights')
            self.predict_first = head_cfg.pop('predict_first')
            if share_weights:
                print("WEIGHT SHARING in HEAD!!!")    
                self.prediction_heads = nn.ModuleList(num_head_layers*[PredictionHead(**head_cfg)])
            
            else:
                print("NO WEIGHT SHARING!!!")    
                head_layer = PredictionHead(**head_cfg)
                self.prediction_heads = _get_clones(head_layer, num_head_layers)

    def forward(self, kp_feats, person_feats):
        with torch.no_grad():
            assert person_feats['kp_type'].max() == 0                
            
        p_type_embed, p_pos_embed =self.kp_encoder(pos_feats = person_feats['person_pos'], joint_types=person_feats['kp_type'] + self.kp_encoder.type_embed.num_embeddings-1)
        kp_type_embed, kp_pos_embed = self.kp_encoder(pos_feats = kp_feats['kp_pos_embed'], joint_types=kp_feats['kp_type'])

        kp_embed= kp_pos_embed + kp_type_embed
        p_embed = p_pos_embed + p_type_embed

        p_nodes = person_feats['person_embed']
        kp_nodes = kp_feats['kp_embed'] 
        
        if self.initial_mlp is not None:
            p_nodes = self.initial_mlp(p_nodes)
            kp_nodes = self.initial_mlp(kp_nodes)

        kp_mask = ~kp_feats['keep_mask']
        p_mask = ~person_feats['keep_mask']

        preds = defaultdict(list)
        if self.predict_first:            
            pred = self.prediction_heads[0](kp_nodes = kp_nodes, p_nodes= p_nodes, kp_coords = kp_feats['coords'], mask=kp_mask,
                                            p_embed = p_embed, kp_embed = kp_embed, type_mask=None)
            for key, val in pred.items():
                preds[key].append(val)

        if self.tr_encoder:
            assert kp_nodes.dim()==4 and p_nodes.dim() == 4
            assert kp_embed.dim()==4 and p_embed.dim() == 4
            assert kp_mask.dim()==3 and p_mask.dim() == 3
            nodes = torch.cat((kp_nodes, p_nodes), dim=1)
            embeds = torch.cat((kp_embed, p_embed), dim=1)
            masks = torch.cat((kp_mask, p_mask), dim=1)

            encoder_out = self.tr_encoder(kp_nodes = nodes, kp_embed = embeds, kp_mask = masks)
            kp_outputs, p_outputs  = encoder_out[:, :, :-1], encoder_out[:, :, -1:]
        
        else:
            kp_outputs, p_outputs = [], []

        for kp_out, p_out, pred_head in zip(kp_outputs, p_outputs, self.prediction_heads[1:]):
            pred = pred_head(kp_nodes = kp_out, p_nodes= p_out, kp_coords = kp_feats['coords'], mask=kp_mask,
                             p_embed = p_embed, kp_embed = kp_embed, type_mask=None)
            for key, val in pred.items():
                preds[key].append(val)
        
        for key, val in preds.items():
            preds[key] = torch.stack(val)

        return preds