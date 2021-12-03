import torch
from mmpose.models.detectors.associative_embedding import AssociativeEmbedding
from mmpose.models.builder import POSENETS
from mmcv.runner import auto_fp16

@POSENETS.register_module()
class AssociativeEmbedding_(AssociativeEmbedding):
    """
    Modified forward to not have different behaviors in train/test 
    and to return both final heatmaps as well as the last feature map before heatmaps
    """
    @auto_fp16(apply_to=('img', ))
    def forward(self,
                img=None):
        x = self.backbone(img)
        assert isinstance(x, (tuple, list))
        if len(x) >1:
            #print("Got multiscale output!")
            ms_output, x = x
        else:
            #print("Got no multiscale output")
            ms_output=None

        feature_maps, bottom_up_outputs = self.bottom_up_kp_head_forward(x)

        return ms_output, feature_maps, bottom_up_outputs


    def bottom_up_kp_head_forward(self, x):
        """Forward function."""
        if isinstance(x, list):
            x = x[0]

        feature_maps = []
        final_outputs = []
        y = self.keypoint_head.final_layers[0](x)
        feature_maps.append(x)
        final_outputs.append(y)

        for i in range(self.keypoint_head.num_deconvs):
            if self.keypoint_head.cat_output[i]:
                x = torch.cat((x, y), 1)

            x = self.keypoint_head.deconv_layers[i](x)
            y = self.keypoint_head.final_layers[i + 1](x)
            feature_maps.append(x)
            final_outputs.append(y)

        return feature_maps, final_outputs