import torch
from mmpose.models.heads.ae_higher_resolution_head import AEHigherResolutionHead
from mmpose.models.builder import HEADS

@HEADS.register_module()
class AEHigherResolutionHeadWithRoot(AEHigherResolutionHead):
    """Modified HigherHRNet head to accomodate center predictions. 
    At test-time, unless self.remove_center_test=False, the network will ignore predictions corresponding to centers 
    (heatmaps at last position). This is done to enable testing the model ignoring center predictions.

    Note that our modified AssociativeEmbedding_ detector does not use the forward of this class directly.
    It is only used to pretrain the HigherHRNet head.
    """
    def __init__(self, remove_center_test, **kwargs):
        super().__init__(**kwargs)
        self.remove_center_test = remove_center_test
        self.num_joints = kwargs['num_joints']
    
    def forward(self, *args, **kwargs):
        if self.training or not self.remove_center_test:
            return super().forward(*args, **kwargs)

        else:
            outputs= super().forward(*args, **kwargs)            
            assert isinstance(outputs, (list, tuple))
            
            updated_outputs = []
            for out in outputs:
                if out.shape[1] > self.num_joints:
                    updated_outputs.append(torch.cat((out[:, :self.num_joints - 1], out[:, self.num_joints: - 1]), dim=1))

                else:
                    updated_outputs.append(out[:, :self.num_joints - 1])

            assert len(outputs) == len(updated_outputs)      
            for out1, out2 in zip(outputs, updated_outputs):
                assert out1.dim() == out2.dim() and out1.size(0) == out2.size(0) and out1.size(2) == out2.size(2) and out1.size(3) == out2.size(3)
            return updated_outputs