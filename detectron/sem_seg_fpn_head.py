from detectron2.modeling.meta_arch.semantic_seg import (
    SEM_SEG_HEADS_REGISTRY,
    SemSegFPNHead,
)

from torch.nn import functional as F

# https://github.com/facebookresearch/detectron2/blob/main/detectron2/modeling/meta_arch/semantic_seg.py


@SEM_SEG_HEADS_REGISTRY.register()
class RefactoredSemSegFPNHead(SemSegFPNHead):
    def forward(self, features, targets=None):
        """
        Returns:
            In training, returns (CxHxW logits, dict of losses)
            In inference, returns (CxHxW logits, {})
        """
        # Actual implementation does not return predicted maps at training
        # Changed: Return logits in training as well
        x = self.layers(features)
        if self.training:
            losses = self.losses(x, targets)
        else:
            losses = {}
        x = F.interpolate(
            x, scale_factor=self.common_stride, mode="bilinear", align_corners=False
        )
        return x, losses
