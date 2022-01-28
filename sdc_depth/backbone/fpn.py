"""
Feature Pyramid Network backbone.
https://pytorch.org/vision/stable/_modules/torchvision/ops/feature_pyramid_network.html
https://github.com/qubvel/segmentation_models.pytorch#architectures-
"""
from typing import Any

import pytorch_lightning as pl
import segmentation_models_pytorch as smp
import torch
from torch import nn
from torchsummaryX import summary


class FPNBackbone(nn.Module):
    @property
    def example_input_array(self):
        return torch.rand(3, 256, 256)

    def __init__(
        self,
        out_channels: int,
        encoder_name="resnet50",
        *args: Any,
        **kwargs: Any,
    ):
        """
        :param out_channels: Number of channels of output mask
        :param encoder_name: Name of the classification model that will be used as an encoder (a.k.a backbone) to
        extract features of different spatial resolution
         https://github.com/qubvel/segmentation_models.pytorch#encoders-
        """
        # https://smp.readthedocs.io/en/latest/models.html#fpn
        super().__init__()

    def _build_model(self, coco_pretrained: bool, encoder_pretrained: bool):
        if coco_pretrained == encoder_pretrained:
            raise ValueError(
                "Either coco_pretrained or encoder_pretrained must be set."
            )

        if coco_pretrained:
            # load torchvision model
            # Get the FPN part of the Mask R-CNN ResNet-50 FPN model
            pass
        elif encoder_pretrained:
            # load smp model
            # SMP only has pretrained weights for encoders
            pass

    def forward(self, x):
        """
        Sequentially pass `x` through model`s encoder and decoder
        Copied from segmentation_models_pytorch.SegmentationModel
        """
        features = self.encoder(x)
        decoder_output = self.decoder(*features)
        masks = self.segmentation_head(decoder_output)
        # segmentation_models_pytorch implementation does not return features
        return features, masks


def main():
    m = FPNBackbone(512, "resnet50")
    m.to("cuda")
    summary(m, m.example_input_array.unsqueeze(0).cuda())


if __name__ == "__main__":
    main()
