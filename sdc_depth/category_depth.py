"""
Category-wise Depth Estimation fully convolutional network (FCN) module.
"""
from typing import Any

import segmentation_models_pytorch as smp
import torch
from torch import nn
from torchsummaryX import summary


class CategoryDepthEstimator(nn.Module):
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
        super().__init__(*args, **kwargs)
        self.fpn = smp.FPN(
            encoder_name=encoder_name,
            encoder_depth=5,
            encoder_weights=None,
            decoder_pyramid_channels=256,
            decoder_segmentation_channels=128,
            decoder_dropout=0.2,
            classes=out_channels,
            upsampling=4,
        )

    def forward(self, x):
        return self.fpn(x)
