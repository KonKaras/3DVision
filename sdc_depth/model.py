"""
SDC-Depth LightningModule.
"""

import os
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl


class SDCDepth(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.backbone = None
        self.semantic_seg_model = None
        self.instance_seg_model = None
        self.category_depth_model = None
        self.instance_depth_model = None
        self.depth_aggregator = None

    def forward(self, x):
        # in lightning, forward defines the prediction/inference actions
        backbone_features = self.backbone(x)
        semantic_seg = self.semantic_seg_model(x)
        instance_seg = self.instance_seg_model(x)
        category_depth = self.category_depth_model(backbone_features, semantic_seg)
        instance_depth = self.instance_depth_model(backbone_features, instance_seg)
        global_depth = self.depth_aggregator(
            semantic_seg, instance_seg, category_depth, instance_depth
        )

        return {
            "semantic_seg": semantic_seg,
            "instance_seg": instance_seg,
            "category_depth": category_depth,
            "instance_depth": instance_depth,
            "global_depth": global_depth,
        }

    def training_step(self, batch, batch_idx):
        # TODO
        return None
