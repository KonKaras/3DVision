from typing import Any, List, Optional

import pytorch_lightning as pl
import torch
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch import nn
from torch.nn import Module

from detectron.panoptic_fpn import RefactoredPanopticFPN


class PanopticBinsModel(pl.LightningModule):
    def __init__(
        self,
        *args: Any,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)
        self.panoptic_backbone, self.panoptic_cfg = RefactoredPanopticFPN.build() #we only need pretrained output, change this pretrained model from zoo
        self.freeze_backbone()

    def freeze_backbone(self):
        self.panoptic_backbone.eval()  # Freeze backbone weights for training -> either detectron2 or mask2former
        for param in self.panoptic_backbone.parameters():
            param.requires_grad = False

    #optional command line
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("PanopticBinsModel")
        # TODO: Add command line args here
        # parser.add_argument("--encoder_layers", type=int, default=12)
        return parent_parser

    def configure_optimizers(self):
        pass

    def forward(self, x) -> Any:
        panoptic_output_batch = self.panoptic_backbone(x)

        batch_features = []
        batch_sem_seg_ids = []
        batch_instance_ids = []
        batch_category_ids = []

        for data_point in panoptic_output_batch:
            #load semantic output of panoptic model
            sem_seg_scores = data_point["sem_seg"]  # n_classes x H x W
            batch_sem_seg_ids.append(
                torch.argmax(sem_seg_scores, dim=0)
            )  # H x W segmentation ids

            batch_features.append(
                data_point["p2"] #p2?
            )  # Multiple features can also be used
            batch_instance_ids.append(
                data_point["panoptic_seg"][0]
            )  # H x W instance ids
            batch_category_ids.append(build_category_id_map(data_point["panoptic_seg"]))
            # H x W category ids
            # TODO: One can also do # H x W x N_classes with binary values

        # Convert lists to batch of tensors
        batch_features = torch.stack(batch_features)  # TODO: might need upsampling
        batch_sem_seg_ids = torch.stack(batch_sem_seg_ids)
        batch_instance_ids = torch.stack(batch_instance_ids)
        batch_category_ids = torch.stack(batch_category_ids)

        # TODO: Positional encoding for labels

        # panoptic_output structure:
        # 'sem_seg' : N_classes x H x W
        # features: {'p2': Tensor, 'p3': Tensor...}

    def training_step(self, *args, **kwargs) -> STEP_OUTPUT:
        return super().training_step(*args, **kwargs)

    def validation_step(self, *args, **kwargs) -> Optional[STEP_OUTPUT]:
        return super().validation_step(*args, **kwargs)


class PositionalLabelEncoding(nn.Module):

    def __init__(self,num_labels):
        


@torch.no_grad()
def build_category_id_map(panoptic_seg_output) -> torch.tensor:
    instance_id_map: torch.tensor = panoptic_seg_output[0]
    instances_list: List = panoptic_seg_output[1]
    category_id_map = torch.zeros_like(
        instance_id_map
    )  # TODO: should we fill this with zeros or some other
    # category id? Like for example background
    for instance in instances_list:
        instance_id = instance["instance_id"]
        category_id = instance["category_id"]
        category_id_map[instance_id_map == instance_id] = category_id
    return category_id_map
