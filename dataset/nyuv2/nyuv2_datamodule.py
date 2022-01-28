"""
LightningDataModule for NYUv2 Dataset
https://pytorch-lightning.readthedocs.io/en/latest/extensions/datamodules.html
"""
import os
from typing import Optional
import albumentations as A
from albumentations.pytorch import ToTensorV2

import pytorch_lightning as pl
#from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from torch.utils.data import random_split, DataLoader

from .nyuv2_dataset import NYUv2Dataset


class NYUv2DataModule(pl.LightningDataModule):
    # TODO
    def __init__(self):
        super().__init__()
        self.dataset = NYUv2Dataset(root=os.path.join(os.getcwd(), "data", "NYUv2"),
                                          rgb_transform=self._rgb_transform(train=True),
                                          seg_transform=self._semantic_seg_transform(train=True),
                                          instance_transform=self._instance_seg_transform(train=True),
                                          depth_transform=self._depth_transform(train=True),
                                          train=True)

        train_len = int(len(self.dataset)*0.8)
        val_len = len(self.dataset) - train_len

        self.train_dataset, self.val_dataset = random_split(self.dataset, [train_len, val_len])

        self.val_dataset.set_transforms(rgb=self._rgb_transform(train=False),
                                          seg=self._semantic_seg_transform(train=False),
                                          inst=self._instance_seg_transform(train=False),
                                          depth=self._depth_transform(train=False))

        self.test_dataset = NYUv2Dataset(root=os.path.join(os.getcwd(), "data", "NYUv2"),
                                          rgb_transform=self._rgb_transform(train=False),
                                          seg_transform=self._semantic_seg_transform(train=False),
                                          instance_transform=self._instance_seg_transform(train=False),
                                          depth_transform=self._depth_transform(train=False),
                                          train=False)

    def _rgb_transform(self, train):
        """
        Transformations for RGB images.
        :param train: Train or test.
        :return:
        """
        # TODO
        if train:
            return A.Compose([
                A.Resize(160, 120),
                ToTensorV2()
            ])
        else:
            return ToTensorV2()

    def _depth_transform(self, train):
        # TODO
        if train:
            return A.Compose([
                A.Resize(160, 120),
                ToTensorV2()
            ])
        else:
            return ToTensorV2()

    def _semantic_seg_transform(self, train):
        # TODO
        if train:
            return A.Compose([
                A.Resize(160, 120),
                ToTensorV2()
            ])
        else:
            return ToTensorV2()

    def _instance_seg_transform(self, train):
        # TODO
        if train:
            return A.Compose([
                A.Flip(),
                A.Resize(160, 120),
                ToTensorV2()
            ])
        else:
            return ToTensorV2()

    def prepare_data(self) -> None:
        # download
        super().prepare_data()

    def setup(self, stage: Optional[str] = None) -> None:
        # Assign train/val datasets for use in dataloaders
        super().setup(stage)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=32)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=32)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=32)

    def predict_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=32)

