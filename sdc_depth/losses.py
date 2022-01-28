"""
A collection of losses throughout the modules
"""
import pytorch_lightning as pl
import torch.nn as nn


def depth_prediction_loss(x, target, dataset="nyuv2"):
    if dataset == "nyuv2":
        # I guess this has to do with L2 being more prone to outliers in the data
        return nn.L1Loss()(x, target)
    else:
        # TODO needs 3 proper params
        return nn.MarginRankingLoss()(x, x, target)


def semantic_segmentation_loss(x, target):
    # TODO
    return 0.0


def instance_segmentation_loss(x, target):
    # TODO
    return 0.0


def combined_loss(x, target, dataset="nyuv2", w_Inst=1.0, w_Segm=1.0, w_Depth=1.0):
    loss_i = w_Inst * instance_segmentation_loss(x, target)
    loss_s = w_Segm * semantic_segmentation_loss(x, target)
    loss_d = w_Depth * depth_prediction_loss(x, target, dataset)
    return loss_i + loss_s + loss_d
