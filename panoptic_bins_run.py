import argparse

import configargparse
import cv2
import torch
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultPredictor
from detectron2.model_zoo import model_zoo
from detectron2.modeling import build_model
from detectron2.utils.visualizer import Visualizer
from matplotlib import pyplot as plt

from dataset.nyuv2_raw.bts_dataloader import DataLoadPreprocess, ToTensorHelper
from dataset.nyuv2_raw.nyuv2_raw_datamodule import NYUv2RawDataModule
from dataset.utils import ImgToFloatTensor
from detectron.panoptic_fpn import RefactoredPanopticFPN

import detectron2.data.transforms as T

from detectron.sem_seg_fpn_head import RefactoredSemSegFPNHead
import torchvision.transforms as TT

from train import add_train_args


def dataload():
    ds = DataLoadPreprocess(mode="train")


def detectron_predictor():
    cfg = get_cfg()  # Get default config
    model_cfg_path = "COCO-PanopticSegmentation/panoptic_fpn_R_50_3x.yaml"
    cfg_file = model_zoo.get_config_file(model_cfg_path)
    cfg.merge_from_file(cfg_file)

    # Normal implementation of some submodules do not return predicted
    # results during training.
    # -> Reimplement their forward functions to return the results
    # as they are needed for downstream tasks

    # Change model architecture for reimplemented modules
    # ! Newly registered models have to be imported so that detectron can see them
    cfg.MODEL.META_ARCHITECTURE = RefactoredPanopticFPN.__name__
    cfg.MODEL.SEM_SEG_HEAD.NAME = RefactoredSemSegFPNHead.__name__
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
        "COCO-PanopticSegmentation/panoptic_fpn_R_50_3x.yaml"
    )  # Let training initialize from model zoo

    model = build_model(cfg)
    model.eval()
    if len(cfg.DATASETS.TEST):
        metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0])
        # TODO: Ignore label 255 as mentioned in dataset conifg of detectron

    checkpointer = DetectionCheckpointer(model)
    checkpointer.load(cfg.MODEL.WEIGHTS)
    aug = T.ResizeShortestEdge(
        [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST
    )

    # predictor = DefaultPredictor(cfg)

    # coco_sample = "data/test.jpg"
    nyu_sample = "data/NYUv2/test_images/1163.png"
    input_image = cv2.imread(nyu_sample)
    #
    # t = TT.Compose(
    #     [
    #         ImgToFloatTensor(),
    #         TT.Normalize(mean=NYUv2RawDataModule.MEAN, std=NYUv2RawDataModule.STD),
    #     ]
    # )

    argparser = configargparse.ArgParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    argparser.add(
        "-c",
        "--config-file",
        is_config_file=True,
        help="config file path",
        default="train-config.yml",
    )
    argparser = add_train_args(argparser)
    argparser = NYUv2RawDataModule.add_argparse_args(argparser)

    args = argparser.parse_args()

    dm = NYUv2RawDataModule(args, cfg)
    dm.setup()

    batch = next(iter(dm.train_dataloader()))
    outputs = model(batch)

    # outputs = predictor(input_image)

    for im, output in zip(batch["image"], outputs):
        im = im.to("cpu").numpy().transpose(1, 2, 0)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        v = Visualizer(im, MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
        out = v.draw_instance_predictions(output["instances"].to("cpu"))
        plt.imshow(out.get_image())
        plt.show()
    # tensor([103.5300, 116.2800, 123.6750], device='cuda:0')
    # tensor([1., 1., 1.], device='cuda:0')
    # Values to be used for image normalization (BGR order, since INPUT.FORMAT defaults to BGR).
    # To train on images of different number of channels, just set different mean & std.
    # Default values are the mean pixel value from ImageNet: [103.53, 116.28, 123.675]
    # _C.MODEL.PIXEL_MEAN = [103.530, 116.280, 123.675]
    # When using pre-trained models in Detectron1 or any MSRA models,
    # std has been absorbed into its conv1 weights, so the std needs to be set 1.
    # Otherwise, you can use [57.375, 57.120, 58.395] (ImageNet std)
    # _C.MODEL.PIXEL_STD = [1.0, 1.0, 1.0]


def main():
    # manual_run()
    detectron_predictor()


if __name__ == "__main__":
    main()
