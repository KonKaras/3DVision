from argparse import ArgumentParser
from typing import Dict, Optional

import numpy as np
from detectron2.config import CfgNode
from imgaug import HeatmapsOnImage
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from tqdm import tqdm

from dataset.nyuv2_raw.bts_dataloader import (
    DataLoadPreprocess,
    ToTensorHelper,
)
import imgaug.augmenters as iaa
import torchvision.transforms as T

from dataset.utils import ImgToFloatTensor


class NYUv2RawDataModule(LightningDataModule):
    @classmethod
    def add_argparse_args(
        cls, parent_parser: ArgumentParser, **kwargs
    ) -> ArgumentParser:
        parser = parent_parser.add_argument_group(cls.__name__)
        # Dataset
        parser.add_argument(
            "--dataset",
            type=str,
            help="dataset to train on, kitti or nyu",
            default="nyu",
        )
        parser.add_argument(
            "--data_path",
            type=str,
            help="path to the data",
            required=True,
        )
        parser.add_argument(
            "--gt_path", type=str, help="path to the groundtruth data", required=True
        )
        parser.add_argument(
            "--filenames_file",
            type=str,
            help="path to the filenames text file",
            required=True,
        )
        parser.add_argument(
            "--input_height", type=int, help="input height", default=480
        )
        parser.add_argument("--input_width", type=int, help="input width", default=640)
        parser.add_argument(
            "--max_depth", type=float, help="maximum depth in estimation", default=10
        )

        # Preprocessing
        parser.add_argument(
            "--do_random_rotate",
            help="if set, will perform random rotation for augmentation",
            action="store_true",
        )
        parser.add_argument(
            "--degree", type=float, help="random rotation maximum degree", default=2.5
        )
        parser.add_argument(
            "--do_kb_crop",
            help="if set, crop input images as kitti benchmark images",
            action="store_true",
        )
        parser.add_argument(
            "--use_right",
            help="if set, will randomly use right images when train on KITTI",
            action="store_true",
        )

        # Online eval = Validation
        parser.add_argument(
            "--data_path_eval",
            type=str,
            help="path to the data for online evaluation",
            required=False,
        )
        parser.add_argument(
            "--gt_path_eval",
            type=str,
            help="path to the groundtruth data for online evaluation",
            required=False,
        )
        parser.add_argument(
            "--filenames_file_eval",
            type=str,
            help="path to the filenames text file for online evaluation",
            required=False,
        )
        parser.add_argument(
            "--min_depth_eval",
            type=float,
            help="minimum depth for evaluation",
            default=1e-3,
        )
        parser.add_argument(
            "--max_depth_eval",
            type=float,
            help="maximum depth for evaluation",
            default=80,
        )
        parser.add_argument(
            "--eigen_crop",
            help="if set, crops according to Eigen NIPS14",
            action="store_true",
        )
        parser.add_argument(
            "--garg_crop",
            help="if set, crops according to Garg  ECCV16",
            action="store_true",
        )

        parser.add_argument(
            "--num_workers",
            type=int,
            help="number of threads to use for data loading",
            default=1,
        )

        parser.add_argument(
            "--standardize_image_input",
            help="if set, divide input images by 255 when loading data. NOTE: Detectron does not do this.",
            action="store_true",
        )

        return parent_parser

    def __init__(self, args, detectron_cfg: CfgNode, **kwargs):
        super().__init__()

        self.args = args

        # Arguments from detectron that might be useful
        self.detectron_cfg = detectron_cfg
        # Mean and Std defined for FPN model in Detectron2 -> use in Normalization
        self.PIXEL_MEAN = detectron_cfg.MODEL.PIXEL_MEAN
        self.PIXEL_STD = detectron_cfg.MODEL.PIXEL_STD
        # self.MIN_SIZE_TEST = detectron_cfg.INPUT.MIN_SIZE_TEST  # TODO: Handle these
        # self.MAX_SIZE_TEST = detectron_cfg.INPUT.MAX_SIZE_TEST
        # self.MIN_SIZE_TRAIN = detectron_cfg.INPUT.MIN_SIZE_TRAIN  # TODO: Handle these
        # self.MAX_SIZE_TRAIN = detectron_cfg.INPUT.MAX_SIZE_TRAIN
        self.INPUT_FORMAT = detectron_cfg.INPUT.FORMAT
        self.MASK_FORMAT = (
            detectron_cfg.INPUT.MASK_FORMAT
        )  # TODO: What to do with this, used here:
        # https://github.com/facebookresearch/detectron2/blob/cbbc1ce26473cb2a5cc8f58e8ada9ae14cb41052/detectron2/data/detection_utils.py#L369

    def prepare_data(self) -> None:
        # NYUv2LabeledDepthDatasetSetup("data/NYUv2-raw/official_splits")
        pass

    def setup(self, stage: Optional[str] = None) -> None:
        self._setup_datasets("train")
        self._setup_datasets("val")  # = test set
        print(f"Train size: {len(self.train_dataset)} Val size {len(self.val_dataset)}")
        print(f"{self.__class__.__name__} setup done.")

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            self.args.batch_size,
            shuffle=(self.train_sampler is None),
            num_workers=self.args.num_workers,
            pin_memory=True,
            sampler=self.train_sampler,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            self.args.batch_size,
            shuffle=False,
            num_workers=1,
            pin_memory=True,
            sampler=self.val_sampler,
        )

    def test_dataloader(self):
        return DataLoader(self.test_dataset, 1, shuffle=False, num_workers=1)

    def predict_dataloader(self):
        raise NotImplementedError

    @property
    def train_transforms(self):

        # TODO: Add additional transforms here
        # Make sure to apply the same transformation to RGB and depth images
        # and other sorts of input
        # ToTensorHelper taken from preprocessing_transforms in bts_dataloader.py

        # pixel_mean
        # tensor([[[103.5300]],
        #
        #         [[116.2800]],
        #
        #         [[123.6750]]], device='cuda:0')

        # pixel_std
        # tensor([[[1.]],
        #
        #         [[1.]],
        #
        #         [[1.]]], device='cuda:0')

        im_transform = T.Compose(
            [
                iaa.Sequential(
                    [
                        iaa.GammaContrast((0.9, 1.1)),
                        iaa.MultiplyBrightness((0.9, 1.1)),
                        iaa.Multiply((0.9, 1.1), per_channel=True),
                    ],
                    random_order=True,
                ).augment_image,
                ImgToFloatTensor(),
                T.Normalize(mean=self.PIXEL_MEAN, std=self.PIXEL_STD),
            ]
        )

        depth_transform = T.Compose(
            [
                ImgToFloatTensor(),
            ]
        )

        # TODO: RESIZE TO MINSIZE
        # 'MIN_SIZE_TRAIN' (640, 672, 704, 736, 768, 800)
        common_transforms = iaa.Sequential(
            [
                # iaa.Resize({"shorter-side": 640, "longer-side": "keep-aspect-ratio"}),
                iaa.Resize(
                    {"height": self.args.input_height, "width": self.args.input_width}
                ),
                # TODO: PanopticFPN expects larger images than what's needed for AdaBins
                # TODO: We can do inference in larger size and then do the downsizing if needed
                iaa.Crop(percent=(0, 0.1), keep_size=True, sample_independently=True),
                # iaa.CoarseDropout((0.0, 0.05), size_percent=(0.02, 0.25)),
                # iaa.HorizontalFlip(0.5),
                # iaa.SaltAndPepper((0, 0.05), per_channel=True),
                # iaa.Affine(
                #     rotate=(-self.args.degree, self.args.degree)
                # ),  # rotate by -45 to 45 degrees (affects heatmaps)
                # iaa.ElasticTransformation(
                #     alpha=50, sigma=5
                # ),  # apply water effect (affects heatmaps)
                # iaa.SaveDebugImageEveryNBatches(folder_path, 100)
            ],
            random_order=True,
        )

        def transform(sample: Dict[str, np.ndarray]):
            # Sample returned by DataLoadPreprocess

            # TODO: Augmentation see train_preprocess in DataLoadPreprocess
            # NOTE: Augmentation can also be done in DataLoadPreprocess
            # TODO: Depth transformation needed? i.e. normalization per scene?
            image = sample["image"]
            depth = sample["depth"]
            # sample["raw_image"] = ImgToFloatTensor()(image)

            depth = HeatmapsOnImage(
                depth,
                shape=image.shape,
                min_value=depth.min(),  # Taken from AdaBins
                max_value=depth.max(),  # Taken from AdaBins (TODO: check if correct)
            )
            # TODO: Should we normalize depth so that it is between 0-1
            image, depth = common_transforms(image=image, heatmaps=depth)
            # TODO: Learn and set what the training size should be
            depth = depth.get_arr()
            sample["image"] = im_transform(image)
            sample["depth"] = depth_transform(depth)
            return sample

        return transform
        # return T.Compose([ToTensorHelper(mode="train", mean=self.MEAN, std=self.STD)])

    # def _build_transform(self, mode):
    #     # Imitate ToTensorHelper from bts_dataloader
    #     def _transform(sample):
    #         image = sample["image"]
    #
    #     return transform

    @property
    def val_transforms(self):
        return transforms.Compose([ToTensorHelper(mode="val")])

    @property
    def test_transforms(self):
        # TODO: MIN_TEST SIZE for detectron
        return transforms.Compose([ToTensorHelper(mode="test")])

    def _setup_datasets(self, mode):
        # Setup dataset, sampler and dataloader
        if mode == "train":
            self.train_dataset = DataLoadPreprocess(
                self.args, mode, transform=self.train_transforms
            )
            self.train_sampler = None

        elif mode == "val":  # ! Switched "online_eval" to "val"
            self.val_dataset = DataLoadPreprocess(
                self.args, mode, transform=self.val_transforms
            )
            self.val_sampler = None

        elif mode == "test":
            self.test_dataset = DataLoadPreprocess(
                self.args, mode, transform=self.test_transforms
            )
        else:
            raise ValueError(f"mode should be one of 'train, test, val'. Got {mode}")

    def check_dataloader(self, mode):
        """
        Iterate over the dataloaders to check if dataset is setup correctly.
        :param mode:
        :return:
        """
        if mode == "train":
            dataloader = self.train_dataloader()
        elif mode == "val":
            dataloader = self.val_dataloader()
        elif mode == "test":
            dataloader = self.test_dataloader()
        else:
            raise ValueError(f"mode should be one of 'train, test, val'. Got {mode}")

        for batch in tqdm(dataloader, unit="batch", desc=f"{mode} DataLoader check."):
            continue
            # TODO: Visualize some batches for sanity check
