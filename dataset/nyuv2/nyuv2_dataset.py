"""
based on https://github.com/xapharius/pytorch-nyuv2
author: Mihai Suteu
date: 15/05/19
"""

import os
import random
import sys
import scipy.io

import cv2
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.datasets.utils import download_url
from tqdm import tqdm

from constants import DEPTHS, IMAGES, INSTANCES, LABELS, NYU_40, NYU_FULL


class NYUv2Dataset(Dataset):
    """
    PyTorch wrapper for the NYUv2 dataset focused on multi-task learning.
    Data sources available: RGB, Semantic Segmentation, Surface Normals, Depth Images.
    If no transformation is provided, the image type will not be returned.
    ### Output
    All images are of size: 640 x 480
    1. RGB: 3 channel input image
    2. Semantic Segmentation: 1 channel representing one of the 14 (0 -
    background) classes. Conversion to int will happen automatically if
    transformation ends in a tensor.
    3. Surface Normals: 3 channels, with values in [0, 1].
    4. Depth Images: 1 channel with floats representing the distance in meters.
    Conversion will happen automatically if transformation ends in a tensor.
    """

    # ssl._create_default_https_context = ssl._create_unverified_context
    URL = (
        "http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat"
    )
    # Specify upto which decimal place the depth values should be stored
    # Incoming data is in meters -> DEPTH_PRECISION = 1e3 would store till millimeter
    DEPTH_PRECISION = 1e3

    LABELS_40_PATH = "data/NYUv2/labels40.mat"

    def __init__(
        self,
        root: str,
        train: bool = True,
        download: bool = False,
        rgb_transform=None,
        seg_transform=None,
        sn_transform=None,
        depth_transform=None,
        train_percentage=0.8,
        label_type=NYU_FULL,
        instance_transform=None,
    ):
        """
        Will return tuples based on what data source has been enabled (rgb, seg etc).
        :param root: path to root folder (eg /data/NYUv2)
        :param train: whether to load the train or test set
        :param download: whether to download and process data if missing
        :param rgb_transform: the transformation pipeline for rbg images
        :param seg_transform: the transformation pipeline for segmentation images. If
        the transformation ends in a tensor, the result will be automatically
        converted to int in [0, 14)
        :param sn_transform: the transformation pipeline for surface normal images
        :param depth_transform: the transformation pipeline for depth images. If the
        transformation ends in a tensor, the result will be automatically converted
        to meters
        :param label_type: Which class labels to use.
        """
        super().__init__()

        assert label_type in [
            NYU_FULL,  # ~900 classes
            NYU_40,
        ]
        if label_type == NYU_FULL:
            self.labels_key = LABELS  # Use the `labels` folder
        else:
            self.labels_key = label_type

        self.root = root

        self.rgb_transform = rgb_transform
        self.seg_transform = seg_transform
        self.sn_transform = sn_transform
        self.depth_transform = depth_transform
        self.instance_transform = depth_transform

        self.train_percentage = train_percentage

        self.train = train
        self._split = "train" if train else "test"

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError(
                "Dataset not complete." + " You can use download=True to download it"
            )

        # rgb folder as ground truth
        self._files = os.listdir(os.path.join(root, f"{self._split}_images"))

    def set_transform(self, rgb, depth, seg, inst):
        self.rgb_transform = rgb
        self.depth_transform = depth
        self.seg_transform = seg
        self.instance_transform = inst

    def __getitem__(self, index: int):
        folder = lambda name: os.path.join(self.root, f"{self._split}_{name}")
        seed = random.randrange(sys.maxsize)
        imgs = []
        filename = self._files[index]

        if self.rgb_transform is not None:
            random.seed(seed)  # ? Why seeds are here, they should be global
            img = cv2.imread(os.path.join(folder(IMAGES), filename))
            img = self.rgb_transform(img)
            imgs.append(img)

        if self.seg_transform is not None:
            img = cv2.imread(
                os.path.join(folder(self.labels_key), filename), cv2.IMREAD_UNCHANGED
            )
            img = self.seg_transform(img)
            if isinstance(img, torch.Tensor):
                # ToTensor scales to [0, 1] by default
                img = (img * 255).long()
            imgs.append(img)
        """
        if self.sn_transform is not None:
            random.seed(seed)
            img = Image.open(os.path.join(folder("sn"), self._files[index]))
            img = self.sn_transform(img)
            imgs.append(img)
        """

        if self.depth_transform is not None:
            img = cv2.imread(
                os.path.join(folder(DEPTHS), filename),
                cv2.IMREAD_ANYDEPTH,
            )
            img = self.depth_transform(img)
            if isinstance(img, torch.Tensor):
                # depth png is uint16
                img = img.float() / self.DEPTH_PRECISION
            imgs.append(img)

        if self.instance_transform is not None:
            img = cv2.imread(
                os.path.join(folder(INSTANCES), filename), cv2.IMREAD_UNCHANGED
            )
            img = self.instance_transform(img)
            imgs.append(img)

        return imgs

    def __len__(self):
        return len(self._files)

    def __repr__(self):
        fmt_str = f"Dataset {self.__class__.__name__}\n"
        fmt_str += f"    Number of data points: {self.__len__()}\n"
        fmt_str += f"    Split: {self._split}\n"
        fmt_str += f"    Root Location: {self.root}\n"
        tmp = "    RGB Transforms: "
        fmt_str += "{0}{1}\n".format(
            tmp, self.rgb_transform.__repr__().replace("\n", "\n" + " " * len(tmp))
        )
        tmp = "    Seg Transforms: "
        fmt_str += "{0}{1}\n".format(
            tmp, self.seg_transform.__repr__().replace("\n", "\n" + " " * len(tmp))
        )
        """
        tmp = "    SN Transforms: "
        fmt_str += "{0}{1}\n".format(
            tmp, self.sn_transform.__repr__().replace("\n", "\n" + " " * len(tmp))
        )
        """
        tmp = "    Depth Transforms: "
        fmt_str += "{0}{1}\n".format(
            tmp, self.depth_transform.__repr__().replace("\n", "\n" + " " * len(tmp))
        )

        tmp = "    Instance Transforms: "
        fmt_str += "{0}{1}\n".format(
            tmp, self.instance_transform.__repr__().replace("\n", "\n" + " " * len(tmp))
        )
        return fmt_str

    def _check_exists(self) -> bool:
        """
        Only checking for folder existence
        """
        for split in ["train", "test"]:
            for type_ in ["images", self.labels_key, "depths", "instances"]:
                path = os.path.join(self.root, f"{split}_{type_}")
                if not os.path.exists(path):
                    return False
        return True

    def download(self):
        if self._check_exists():
            print("Data exists.")
            return
        self.download_labeled_nyuv2()
        print("Done!")

    def download_labeled_nyuv2(self):
        for key in [
            self.labels_key,
            DEPTHS,
            IMAGES,
            INSTANCES,
        ]:
            print(f"Setting up {key} dataset structure.")
            self.setup_dataset_structure(key=key)

    def setup_dataset_structure(self, key: str):
        # Download from url and create png files
        train_dst = os.path.join(self.root, "train_" + key)
        test_dst = os.path.join(self.root, "test_" + key)

        num_images = 1449

        if not os.path.exists(train_dst) or not os.path.exists(test_dst):
            os.mkdir(train_dst)
            os.mkdir(test_dst)
        if key == NYU_40:
            # No need to download, just use the labels40.mat file
            mat_file = self.LABELS_40_PATH
        else:
            mat_file = os.path.join(self.root, self.URL.split("/")[-1])
            if not os.path.exists(mat_file):
                download_url(self.URL, self.root)
        self._create_png_files(
            mat_file,
            self.root,
            key,
            int(num_images * self.train_percentage),
            self.DEPTH_PRECISION,
        )

    def _create_png_files(
        self,
        mat_file: str,
        root: str,
        key: str,
        num_train: int,
        depth_precision: float = 1e3,
    ):
        """
        Extract the arrays from the mat file into images according to key
        :param mat_file: path to the official labelled dataset .mat file
        :param root: The root directory of the dataset
        :param key: what kind of data to extract, must correspond to matlab naming
        :param num_train: the max number of train images (for splitting)
        """
        print("Extracting " + key)
        if key == NYU_40:
            print(f"Extracting {NYU_40}")
            files = scipy.io.loadmat(mat_file, squeeze_me=False)["labels40"]
            for i in tqdm(range(files.shape[-1])):
                file = files[:, :, i]
                id_ = str(i + 1).zfill(4)
                folder = "train" if i <= num_train else "test"
                save_path = os.path.join(root, folder + "_" + key, id_ + ".png")
                img = file.astype(np.uint8)
                cv2.imwrite(save_path, img)
        else:
            files = h5py.File(mat_file, "r")[key]
            for i, file in tqdm(enumerate(files)):
                id_ = str(i + 1).zfill(4)
                folder = "train" if i <= num_train else "test"
                save_path = os.path.join(root, folder + "_" + key, id_ + ".png")
                if key == DEPTHS:
                    # ! Don't save as uint16 if depth_precision >= 1e4
                    # plt.imshow(file.T)
                    # plt.title(f"raw {id_}")
                    # plt.show()
                    img = (file * depth_precision).astype(np.uint16).T
                    # plt.imshow(img)
                    # plt.title(f"uint16 {id_}")
                    # plt.show()
                else:
                    img = file.astype(np.uint8).T
                cv2.imwrite(save_path, img)
