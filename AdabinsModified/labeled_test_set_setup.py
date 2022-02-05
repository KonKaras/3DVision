import os
from pathlib import Path

import cv2
import h5py
import numpy as np
import scipy.io
from torchvision.datasets.utils import download_url
from tqdm import tqdm

from constants import IMAGES, RAW_DEPTHS, SCENES


# Adapted from https://github.com/cleinc/bts/blob/master/utils/extract_official_train_test_set_from_mat.py


class NYUv2LabeledDepthDatasetSetup:
    URL = (
        "http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat"
    )

    DATA_TYPES = [IMAGES, RAW_DEPTHS]
    DEPTH_PRECISION = 1e3

    def __init__(
        self,
        root: str,
    ):
        self.root = root
        self.download()
        if not self._check_exists():
            raise RuntimeError(
                "Dataset not complete." + " You can use download=True to download it"
            )

    def _check_exists(self) -> bool:
        """
        Only checking for folder existence
        """
        for split in ["train", "test"]:
            split_path = os.path.join(self.root, split)
            if not os.path.exists(split_path) or not [
                subfolder
                for subfolder in Path(split_path).iterdir()
                if subfolder.is_dir()
            ]:
                return False
        return True

    def download(self):
        if self._check_exists():
            print("Data exists.")
            return
        print(f"Setting up dataset structure.")
        self.setup_dataset_structure()
        print("Done!")

    def setup_dataset_structure(self):
        # Download from url and create png files
        train_dst = os.path.join(self.root, "train")
        test_dst = os.path.join(self.root, "test")

        if not os.path.exists(train_dst) or not os.path.exists(test_dst):
            os.mkdir(train_dst)
            os.mkdir(test_dst)
        mat_file = os.path.join(self.root, self.URL.split("/")[-1])
        if not os.path.exists(mat_file):
            download_url(self.URL, self.root)
        splits_mat_file = str(Path(self.root).parent / "splits.mat")
        self._create_png_files(
            mat_file,
            splits_mat_file,
            train_dst,
            test_dst,
            self.DEPTH_PRECISION,
        )

    def _create_png_files(
        self,
        mat_file: str,
        splits_mat_file: str,
        train_dir,
        test_dir,
        depth_precision: float = 1e3,
    ):
        """
        Extract the arrays from the mat file into images according to key
        :param mat_file: path to the official labelled dataset .mat file
        """
        h5_file = h5py.File(mat_file, "r")
        train_test_split = scipy.io.loadmat(splits_mat_file)

        # Get train and test indices
        test_idxs = set([int(x) for x in train_test_split["testNdxs"]])
        train_idxs = set([int(x) for x in train_test_split["trainNdxs"]])
        print(f"{len(train_idxs):d} training images")
        print(f"{len(test_idxs):d} test images")

        raw_depths = h5_file[RAW_DEPTHS]
        images = h5_file[IMAGES]
        """
        scenes = [
            "".join(chr(c) for c in h5_file[obj_ref])
            for obj_ref in h5_file["sceneTypes"][0]
        ]
        """
        scenes = []

        refs = h5_file[SCENES][()][0]

        for i in refs:
            name_ref = h5py.h5r.get_name(i, h5_file.id)
            name_as_array = np.array(h5_file[name_ref][()], dtype='int')
            scene = ""
            for subarr in name_as_array:
                for ascii_val in subarr:
                    scene = scene + chr(ascii_val)

            #scene = ''.join(map(chr, name_as_array))
            scenes.append(scene)


        print("processing images")
        for i, image in tqdm(enumerate(images), total=len(images), unit="img"):
            scene = scenes[i]
            depth_img = (raw_depths[i, :, :].T * depth_precision).astype(np.uint16)
            image = image.astype(np.uint8).T

            idx = i + 1
            scene_folder = ""
            if idx in train_idxs:
                split_folder = train_dir
                scene_folder = os.path.join(split_folder, scene)

            else:
                assert (
                    idx in test_idxs
                ), f"index {idx:d} neither found in training set nor in test set"
                split_folder = test_dir
                scene_folder = os.path.join(split_folder, scene.rsplit('_', 1)[0])

            #scene_folder = os.path.join(split_folder, scene.split('_')[0])
            Path(scene_folder).mkdir(exist_ok=True)

            # Loaded image is RGB, cv2 needs BGR
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            # Converts white boundary to black
            image_black_boundary = np.zeros((480, 640, 3), dtype=np.uint8)
            image_black_boundary[7:474, 7:632, :] = image[7:474, 7:632, :]

            depth_filename = f"sync_depth_{i:05d}.png"
            image_filename = f"rgb_{i:05d}.jpg"
            cv2.imwrite(os.path.join(scene_folder, depth_filename), depth_img)
            cv2.imwrite(
                os.path.join(scene_folder, image_filename), image_black_boundary
            )

        print("Finished")


def main():
    ds = NYUv2LabeledDepthDatasetSetup("data/NYUv2-raw/official_splits")


if __name__ == "__main__":
    main()
