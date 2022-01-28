import os

from constants import DEPTHS, NYU_40
from dataset.nyuv2.nyuv2_dataset import NYUv2Dataset

DATASET_ROOT_DIR = "data"


def main():

    dataset = NYUv2Dataset(
        root=os.path.join(DATASET_ROOT_DIR, "NYUv2"),
        download=True,
        train=False,
    )
    # dataset.download_labeled_nyuv2()
    dataset.setup_dataset_structure(NYU_40)


if __name__ == "__main__":
    main()
