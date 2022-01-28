from pycocotools.coco import COCO


PANOPTIC_VAL_ANNOTATION_PATH = "data/coco/annotations/panoptic_val2017.json"
PANOPTIC_TRAIN_ANNOTATION_PATH = "data/coco/annotations/panoptic_train2017.json"


def prepare_panoptic_fpn():


def download():
    annotation_file = PANOPTIC_VAL_ANNOTATION_PATH
    coco = COCO(annotation_file)
    coco.info()


def main():
    download()


if __name__ == "__main__":
    main()
