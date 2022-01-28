import cv2
import json
import os
import random
import numpy as np
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.engine import DefaultTrainer
from detectron2.model_zoo import model_zoo
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_panoptic, register_coco_panoptic_separated, register_coco_instances
from detectron2.data.datasets.builtin_meta import _get_builtin_metadata
from detectron2.utils.visualizer import ColorMode


def _get_cfg():
    # https://colab.research.google.com/drive/16jcaJoc6bCFAQ96jDe2HwtXj7BMD_-m5?pli=1#scrollTo=HUjkwRsOn1O0
    # model_cfg_path = config file name relative to detectron2's "configs/"
    # model_cfg_path = "COCO-PanopticSegmentation/panoptic_fpn_R_50_1x.yaml"
    """
    register_coco_panoptic_separated(name="cocosuper_train",
                                     metadata=_get_builtin_metadata("coco_panoptic_separated"),
                                     image_root="./data/coco/images/train2017",
                                     panoptic_root="./data/coco/panoptic_train2017",
                                     panoptic_json="./data/coco/annotations/panoptic_train2017_superid.json",
                                     instances_json="./data/coco/annotations/instances_train2017_superid.json",
                                     sem_seg_root="./data/coco/panoptic_stuff_train2017")
    register_coco_panoptic_separated(name="cocosuper_val",
                                     metadata=_get_builtin_metadata("coco_panoptic_separated"),
                                     image_root="./data/coco/images/val2017",
                                     panoptic_root="./data/coco/panoptic_val2017",
                                     panoptic_json="./data/coco/annotations/panoptic_val2017_superid.json",
                                     instances_json="./data/coco/annotations/instances_val2017_superid.json",
                                     sem_seg_root="./data/coco/panoptic_stuff_val2017")
    print("Registered")
    """
    model_cfg_path = "COCO-PanopticSegmentation/panoptic_fpn_R_50_3x.yaml"

    cfg = get_cfg()  # Get default config
    cfg_file = model_zoo.get_config_file(model_cfg_path)
    cfg.merge_from_file(cfg_file)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model_cfg_path)
    """"
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-PanopticSegmentation/panoptic_fpn_R_50_3x.yaml"))
    cfg.DATASETS.TRAIN = ("cocosuper_train_separated",)
    cfg.DATASETS.TEST = ()
    cfg.DATALOADER.NUM_WORKERS = 4
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-PanopticSegmentation/panoptic_fpn_R_50_3x.yaml")  # Let training initialize from model zoo
    cfg.OUTPUT_DIR = "./output"
    cfg.SOLVER.IMS_PER_BATCH = 1
    cfg.SOLVER.BASE_LR = 0.01  # pick a good LR
    cfg.SOLVER.MAX_ITER = 30000  # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
    cfg.SOLVER.STEPS = [300, 600]  # do not decay learning rate
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 1024  # faster, and good enough for this toy dataset (default: 512)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 27  # only has one class (ballon). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
    # NOTE: this config means the number of classes, but a few popular unofficial tutorials incorrect uses num_classes+1 here.
    """
    return cfg

def main():
    cfg = _get_cfg()
    train = False
    """
    # https://github.com/facebookresearch/detectron2/blob/cbbc1ce26473cb2a5cc8f58e8ada9ae14cb41052/detectron2/engine/defaults.py#L251
    pred = DefaultPredictor(cfg)
    coco_sample = "data/test.jpg"
    nyu_sample = "data/NYUv2/test_images/1163.png"
    inputs = cv2.imread(coco_sample)
    outputs = pred(inputs)
    """
    if train:
        os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
        trainer = DefaultTrainer(cfg)
        trainer.resume_or_load(resume=False)
        trainer.train()
    else:
        predictor = DefaultPredictor(cfg)

        coco_sample = "data/test.jpg"
        nyu_sample = "data/NYUv2-raw/raw/sync/bathroom_0030/rgb_00000.jpg"
        inputs = cv2.imread(nyu_sample)
        outputs = predictor(inputs)

        v = Visualizer(inputs[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        cv2.imshow(mat=out.get_image(), winname="Output")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # Outputs Dict of tensors
    """
    'sem_seg',
    'instances,
    'panoptic_seg'
    """

    #print(outputs["panoptic_seg"][1])
    """
    with open(os.path.join("other", "coco_train_supercategories_map_by_id.json"), "r") as COCO:
        mapping = json.loads(COCO.read())
        
        for panoptic in outputs["panoptic_seg"][1]:
            entry = panoptic["category_id"]
            print(mapping[str(entry)])
            panoptic["category_id"] = mapping[str(entry)]["super_id"]
            
    print(outputs["panoptic_seg"][1])

    # TODO: See what the dataloaders for the COCO dataset return
    # See build_detection_train_loader(cfg) and build_detection_test_loader(cfg)
    # TODO: Visualize inference results of NYU and COCO
    v = Visualizer(inputs[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    cv2.imshow(mat=out.get_image(), winname="Output")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    """
if __name__ == "__main__":
    main()
