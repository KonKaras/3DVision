import cv2
import os

import torch
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer
from detectron2.model_zoo import model_zoo
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
from deprecated.BatchPredictor import BatchPredictor
def _get_cfg():

    model_cfg_path = "COCO-PanopticSegmentation/panoptic_fpn_R_50_3x.yaml"

    cfg = get_cfg()  # Get default config
    #print(cfg.INPUT.FORMAT) # -> BGR
    cfg_file = model_zoo.get_config_file(model_cfg_path)
    cfg.merge_from_file(cfg_file)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model_cfg_path)
    #cfg.INPUT.FORMAT = "RGB"

    cfg.MIN_SIZE_TRAIN = (480,)
    cfg.MIN_SIZE_TEST = (480,)

    return cfg

def main():
    detectron2_demo()

def get_panoptic_custom_model():
    cfg = _get_cfg()
    model = build_model(cfg) #torch module!
    model_cfg_path = "COCO-PanopticSegmentation/panoptic_fpn_R_50_3x.yaml"
    DetectionCheckpointer(model).load(model_zoo.get_checkpoint_url(model_cfg_path))
    model.train(False)
    return model

def get_panoptic_predictor():
    cfg = _get_cfg()
    predictor = BatchPredictor(cfg)
    predictor.requires_grad = False
    return predictor

def visualize(input_img, map, seg_info):
    cfg = _get_cfg()
    input = input_img
    if type(input_img) == torch.Tensor:
        input = input_img.cpu().numpy().transpose(1,2,0)
    else:
        input = input_img[:, :, ::-1]

    v = Visualizer(input, MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
    out = v.draw_panoptic_seg_predictions(map.to("cpu"), seg_info)

    cv2.imshow(mat=out.get_image(), winname="Output")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def detectron2_demo():
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
        predictor = get_panoptic_custom_model()#get_panoptic_predictor()#DefaultPredictor(cfg)

        coco_sample = "data/test.jpg"
        nyu_sample = "data/NYUv2-raw/raw/sync/kitchen_0003/rgb_00064.jpg"
        nyu_sample2 = "data/NYUv2-raw/raw/sync/living_room_0033/rgb_00022.jpg"
        inputs = cv2.imread(nyu_sample)
        inputs2 = cv2.imread(nyu_sample2)
        list = []
        dict = {'image': torch.from_numpy(inputs.transpose(2, 0, 1))} #convert to 3xhxw
        dict2 = {'image': torch.from_numpy(inputs2.transpose(2, 0, 1))} #convert to 3xhxw

        list.append(dict)
        list.append(dict2)

        print(list)

        all_images_outputs = predictor(list)#["panoptic_seg"]

        for output in all_images_outputs:
            #print(output)
            mask, info = output["panoptic_seg"]
            print(info)
            visualize(inputs, mask, info)

if __name__ == "__main__":
    main()
