import os
from pathlib import Path
from typing import Tuple

import numpy as np
import detectron2.structures

from constants import NYUv2_ROOT
from dataset.coco import pycococreator
from PIL import Image
import cv2
import json


def rgb2id(color):
    # https://github.com/cocodataset/panopticapi/blob/master/panopticapi/utils.py
    if isinstance(color, np.ndarray) and len(color.shape) == 3:
        if color.dtype == np.uint8:
            color = color.astype(np.int32)
        return color[:, :, 0] + 256 * color[:, :, 1] + 256 * 256 * color[:, :, 2]
    return int(color[0] + 256 * color[1] + 256 * 256 * color[2])


def id2rgb(id_map):
    # https://github.com/cocodataset/panopticapi/blob/master/panopticapi/utils.py
    if isinstance(id_map, np.ndarray):
        id_map_copy = id_map.copy()
        rgb_shape = tuple(list(id_map.shape) + [3])
        rgb_map = np.zeros(rgb_shape, dtype=np.uint8)
        for i in range(3):
            rgb_map[..., i] = id_map_copy % 256
            id_map_copy //= 256
        return rgb_map
    color = []
    for _ in range(3):
        color.append(id_map % 256)
        id_map //= 256
    return color


"""

function [instanceMasks, instanceLabels] = get_instance_masks(...
    imgObjectLabels, imgInstances)
  
  [H, W] = size(imgObjectLabels);  

  pairs = unique([imgObjectLabels(:), uint16(imgInstances(:))], 'rows');
  pairs(sum(pairs, 2) == 0, :) = [];
  
  N = size(pairs, 1);
  
  instanceMasks = false(H, W, N);
  instanceLabels = zeros(N, 1);
  for ii = 1 : N
    instanceMasks(:,:,ii) = imgObjectLabels == pairs(ii,1) & imgInstances == pairs(ii,2);
    instanceLabels(ii) = pairs(ii,1);
  end
end
"""


def get_instance_masks(label_map, instance_map) -> Tuple[np.ndarray, np.ndarray]:
    """
    :param label_map: HxW label map. 0 indicates a missing label.
    :param instance_map: HxW instance map. 0 indicates a missing label.
    :return: instance_masks: binary masks of size HxWxN where N is the number of
                 total objects in the room.

            instance_labels: Nx1 vector of class (RGB)-labels for each instance mask.
    """
    h, w = label_map.shape
    stacked = np.stack([label_map, instance_map], axis=2)
    #print(f"Stacked shape {stacked.shape}")
    # Find unique combinations of labels and instances
    unique_values = count_colors(stacked)

    # TODO: Drop all zeros

    num_instances = len(unique_values)

    instance_masks = np.zeros((h, w, num_instances), dtype=bool)
    instance_labels = np.zeros((num_instances, 1), dtype="uint8")

    for i in range(num_instances):
        # Iterate over the instances
        label_value = unique_values[i, 0]
        instance_value = unique_values[i, 1]
        # Check for each pixel if the pixel has the label and instance values for this instance
        instance_masks[:, :, i] = (label_map == label_value) & (
            instance_map == instance_value
        )
        instance_labels[i] = label_value

    #print(unique_values.shape)
    return instance_masks, instance_labels


def count_colors(img: np.ndarray):
    """
    Count unique colors in an image.
    :param img:
    :return:
    """
    # https://stackoverflow.com/a/56606457
    if len(img.shape) == 2:
        # 2D array
        return np.unique(img, return_counts=False)
    # H x W x C -> need some magic
    return np.unique(img.reshape(-1, img.shape[-1]), axis=0, return_counts=False)


def main():
    train_instances_path = Path(NYUv2_ROOT, "train_instances")
    train_instances_ls = [f for f in train_instances_path.iterdir()]
    train_instances_ls.sort(key=lambda p: int(p.stem))

    image_id = 0
    segmentation_id = 0

    all_image_dicts = []

    print("Preprocessing instances...")

    for instance_path in train_instances_ls:

        image_dict = {}
        dir_name = str(image_id + 1).zfill(4)
        os.mkdir(f"temp/{dir_name}")
        label_path = Path(NYUv2_ROOT, "train_labels", instance_path.name)
        #instance_map = np.array(Image.open(str(instance_path)))
        #label_map = np.array(Image.open(str(label_path)))

        instance_map = cv2.imread(str(instance_path), cv2.IMREAD_UNCHANGED)

        h, w = instance_map.shape

        image_dict["file_name"] = os.path.join(NYUv2_ROOT, "train_images", instance_path.name)
        image_dict["height"] = h
        image_dict["width"] = w
        image_dict["image_id"] = instance_path.name
        image_dict["sem_seg_file_name"] = str(label_path)

        label_map = cv2.imread(str(label_path), cv2.IMREAD_UNCHANGED)

        #print(f"Instance shape {instance_map.shape}")
        #print(f"Label shape {label_map.shape}")

        unique_instances = count_colors(instance_map)
        unique_labels = count_colors(label_map)

        instance_masks, instance_labels = get_instance_masks(label_map, instance_map)
        image = cv2.imread(str(Path(NYUv2_ROOT, "train_images", instance_path.name)), cv2.IMREAD_UNCHANGED)
        num_instances = len(instance_labels)
        annotations = []

        for instance_idx in range(num_instances):
            # Iterate over the instance masks
            class_id = instance_labels[instance_idx]
            instance_mask = instance_masks[:, :, instance_idx]
            binary_mask = np.asarray(Image.open(instance_path).convert('1')).astype(np.uint8)
            out_path = f"temp/{dir_name}/instance_{instance_idx}_class_{list(class_id)}.png"
            cv2.imwrite(out_path, instance_mask * 255)
            category_info = {'id': int(class_id), 'is_crowd': 0}

            anno = pycococreator.create_annotation_info(
                segmentation_id,
                image_id,
                category_info,
                instance_mask,
                image.size,
                tolerance=2
            )
            """
            anno = {}
            anno["bbox"] = extract_bboxes(instance_mask).tolist()
            anno["bbox_mode"] = detectron2.structures.BoxMode.XYXY_ABS
            anno["category_id"] = int(class_id)
            """

            #TODO segmentation dict

            annotations.append(anno)
            # cv2.imshow(f"Instance {idx} Class {list(class_id)}", instance_map)
            # cv2.waitKey()
            # cv2.destroyAllWindows()
            segmentation_id += 1

        image_dict["annotations"] = annotations
        image_id += 1
        all_image_dicts.append(image_dict)

    print("Storing nyuv2 dict as json...")
    with open('../data/NYUv2/annotations/nyuv2_dict.json', 'w') as d:
        json.dump(all_image_dicts, d)

def preprocess_for_detectron(instance_masks, instance_labels):
    """
    path = os.path.join(self.root, "train_instances")
    annotations = {}
    with os.scandir(path) as files:
        for image in files:
             # annotations["bbox"] = extract_bboxes(masks)
            annotations["bbox_mode"] = detectron2.structures.BoxMode.XYXY_ABS
            annotations["segments_info"] = []
    """
    return 0

# https://github.com/matterport/Mask_RCNN/blob/master/mrcnn/utils.py
def extract_bboxes(mask):

    """Compute bounding boxes from masks.
    mask: [height, width, num_instances]. Mask pixels are either 1 or 0.
    Returns: bbox array [num_instances, (y1, x1, y2, x2)].
    """
    #boxes = np.zeros([mask.shape[-1], 4], dtype=np.int32)
    #for i in range(mask.shape[-1]):

    # Bounding box.
    horizontal_indicies = np.where(np.any(mask, axis=0))[0]
    vertical_indicies = np.where(np.any(mask, axis=1))[0]
    if horizontal_indicies.shape[0]:
        x1, x2 = horizontal_indicies[[0, -1]]
        y1, y2 = vertical_indicies[[0, -1]]
        # x2 and y2 should not be part of the box. Increment by 1.
        x2 += 1
        y2 += 1
    else:
        # No mask for this instance. Might happen due to
        # resizing or cropping. Set bbox to zeros
        x1, x2, y1, y2 = 0, 0, 0, 0
    box = np.array([y1, x1, y2, x2])
    return box.astype(np.int32)

if __name__ == "__main__":
    main()
