from typing import Dict, List

import torch
from detectron2 import model_zoo
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.modeling import (
    META_ARCH_REGISTRY,
    PanopticFPN,
    build_model,
    detector_postprocess,
)
from detectron2.modeling.meta_arch.panoptic_fpn import (
    combine_semantic_and_instance_outputs,
)
from detectron2.modeling.postprocessing import sem_seg_postprocess
from detectron2.structures import ImageList

# Register architecture to Detectron
from dataset.nyuv2_raw.nyuv2_raw_datamodule import NYUv2RawDataModule
from dataset.utils import batch_of_tensors_to_image_list
from .sem_seg_fpn_head import RefactoredSemSegFPNHead
import torchvision.transforms as T


@META_ARCH_REGISTRY.register()
class RefactoredPanopticFPN(PanopticFPN):
    @classmethod
    def build(
        cls,
        model_cfg_path="COCO-PanopticSegmentation/panoptic_fpn_R_50_3x.yaml",
        roi_thresh_test=0.5,
    ):
        """
        Create model and load checkpoint.
        :return:
        """
        # model_cfg_path = config file name relative to detectron2's "configs/"
        # model_cfg_path = "COCO-PanopticSegmentation/panoptic_fpn_R_50_1x.yaml"
        # model_cfg_path = "COCO-PanopticSegmentation/panoptic_fpn_R_50_3x.yaml"

        cfg = get_cfg()  # Get default config
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
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = roi_thresh_test

        ckpt_url = model_zoo.get_checkpoint_url(model_cfg_path)
        cfg.MODEL.WEIGHTS = ckpt_url  # Let training initialize from model zoo

        model: PanopticFPN = build_model(cfg)  # returns a torch.nn.Module
        DetectionCheckpointer(model).load(
            ckpt_url
        )  # load a file, usually from cfg.MODEL.WEIGHTS

        return model, cfg

    def forward(self, batched_inputs):
        """
        Args:
            # TODO: Input type has changed
            batched_inputs: a list, batched outputs of :class:`DatasetMapper`.
                Each item in the list contains the inputs for one image.

                For now, each item in the list is a dict that contains:

                * "image": Tensor, image in (C, H, W) format.
                * "instances": Instances
                * "sem_seg": semantic segmentation ground truth.
                * Other information that's included in the original dicts, such as:
                  "height", "width" (int): the output resolution of the model, used in inference.
                  See :meth:`postprocess` for details.

        Returns:
            @Training:
                # TODO:
            @Inference:
                list[dict]:
                    each dict has the results for one image. The dict contains the following keys:

                    * "instances": see :meth:`GeneralizedRCNN.forward` for its format.
                    * "sem_seg": see :meth:`SemanticSegmentor.forward` for its format.
                    * "panoptic_seg": See the return value of
                      :func:`combine_semantic_and_instance_outputs` for its format.
        """
        if not self.training:
            return self.inference(batched_inputs, do_postprocess=True)
        # TODO: Replace preprocessing
        # images = batched_inputs["image"]
        # images = self.preprocess_image(
        #     batched_inputs
        # )  # TODO: Dump this and let pytorch DataLoader handle batching
        images = batched_inputs["image"].to(self.device)
        images = batch_of_tensors_to_image_list(images, self.backbone.size_divisibility)
        features = self.backbone(images.tensor)

        assert "sem_seg" in batched_inputs[0]
        gt_sem_seg = [x["sem_seg"].to(self.device) for x in batched_inputs]
        gt_sem_seg = ImageList.from_tensors(
            gt_sem_seg, self.backbone.size_divisibility, self.sem_seg_head.ignore_value
        ).tensor
        sem_seg_results, sem_seg_losses = self.sem_seg_head(features, gt_sem_seg)

        gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        proposals, proposal_losses = self.proposal_generator(
            images, features, gt_instances
        )
        detector_results, detector_losses = self.roi_heads(
            images, features, proposals, gt_instances
        )

        losses = sem_seg_losses
        losses.update(proposal_losses)
        losses.update(detector_losses)
        return {
            "losses": losses,
            "sem_seg_results": sem_seg_results,
            "detector_results": detector_results,
            "features": features,
            # TODO: Other results needed for SDC
        }

    # def preprocess_image(self, batched_inputs: Dict[str, torch.Tensor]):
    #     # Compatible with torch DataLoader
    #     images = batched_inputs["image"]
    #     images = to_image_list(images, self.backbone.size_divisibility)
    #     return images

    def inference(
        self, batched_inputs: List[Dict[str, torch.Tensor]], do_postprocess: bool = True
    ):
        """mages = [x["image"] for x in batched_inputs]
        Run inference on the given inputs.

        Args:
            batched_inputs (list[dict]): same as in :meth:`forward`
            do_postprocess (bool): whether to apply post-processing on the outputs.

        Returns:
            When do_postprocess=True, see docs in :meth:`forward`.
            Otherwise, returns a (list[Instances], list[Tensor]) that contains
            the raw detector outputs, and raw semantic segmentation outputs.
        """
        # TODO: Replace preprocessing
        # images = self.preprocess_image(batched_inputs)
        # transform = T.Compose(
        #     [
        #         # ImgToFloatTensor(),
        #         T.Normalize(mean=NYUv2RawDataModule.MEAN, std=NYUv2RawDataModule.STD),
        #     ]
        # )
        # images = [x["image"].to(self.device) for x in batched_inputs]
        # images = torch.stack([transform(i).to(self.device) for i in images])
        images = batched_inputs["image"].to(self.device)
        images = batch_of_tensors_to_image_list(images, self.backbone.size_divisibility)

        features = self.backbone(images.tensor)
        sem_seg_results, sem_seg_losses = self.sem_seg_head(features, None)
        proposals, _ = self.proposal_generator(images, features, None)
        detector_results, _ = self.roi_heads(images, features, proposals, None)

        # TODO: Return logits and confidences
        # TODO: Fix postprocessing

        if do_postprocess:
            # dict of lists to list of dicts
            features = [dict(zip(features, t)) for t in zip(*features.values())]

            processed_results = []
            # for sem_seg_result, detector_result, input_per_image, image_size in zip(
            #     sem_seg_results, detector_results, batched_inputs, images.image_sizes
            # ):
            #     height = input_per_image.get("height", image_size[0])
            #     width = input_per_image.get("width", image_size[1])
            for sem_seg_result, detector_result, image_size, feature_maps in zip(
                sem_seg_results, detector_results, images.image_sizes, features
            ):
                height = image_size[0]
                width = image_size[1]
                sem_seg_r = sem_seg_postprocess(
                    sem_seg_result, image_size, height, width
                )
                detector_r = detector_postprocess(detector_result, height, width)

                processed_results.append(
                    {
                        "sem_seg": sem_seg_r,
                        "instances": detector_r,
                        "features": feature_maps,
                    }
                )

                panoptic_r = combine_semantic_and_instance_outputs(
                    detector_r,
                    sem_seg_r.argmax(dim=0),
                    self.combine_overlap_thresh,
                    self.combine_stuff_area_thresh,
                    self.combine_instances_score_thresh,
                )
                processed_results[-1]["panoptic_seg"] = panoptic_r
            return processed_results
        else:
            return {
                "sem_seg_results": sem_seg_results,
                "detector_results": detector_results,
                "features": features,
            }
