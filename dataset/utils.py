from typing import List

import numpy as np
import torch
from detectron2.layers import shapes_to_tensor
from detectron2.structures import ImageList
from torch.nn import functional as F


class ImgToFloatTensor:
    def __call__(self, img: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(img.astype("float32").transpose(2, 0, 1))


def batch_of_tensors_to_image_list(
    tensors: torch.Tensor, size_divisibility: int = 0, pad_value: float = 0.0
) -> ImageList:
    # original implementation ImageList.from_tensors expects list of tensors
    # but torch DataLoader will return tensor of tensors as a batch

    assert len(tensors) > 0
    # assert isinstance(tensors, (tuple, list))
    # for t in tensors:
    #     assert isinstance(t, torch.Tensor), type(t)
    #     assert t.shape[:-2] == tensors[0].shape[:-2], t.shape

    image_sizes = [(im.shape[-2], im.shape[-1]) for im in tensors]
    image_sizes_tensor = [shapes_to_tensor(x) for x in image_sizes]
    max_size = torch.stack(image_sizes_tensor).max(0).values

    if size_divisibility > 1:
        stride = size_divisibility
        # the last two dims are H,W, both subject to divisibility requirement
        max_size = (max_size + (stride - 1)) // stride * stride

    # handle weirdness of scripting and tracing ...
    if torch.jit.is_scripting():
        max_size: List[int] = max_size.to(dtype=torch.long).tolist()
    else:
        if torch.jit.is_tracing():
            image_sizes = image_sizes_tensor

    if len(tensors) == 1:
        # This seems slightly (2%) faster.
        image_size = image_sizes[0]
        padding_size = [
            0,
            max_size[-1] - image_size[1],
            0,
            max_size[-2] - image_size[0],
        ]
        batched_imgs = F.pad(tensors[0], padding_size, value=pad_value).unsqueeze_(0)
    else:
        # max_size can be a tensor in tracing mode, therefore convert to list
        batch_shape = [len(tensors)] + list(tensors[0].shape[:-2]) + list(max_size)
        batched_imgs = tensors[0].new_full(batch_shape, pad_value)
        for img, pad_img in zip(tensors, batched_imgs):
            pad_img[..., : img.shape[-2], : img.shape[-1]].copy_(img)

    return ImageList(batched_imgs.contiguous(), image_sizes)
