# Copyright 2024 Microsoft and the HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Processor class for Phi4Multimodal
"""

import math
from enum import Enum
from typing import Optional, Union

import torch
import torchvision

from transformers.image_processing_utils import BaseImageProcessor, BatchFeature
from transformers.image_utils import (
    ImageInput,
    make_list_of_images,
    valid_images,
)
from transformers.utils import TensorType, logging


logger = logging.get_logger(__name__)

# Special tokens
_COMPATIBLE_IMAGE_SPECIAL_TOKEN_PATTERN = r"<\|image_\d+\|>"  # For backward compatibility
_COMPATIBLE_AUDIO_SPECIAL_TOKEN_PATTERN = r"<\|audio_\d+\|>"  # For backward compatibility
_IMAGE_SPECIAL_TOKEN = "<|endoftext10|>"
_AUDIO_SPECIAL_TOKEN = "<|endoftext11|>"
_IMAGE_SPECIAL_TOKEN_ID = 200010  # '<|endoftext10|>', or we can better name it (in `tokenizer_config.json`)
_AUDIO_SPECIAL_TOKEN_ID = 200011  # '<|endoftext11|>'


class InputMode(Enum):
    LANGUAGE = 0
    VISION = 1
    SPEECH = 2
    VISION_SPEECH = 3


class Phi4MultimodalImageProcessor(BaseImageProcessor):
    r"""
    Constructs a Phi4Multimodal image processor.
    """

    model_input_names = ["input_image_embeds", "image_sizes", "image_attention_mask"]

    def __init__(
        self,
        dynamic_hd,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.dynamic_hd = dynamic_hd

    def find_closest_aspect_ratio(self, aspect_ratio, target_ratios, width, height, image_size):
        best_ratio_diff = float("inf")
        best_ratio = (1, 1)
        area = width * height
        for ratio in target_ratios:
            target_aspect_ratio = ratio[0] / ratio[1]
            ratio_diff = abs(aspect_ratio - target_aspect_ratio)
            if ratio_diff < best_ratio_diff:
                best_ratio_diff = ratio_diff
                best_ratio = ratio
            elif ratio_diff == best_ratio_diff:
                if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                    best_ratio = ratio
        return best_ratio

    def dynamic_preprocess(self, image, min_num=1, max_num=12, image_size=384, mask_size=27, use_thumbnail=True):
        orig_width, orig_height = image.size

        w_crop_num = math.ceil(orig_width / float(image_size))
        h_crop_num = math.ceil(orig_height / float(image_size))
        if w_crop_num * h_crop_num > max_num:
            aspect_ratio = orig_width / orig_height

            # calculate the existing image aspect ratio
            target_ratios = set(
                (i, j)
                for n in range(min_num, max_num + 1)
                for i in range(1, n + 1)
                for j in range(1, n + 1)
                if i * j <= max_num and i * j >= min_num
            )
            target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

            # find the closest aspect ratio to the target
            target_aspect_ratio = self.find_closest_aspect_ratio(
                aspect_ratio, target_ratios, orig_width, orig_height, image_size
            )

            # calculate the target width and height
            target_width = image_size * target_aspect_ratio[0]
            target_height = image_size * target_aspect_ratio[1]
        else:
            target_width = image_size * w_crop_num
            target_height = image_size * h_crop_num
            target_aspect_ratio = (w_crop_num, h_crop_num)

        # Calculate the ratio
        ratio_width = target_width / orig_width
        ratio_height = target_height / orig_height
        if ratio_width < ratio_height:
            new_size = (target_width, int(orig_height * ratio_width))
            padding_width = 0
            padding_height = target_height - int(orig_height * ratio_width)
        else:
            new_size = (int(orig_width * ratio_height), target_height)
            padding_width = target_width - int(orig_width * ratio_height)
            padding_height = 0

        attention_mask = torch.ones((int(mask_size * target_aspect_ratio[1]), int(mask_size * target_aspect_ratio[0])))
        if padding_width >= 14:
            attention_mask[:, -math.floor(padding_width / 14) :] = 0
        if padding_height >= 14:
            attention_mask[-math.floor(padding_height / 14) :, :] = 0
        assert attention_mask.sum() > 0

        if min(new_size[1], target_height) < 10 or min(new_size[0], target_width) < 10:
            raise ValueError(f"the aspect ratio is very extreme {new_size}")

        image = torchvision.transforms.functional.resize(
            image,
            [new_size[1], new_size[0]],
        )

        resized_img = torchvision.transforms.functional.pad(
            image, [0, 0, padding_width, padding_height], fill=[255, 255, 255]
        )

        return resized_img, attention_mask

    def pad_to_max_num_crops(self, images, max_crops=5):
        """
        images: B x 3 x H x W, B<=max_crops
        """
        B, _, H, W = images.shape
        if B < max_crops:
            pad = torch.zeros(max_crops - B, 3, H, W, dtype=images.dtype, device=images.device)
            images = torch.cat([images, pad], dim=0)
        return images

    def pad_mask_to_max_num_crops(self, masks, max_crops=5):
        B, H, W = masks.shape
        if B < max_crops:
            pad = torch.ones(max_crops - B, H, W, dtype=masks.dtype, device=masks.device)
            masks = torch.cat([masks, pad], dim=0)
        return masks

    def preprocess(
        self,
        images: ImageInput,
        return_tensors: Optional[Union[str, TensorType]] = None,
    ):
        """
        Args:
            images (`ImageInput`):
                Image to preprocess. Expects a single or batch of images with pixel values ranging from 0 to 255. If
                passing in images with pixel values between 0 and 1, set `do_rescale=False`.
            return_tensors (`str` or `TensorType`, *optional*):
                The type of tensors to return. Can be one of:
                - Unset: Return a list of `np.ndarray`.
                - `TensorType.TENSORFLOW` or `'tf'`: Return a batch of type `tf.Tensor`.
                - `TensorType.PYTORCH` or `'pt'`: Return a batch of type `torch.Tensor`.
                - `TensorType.NUMPY` or `'np'`: Return a batch of type `np.ndarray`.
                - `TensorType.JAX` or `'jax'`: Return a batch of type `jax.numpy.ndarray`.
        """
        images = make_list_of_images(images)

        if not valid_images(images):
            raise ValueError(
                "Invalid image type. Must be of type PIL.Image.Image, numpy.ndarray, "
                "torch.Tensor, tf.Tensor or jax.ndarray."
            )

        # Basic settings.
        img_processor = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        dyhd_base_resolution = 448

        # Dynamic HD
        base_resolution = dyhd_base_resolution
        images = [image.convert("RGB") for image in images]
        # cover 384 and 448 resolution
        mask_resolution = base_resolution // 14
        elems, image_attention_masks = [], []
        for im in images:
            elem, attention_mask = self.dynamic_preprocess(
                im, max_num=self.dynamic_hd, image_size=base_resolution, mask_size=mask_resolution
            )
            elems.append(elem)
            image_attention_masks.append(attention_mask)
        hd_images = [img_processor(im) for im in elems]
        global_image = [
            torch.nn.functional.interpolate(
                im.unsqueeze(0).float(),
                size=(base_resolution, base_resolution),
                mode="bicubic",
            ).to(im.dtype)
            for im in hd_images
        ]
        shapes = [[im.size(1), im.size(2)] for im in hd_images]
        mask_shapes = [[mask.size(0), mask.size(1)] for mask in image_attention_masks]
        global_attention_mask = [torch.ones((1, mask_resolution, mask_resolution)) for _ in hd_images]
        hd_images_reshape = [
            im.reshape(1, 3, h // base_resolution, base_resolution, w // base_resolution, base_resolution)
            .permute(0, 2, 4, 1, 3, 5)
            .reshape(-1, 3, base_resolution, base_resolution)
            .contiguous()
            for im, (h, w) in zip(hd_images, shapes)
        ]
        attention_masks_reshape = [
            mask.reshape(1, h // mask_resolution, mask_resolution, w // mask_resolution, mask_resolution)
            .permute(0, 1, 3, 2, 4)
            .reshape(-1, mask_resolution, mask_resolution)
            .contiguous()
            for mask, (h, w) in zip(image_attention_masks, mask_shapes)
        ]
        downsample_attention_masks = [
            mask[:, 0::2, 0::2]
            .reshape(
                1,
                h // mask_resolution,
                w // mask_resolution,
                mask_resolution // 2 + mask_resolution % 2,
                mask_resolution // 2 + mask_resolution % 2,
            )
            .permute(0, 1, 3, 2, 4)
            for mask, (h, w) in zip(attention_masks_reshape, mask_shapes)
        ]
        downsample_attention_masks = [
            mask.reshape(mask.size(1) * mask.size(2), mask.size(3) * mask.size(4))
            for mask in downsample_attention_masks
        ]
        num_img_tokens = [
            256 + 1 + int(mask.sum().item()) + int(mask[:, 0].sum().item()) + 16 for mask in downsample_attention_masks
        ]

        hd_images_reshape = [
            torch.cat([_global_image] + [_im], dim=0) for _global_image, _im in zip(global_image, hd_images_reshape)
        ]
        hd_masks_reshape = [
            torch.cat([_global_mask] + [_mask], dim=0)
            for _global_mask, _mask in zip(global_attention_mask, attention_masks_reshape)
        ]
        max_crops = max([img.size(0) for img in hd_images_reshape])
        image_transformed = [self.pad_to_max_num_crops(im, max_crops) for im in hd_images_reshape]
        image_transformed = torch.stack(image_transformed, dim=0)
        mask_transformed = [self.pad_mask_to_max_num_crops(mask, max_crops) for mask in hd_masks_reshape]
        mask_transformed = torch.stack(mask_transformed, dim=0)

        returned_input_image_embeds = image_transformed
        returned_image_sizes = torch.tensor(shapes, dtype=torch.long)
        returned_image_attention_mask = mask_transformed
        returned_num_img_tokens = num_img_tokens

        data = {
            "input_image_embeds": returned_input_image_embeds,
            "image_sizes": returned_image_sizes,
            "image_attention_mask": returned_image_attention_mask,
            "num_img_tokens": returned_num_img_tokens,
        }

        return BatchFeature(data=data, tensor_type=return_tensors)


__all__ = ["Phi4MultimodalImageProcessor"]
