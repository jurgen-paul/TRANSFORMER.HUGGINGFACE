#                🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨
#           This file was automatically generated from src/transformers/models/gemma3/modular_gemma3.py.
#               Do NOT edit this file manually as any edits will be overwritten by the generation of
#             the file from the modular. If any change should be done, please apply the change to the
#                          modular_gemma3.py file directly. One of our CI enforces this.
#                🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨
# coding=utf-8
# Copyright 2025 Google Inc. HuggingFace Inc. team. All rights reserved.
#
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
import itertools
import math
import re
from collections.abc import Sequence
from typing import Any, Optional, Union, cast

import PIL.Image
import torch

from ...feature_extraction_utils import BatchFeature
from ...processing_utils import (
    ImagesKwargs,
    ProcessingKwargs,
    ProcessorMixin,
    TextKwargs,
    Unpack,
    _validate_images_text_input_order,
)
from ...tokenization_utils_base import TextInput
from ..gemma import GemmaTokenizer, GemmaTokenizerFast
from ..siglip import SiglipImageProcessor


class Gemma3TextKwargs(TextKwargs):
    pass


class Gemma3ImagesKwargs(ImagesKwargs):
    do_pan_and_scan: Optional[bool]
    pan_and_scan_min_crop_size: Optional[int]
    pan_and_scan_max_num_crops: Optional[int]
    pan_and_scan_min_ratio_to_activate: Optional[float]
    do_convert_rgb: Optional[bool]


class Gemma3ProcessorKwargs(ProcessingKwargs, total=False):
    text_kwargs: Gemma3TextKwargs
    images_kwargs: Gemma3ImagesKwargs
    _defaults = {
        "text_kwargs": {
            "padding": False,
        },
        "images_kwargs": {
            "data_format": "channels_first",
            "do_pan_and_scan": False,
            "pan_and_scan_min_crop_size": 256,
            "pan_and_scan_max_num_crops": 4,
            "pan_and_scan_min_ratio_to_activate": 1.2,
            "do_resize": True,
            "size": {"height": 896, "width": 896},
            "resample": PIL.Image.Resampling.BICUBIC,
            "do_rescale": True,
            "rescale_factor": 1 / 255,
            "do_normalize": True,
            "image_mean": (127.5,) * 3,
            "image_std": (127.5,) * 3,
            "do_convert_rgb": None,
        },
    }


BatchedImageInput = Sequence[PIL.Image.Image]
BatchedMultiImageInput = Sequence[BatchedImageInput]
Gemma3ProcessorImageInput = Union[PIL.Image.Image, BatchedImageInput, BatchedMultiImageInput]

PanAndScannedImage = tuple[PIL.Image.Image, Sequence[PIL.Image.Image]]
BatchedPanAndScannedImage = Sequence[Sequence[PanAndScannedImage]]
MutablePanAndScannedImage = tuple[PIL.Image.Image, list[PIL.Image.Image]]
MutableBatchedPanAndScannedImage = list[list[MutablePanAndScannedImage]]

TextInputTypes = Union[TextInput, Sequence[TextInput]]


def pan_and_scan(
    image: PIL.Image.Image,
    pan_and_scan_min_crop_size: int,
    pan_and_scan_max_num_crops: int,
    pan_and_scan_min_ratio_to_activate: float,
    **unused_kwargs,
) -> Sequence[PIL.Image.Image]:
    w, h = image.size

    # Square or landscape image.
    if w >= h:
        # Only apply PaS if the image is sufficiently exaggerated
        if w / h < pan_and_scan_min_ratio_to_activate:
            return []

        # Select ideal number of crops close to the image aspect ratio and such that crop_size > min_crop_size.
        num_crops_w = int(math.floor(w / h + 0.5))  # Half round up rounding.
        num_crops_w = min(int(math.floor(w / pan_and_scan_min_crop_size)), num_crops_w)

        # Make sure the number of crops is in range [2, pan_and_scan_max_num_crops].
        num_crops_w = max(2, num_crops_w)
        num_crops_w = min(pan_and_scan_max_num_crops, num_crops_w)
        num_crops_h = 1

    # Portrait image.
    else:
        # Only apply PaS if the image is sufficiently exaggerated
        if h / w < pan_and_scan_min_ratio_to_activate:
            return []

        # Select ideal number of crops close to the image aspect ratio and such that crop_size > min_crop_size.
        num_crops_h = int(math.floor(h / w + 0.5))
        num_crops_h = min(int(math.floor(h / pan_and_scan_min_crop_size)), num_crops_h)

        # Make sure the number of crops is in range [2, pan_and_scan_max_num_crops].
        num_crops_h = max(2, num_crops_h)
        num_crops_h = min(pan_and_scan_max_num_crops, num_crops_h)
        num_crops_w = 1

    crop_size_w = int(math.ceil(w / num_crops_w))
    crop_size_h = int(math.ceil(h / num_crops_h))

    # Don't apply PaS if crop size is too small.
    if min(crop_size_w, crop_size_h) < pan_and_scan_min_crop_size:
        return []

    crop_positions_w = [crop_size_w * i for i in range(num_crops_w)]
    crop_positions_h = [crop_size_h * i for i in range(num_crops_h)]

    # Generate crops.
    return [
        image.crop((pos_w, pos_h, pos_w + crop_size_w, pos_h + crop_size_h))
        for pos_h, pos_w in itertools.product(crop_positions_h, crop_positions_w)
    ]


class Gemma3Processor(ProcessorMixin):
    attributes = ["image_processor", "tokenizer"]
    valid_kwargs = ["chat_template"]
    image_processor_class = "SiglipImageProcessor"
    tokenizer_class = ("GemmaTokenizer", "GemmaTokenizerFast")

    def __init__(
        self,
        image_processor: SiglipImageProcessor,
        tokenizer: Union[GemmaTokenizer, GemmaTokenizerFast],
        chat_template: Optional[str] = None,
        num_mm_soft_tokens_per_image: int = 256,
        **kwargs,
    ):
        try:
            self.image_seq_length = getattr(image_processor, "image_seq_length")
        except AttributeError as e:
            raise ValueError("`image_processor` is missing the required `image_seq_length` attribute.") from e

        self.image_token_id = tokenizer.convert_tokens_to_ids("<image_soft_token>")
        self.full_image_sequence = (
            "\n\n<start_of_image>"
            + "".join(["<image_soft_token>"] * num_mm_soft_tokens_per_image)
            + "<end_of_image>\n\n"
        )

        super().__init__(
            image_processor=image_processor,
            tokenizer=tokenizer,
            chat_template=chat_template,
            **kwargs,
        )

    def __call__(
        self,
        images: Optional[Gemma3ProcessorImageInput] = None,
        text: Optional[TextInputTypes] = None,
        videos: Optional[Any] = None,
        audio: Optional[Any] = None,
        **kwargs: Unpack[Gemma3ProcessorKwargs],
    ) -> BatchFeature:
        del videos, audio  # Unsupported modalities for Gemma 3

        if text is None and images is None:
            raise ValueError("Provide at least one of `text` or `images`.")

        # Check if images and text inputs are reversed for backward compatibility
        images, text = _validate_images_text_input_order(images, text)

        output_kwargs = self._merge_kwargs(
            Gemma3ProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )

        if images is not None:
            batched_images = self._process_images(images=images, **output_kwargs["images_kwargs"])
            batch_flattened_images = self._batch_flatten_pas_images(batched_images=batched_images)
            # The Hugging Face implementation of the SigLIP Vision Model expects a single Tensor of shape [F, C, W, H]
            # where:
            #
            # - F is the number of images to encode
            # - C is the number of channels in each image
            # - W and H are the width and height of the image, which in this case are the same since Gemma 3 only
            #   supports 896x896 images.
            #
            # So we concat all images across all batches into a single flat list prior to sending it to the
            # `Gemma3ForConditionalGeneration.vision_model` for ecnoding and use `torch.masked_scatter()` in that the
            # `Gemma3ForConditionalGeneration` model class to sequentially update the language model embeddings wth the
            # pooled vision embdeddings.
            pixel_values = torch.cat(
                [
                    self.image_processor(prompt_images, **output_kwargs["images_kwargs"])["pixel_values"]
                    for prompt_images in batch_flattened_images
                ]
            )
        else:
            batched_images = None
            pixel_values = None

        batched_input = self._process_text(text=text, batched_images=batched_images, **output_kwargs["text_kwargs"])

        if pixel_values is not None:
            batched_input.update(
                pixel_values=pixel_values,
                image_soft_token_mask=batched_input["input_ids"] == self.image_token_id,
            )

        return batched_input

    def _process_images(
        self, images: Gemma3ProcessorImageInput, **kwargs: Unpack[Gemma3ImagesKwargs]
    ) -> BatchedPanAndScannedImage:
        if isinstance(images, PIL.Image.Image):
            images_lists: MutableBatchedPanAndScannedImage = [[(images, [])]]
        elif isinstance(images[0], PIL.Image.Image):
            images = cast(BatchedImageInput, images)
            images_lists: MutableBatchedPanAndScannedImage = [[(i, [])] for i in images]
        else:
            images = cast(BatchedMultiImageInput, images)
            images_lists: MutableBatchedPanAndScannedImage = [[(i, []) for i in il] for il in images]

        if getattr(kwargs, "do_pan_and_scan", False):
            for images_list in images_lists:
                for image, crops in images_list:
                    crops.extend(pan_and_scan(image=image, **kwargs))

        return images_lists

    def _process_text(
        self,
        text: Optional[TextInputTypes] = None,
        batched_images: Optional[BatchedPanAndScannedImage] = None,
        **kwargs: Unpack[Gemma3TextKwargs],
    ) -> BatchFeature:
        if batched_images and not text:
            text = [" ".join(["<image>"] * len(images)) for images in batched_images]

        if batched_images and text:
            if isinstance(text, str):
                text = [text]

            if (bi_l := len(batched_images)) != (t_l := len(text)):
                raise ValueError(f"Received inconsistently sized batches of images ({bi_l}) and text ({t_l}).")

            for prompt, images in zip(text, batched_images):
                image_indexes = [m.start() for m in re.finditer("<image>", prompt)]

                if (i_l := len(images)) != (idx_l := len(image_indexes)):
                    raise ValueError(f"Prompt contained {idx_l} image tokens but received {i_l} images.")

                # Insert additional image tokens for Pan-and-Scan crops
                for (_, pas_images), idx in reversed(list(zip(images, image_indexes))):
                    if pas_images:
                        formatted_image_text = (
                            "Here is the original image <image> and here are some crops to help you see better "
                            + " ".join(["<image>"] * len(pas_images))
                        )
                        prompt = prompt[:idx] + formatted_image_text + prompt[idx + len("<image>") :]

            # Expand placeholder image tokens to the full image token sequence
            text = [prompt.replace("<image>", self.full_image_sequence) for prompt in text]

        inputs = self.tokenizer(text=text, **kwargs)
        return BatchFeature({**inputs})

    def _batch_flatten_pas_images(
        self,
        batched_images: BatchedPanAndScannedImage,
    ) -> Sequence[Sequence[PIL.Image.Image]]:
        """Converts the Sequence[tuple[Image, Sequence[Image]]] into a Sequence[Image]"""
        batch_flattened: list[list[PIL.Image.Image]] = []

        for images in batched_images:
            prompt_flattened: list[PIL.Image.Image] = []
            for image, pas_images in images:
                prompt_flattened.extend([image] + pas_images)
            batch_flattened.append(prompt_flattened)

        return batch_flattened

    # Copied from transformers.models.clip.processing_clip.CLIPProcessor.batch_decode with CLIP->Gemma
    def batch_decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to GemmaTokenizerFast's [`~PreTrainedTokenizer.batch_decode`]. Please
        refer to the docstring of this method for more information.
        """
        return self.tokenizer.batch_decode(*args, **kwargs)

    # Copied from transformers.models.clip.processing_clip.CLIPProcessor.decode with CLIP->Gemma
    def decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to GemmaTokenizerFast's [`~PreTrainedTokenizer.decode`]. Please refer to
        the docstring of this method for more information.
        """
        return self.tokenizer.decode(*args, **kwargs)

    @property
    # Copied from transformers.models.clip.processing_clip.CLIPProcessor.model_input_names with CLIP->PaliGemma
    def model_input_names(self):
        tokenizer_input_names = self.tokenizer.model_input_names
        image_processor_input_names = self.image_processor.model_input_names
        return list(dict.fromkeys(tokenizer_input_names + image_processor_input_names))


__all__ = ["Gemma3Processor"]
