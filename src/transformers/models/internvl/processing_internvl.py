# coding=utf-8
# Copyright 2024 HuggingFace Inc. team. All rights reserved.
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


from typing import List, Optional, Union

from transformers.processing_utils import ImagesKwargs, ProcessingKwargs, ProcessorMixin, Unpack
from transformers.tokenization_utils_base import PreTokenizedInput, TextInput

from ...image_processing_utils import BatchFeature
from ...image_utils import ImageInput, make_batched_videos, make_flat_list_of_images


class InternVLImagesKwargs(ImagesKwargs, total=False):
    min_patches: Optional[int]
    max_patches: Optional[int]


class InternVLProcessorKwargs(ProcessingKwargs, total=False):
    images_kwargs: InternVLImagesKwargs
    _defaults = {
        "text_kwargs": {
            "padding_side": "left",
        },
        "images_kwargs": {
            "min_patches": 1,
            "max_patches": 12,
        },
    }


class InternVLProcessor(ProcessorMixin):
    r"""
    Constructs a InternVL processor which wraps a [`GotOcr2ImageProcessor`] and
    [`PretrainedTokenizerFast`] tokenizer into a single processor that inherits both the image processor and
    tokenizer functionalities. See the [`~InternVLProcessor.__call__`] and [`~InternVLProcessor.decode`] for more information.
    Args:
        image_processor ([`GotOcr2ImageProcessor`], *optional*):
            The image processor is a required input.
        tokenizer ([`PreTrainedTokenizer`, `PreTrainedTokenizerFast`], *optional*):
            The tokenizer is a required input.
        image_seq_length (`int`, *optional*, defaults to 256):
            The number of image token to use per image patch.
        chat_template (`str`, *optional*): A Jinja template which will be used to convert lists of messages
            in a chat into a tokenizable string.
    """

    attributes = ["image_processor", "tokenizer"]
    valid_kwargs = ["chat_template", "image_seq_length"]
    image_processor_class = "AutoImageProcessor"
    tokenizer_class = "AutoTokenizer"

    def __init__(
        self, image_processor=None, tokenizer=None, image_seq_length: int = 256, chat_template=None, **kwargs
    ):
        self.image_seq_length = image_seq_length

        super().__init__(image_processor, tokenizer, chat_template=chat_template)

    def __call__(
        self,
        images: Optional[ImageInput] = None,
        text: Optional[Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]]] = None,
        audio=None,
        videos=None,
        **kwargs: Unpack[InternVLProcessorKwargs],
    ) -> BatchFeature:
        """
        Main method to prepare for the model one or several sequences(s) and image(s). This method forwards the `text`
        and `kwargs` arguments to PreTrainedTokenizerFast's [`~PreTrainedTokenizerFast.__call__`] to encode the text if `text`
        is not `None`, otherwise encode default OCR queries which depends on the `format`, `box`, `color`, `multi_page` and
        `crop_to_patches` arguments. To prepare the vision inputs, this method forwards the `images` and `kwrags` arguments to
        GotOcr2ImageProcessor's [`~GotOcr2ImageProcessor.__call__`] if `images` is not `None`.

        Args:
            images (`PIL.Image.Image`, `np.ndarray`, `torch.Tensor`, `List[PIL.Image.Image]`, `List[np.ndarray]`, `List[torch.Tensor]`):
                The image or batch of images to be prepared. Each image can be a PIL image, NumPy array or PyTorch
                tensor. Both channels-first and channels-last formats are supported.
            text (`str`, `List[str]`, `List[List[str]]`):
                The sequence or batch of sequences to be encoded. Each sequence can be a string or a list of strings
                (pretokenized string). If the sequences are provided as list of strings (pretokenized), you must set
                `is_split_into_words=True` (to lift the ambiguity with a batch of sequences).

            return_tensors (`str` or [`~utils.TensorType`], *optional*):
                If set, will return tensors of a particular framework. Acceptable values are:
                - `'tf'`: Return TensorFlow `tf.constant` objects.
                - `'pt'`: Return PyTorch `torch.Tensor` objects.
                - `'np'`: Return NumPy `np.ndarray` objects.
                - `'jax'`: Return JAX `jnp.ndarray` objects.

        Returns:
            [`BatchFeature`]: A [`BatchFeature`] with the following fields:

            - **input_ids** -- List of token ids to be fed to a model. Returned when `text` is not `None`.
            - **attention_mask** -- List of indices specifying which tokens should be attended to by the model (when
              `return_attention_mask=True` or if *"attention_mask"* is in `self.model_input_names` and if `text` is not
              `None`).
            - **pixel_values** -- Pixel values to be fed to a model. Returned when `images` is not `None`.
        """
        if text is None:
            raise ValueError("You have to specify text.")

        output_kwargs = self._merge_kwargs(
            InternVLProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )

        min_patches = output_kwargs["images_kwargs"].pop("min_patches")
        max_patches = output_kwargs["images_kwargs"].pop("max_patches")

        if not isinstance(text, (list, tuple)):
            text = [text]
        image_video_inputs = {}
        if images is not None or videos is not None:
            images = make_flat_list_of_images(images) if images is not None else None
            videos = make_batched_videos(videos) if videos is not None else None
            image_index = 0
            video_index = 0
            processed_text = []
            image_video_inputs = []  # List to store processed image/video patches
            # Support interlaced image and video in prompts
            for prompt in text:
                new_prompt = prompt
                while "<image>" in new_prompt or "<video>" in new_prompt:
                    if "<image>" in new_prompt and (
                        "<video>" not in new_prompt or new_prompt.index("<image>") < new_prompt.index("<video>")
                    ):
                        image_patches = self.image_processor.crop_image_to_patches(
                            images[image_index],
                            patch_size=output_kwargs["images_kwargs"].get("size"),
                            min_patches=min_patches,
                            max_patches=max_patches,
                        )
                        image_video_inputs.append(image_patches)
                        new_prompt = new_prompt.replace(
                            "<image>", f"<img>{'<IMG_CONTEXT>' * self.image_seq_length * len(image_patches)}</img>", 1
                        )
                        image_index += 1
                    else:
                        video = videos[video_index]
                        for index_image, image_group in enumerate(video):
                            image_group = self.image_processor.crop_image_to_patches(
                                image_group,
                                patch_size=output_kwargs["images_kwargs"].get("size"),
                                min_patches=1,
                                max_patches=1,
                            )
                            video[index_image] = image_group
                        video_prompt = "\n".join(
                            f"Frame{i+1}: <img>{'<IMG_CONTEXT>'*self.image_seq_length* len(video[i])}</img>"
                            for i in range(len(video))
                        )
                        new_prompt = new_prompt.replace("<video>", video_prompt, 1)
                        image_video_inputs.append([image for group in video for image in group])
                        video_index += 1

                processed_text.append(new_prompt)
            text = processed_text
            # Single call to process all interleaved image/video patches
            image_video_inputs = self.image_processor(images=image_video_inputs, **output_kwargs["images_kwargs"])

        text_inputs = self.tokenizer(text, **output_kwargs["text_kwargs"])

        return BatchFeature(data={**text_inputs, **image_video_inputs})

    def batch_decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to PreTrainedTokenizerFast's [`~PreTrainedTokenizer.batch_decode`]. Please
        refer to the docstring of this method for more information.
        """
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to PreTrainedTokenizerFast's [`~PreTrainedTokenizer.decode`]. Please refer to
        the docstring of this method for more information.
        """
        return self.tokenizer.decode(*args, **kwargs)

    @property
    def model_input_names(self):
        tokenizer_input_names = self.tokenizer.model_input_names
        image_processor_input_names = self.image_processor.model_input_names
        return list(tokenizer_input_names) + list(image_processor_input_names)


__all__ = ["InternVLProcessor"]
