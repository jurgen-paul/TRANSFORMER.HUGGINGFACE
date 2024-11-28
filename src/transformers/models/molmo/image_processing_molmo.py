#                🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨
#           This file was automatically generated from src/transformers/models/molmo/modular_molmo.py.
#               Do NOT edit this file manually as any edits will be overwritten by the generation of
#             the file from the modular. If any change should be done, please apply the change to the
#                          modular_molmo.py file directly. One of our CI enforces this.
#                🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨
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


from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from ...feature_extraction_utils import BatchFeature
from ...image_processing_utils import BaseImageProcessor, get_size_dict
from ...image_transforms import convert_to_rgb, normalize, pad, resize
from ...image_utils import (
    OPENAI_CLIP_MEAN,
    OPENAI_CLIP_STD,
    ChannelDimension,
    ImageInput,
    PILImageResampling,
    get_image_size,
    infer_channel_dimension_format,
    is_scaled_image,
    is_valid_image,
    to_numpy_array,
    valid_images,
    validate_kwargs,
    validate_preprocess_arguments,
)
from ...utils import TensorType, logging


logger = logging.get_logger(__name__)


### IMAGE PROCESSING CODE


def make_batched_images(images) -> List[List[ImageInput]]:
    """
    Accepts images in list or nested list format, and makes a list of images for preprocessing.

    Args:
        images (`Union[List[List[ImageInput]], List[ImageInput], ImageInput]`):
            The input image.

    Returns:
        list: A list of images.
    """
    if isinstance(images, (list, tuple)) and isinstance(images[0], (list, tuple)) and is_valid_image(images[0][0]):
        return [img for img_list in images for img in img_list]

    elif isinstance(images, (list, tuple)) and is_valid_image(images[0]):
        return images

    elif is_valid_image(images):
        return [images]

    raise ValueError(f"Could not make batched video from {images}")


def get_resize_output_image_size(
    image: np.ndarray,
    size: Union[int, Tuple[int, int], List[int], Tuple[int]],
) -> tuple:
    original_height, original_width = get_image_size(image)

    scale_y = size["height"] / original_height
    scale_x = size["width"] / original_width
    scale = min(scale_x, scale_y)

    # Compute new dimensions
    new_height = round(original_height * scale)
    new_width = round(original_width * scale)
    return {"height": new_height, "width": new_width}


def pad_to_bounding_box(
    image: np.ndarray, offset_height: int, offset_width: int, target_height: int, target_width: int, value: int = 0
) -> np.ndarray:
    """
    Pad the input image to the target height and width using the transformers `pad` function.

    Args:
        image: The input image to be padded.
        offset_height: The number of pixels to add to the top of the image.
        offset_width: The number of pixels to add to the left of the image.
        target_height: The target height of the padded image.
        target_width: The target width of the padded image.
        value: The constant value used for padding (default is 0).

    Returns:
        A padded image of size (target_height, target_width).
    """
    height, width = image.shape[:2]
    after_padding_height = target_height - offset_height - height
    after_padding_width = target_width - offset_width - width
    return np.pad(
        image,
        [
            (offset_height, after_padding_height),
            (offset_width, after_padding_width),
            (0, 0),  # don't pad on the channel dim
        ],
        mode="constant",
        constant_values=value,
    )


class MolmoImageProcessor(BaseImageProcessor):
    """
    Image processor for the Molmo model.

    This processor handles resizing, padding, grid shape, and patch extraction from images,
    converting them into inputs suitable for the Molmo model.
    """

    model_input_names = ["pixel_values", "input_ids", "image_input_idx", "image_masks"]

    def __init__(
        self,
        max_num_crops: int = 12,
        overlap_margins: Tuple[int, int] = [4, 4],
        size: Dict[str, int] = None,
        tokens_per_image_width: int = 12,
        tokens_per_image_height: int = 12,
        image_patch_size: int = 14,
        image_padding_mask: bool = True,
        do_normalize: bool = True,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        do_convert_rgb: bool = True,
        do_resize: bool = True,
        resample: PILImageResampling = PILImageResampling.BILINEAR,
        do_pad: Optional[bool] = True,
        padding_value: float = 1.0,
        padding_mode: str = "constant",
        do_split_into_crops: bool = True,
        do_rescale: bool = True,
        rescale_factor: Union[int, float] = 1 / 255,
        image_patch_token: str = "<im_patch>",
        image_column_token: str = "<im_col>",
        image_start_token: str = "<im_start>",
        image_end_token: str = "<im_end>",
        **kwargs,
    ):
        super().__init__(**kwargs)
        size = size if size is not None else {"height": 336, "width": 336}
        size = get_size_dict(size, default_to_square=False)

        self.do_resize = do_resize
        self.size = size
        self.resample = resample
        self.do_pad = do_pad
        self.padding_value = padding_value
        self.padding_mode = padding_mode
        self.do_split_into_crops = do_split_into_crops
        self.do_rescale = do_rescale
        self.rescale_factor = rescale_factor
        self.max_num_crops = max_num_crops
        self.overlap_margins = overlap_margins
        self.tokens_per_image_width = tokens_per_image_width
        self.tokens_per_image_height = tokens_per_image_height
        self.image_patch_size = image_patch_size
        self.image_padding_mask = image_padding_mask
        self.do_normalize = do_normalize
        self.image_mean = image_mean if image_mean is not None else OPENAI_CLIP_MEAN
        self.image_std = image_std if image_std is not None else OPENAI_CLIP_STD
        self.do_convert_rgb = do_convert_rgb
        self.image_patch_token = image_patch_token
        self.image_column_token = image_column_token
        self.image_start_token = image_start_token
        self.image_end_token = image_end_token
        self._valid_processor_keys = [
            "images",
            "do_resize",
            "size",
            "resample",
            "do_rescale",
            "rescale_factor",
            "do_normalize",
            "image_mean",
            "image_std",
            "do_convert_rgb",
            "return_tensors",
            "data_format",
            "input_data_format",
            "do_pad",
            "do_split_into_crops",
            "padding_mode",
            "padding_value",
        ]

        # TODO move these to configuration once processing is done.
        self.tokens_per_image = tokens_per_image_height * tokens_per_image_width
        self.patches_per_image_width = size["width"] // image_patch_size
        self.patches_per_image_height = size["height"] // image_patch_size
        self.total_margin_pixels = image_patch_size * (overlap_margins[1] + overlap_margins[0])
        self.crop_patches = self.size["width"] // self.image_patch_size  # patches per crop dim
        self.crop_window_patches = self.crop_patches - (
            self.overlap_margins[1] + self.overlap_margins[0]
        )  # usable patches
        self.crop_window_size = self.crop_window_patches * self.image_patch_size
        self.crop_size = size["width"]

    def resize(
        self,
        image: np.ndarray,
        size: Dict[str, int],
        resample: PILImageResampling = PILImageResampling.BICUBIC,
        data_format: Optional[Union[str, ChannelDimension]] = None,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
        **kwargs,
    ) -> np.ndarray:
        """
        Resize an image. The shortest edge of the image is resized to size["shortest_edge"], with the longest edge
        resized to keep the input aspect ratio.

        Args:
            image (`np.ndarray`):
                Image to resize.
            size (`Dict[str, int]`):
                Size of the output image.
            resample (`PILImageResampling`, *optional*, defaults to `PILImageResampling.BICUBIC`):
                Resampling filter to use when resiizing the image.
            data_format (`str` or `ChannelDimension`, *optional*):
                The channel dimension format of the image. If not provided, it will be the same as the input image.
            input_data_format (`ChannelDimension` or `str`, *optional*):
                The channel dimension format of the input image. If not provided, it will be inferred.
        """
        size = get_size_dict(size)
        if "height" not in size or "width" not in size:
            raise ValueError(f"The `size` dictionary must contain the keys `height` and `width`. Got {size.keys()}")
        output_size = (size["height"], size["width"])

        return resize(
            image,
            size=output_size,
            resample=resample,
            data_format=data_format,
            input_data_format=input_data_format,
            **kwargs,
        )

    def pad(
        self,
        image: np.ndarray,
        size: Dict[str, int],
        mode: str = "constant",
        constant_values: float = 1.0,
        data_format: Optional[Union[str, ChannelDimension]] = None,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
    ) -> np.ndarray:
        """
        Pad an image to `(size["height"], size["width"])`.

        Args:
            image (`np.ndarray`):
                Image to pad.
            size (`Dict[str, int]`):
                Dictionary in the format `{"height": int, "width": int}` specifying the size of the output image.
            data_format (`ChannelDimension` or `str`, *optional*):
                The data format of the output image. If unset, the same format as the input image is used.
            input_data_format (`ChannelDimension` or `str`, *optional*):
                The channel dimension format of the input image. If not provided, it will be inferred.
        """
        if "height" not in size or "width" not in size:
            raise ValueError("Size must contain 'height' and 'width'.")
        new_size = get_resize_output_image_size(image, size)
        padding_height = size["height"] - new_size["height"]
        padding_width = size["width"] - new_size["width"]
        padding_top = padding_height // 2
        padding_bottom = padding_height - padding_top
        padding_left = padding_width // 2
        padding_right = padding_width - padding_left

        padded_image = pad(
            image,
            padding=((padding_top, padding_bottom), (padding_left, padding_right)),
            mode=mode,
            constant_values=constant_values,
            data_format=data_format,
            input_data_format=input_data_format,
        )

        mask_padding = [
            [padding_top, size["height"] - new_size["height"] - padding_top],
            [padding_left, size["width"] - new_size["width"] - padding_left],
        ]
        if input_data_format == ChannelDimension.FIRST:
            image_to_pad = image[0, :, :]
        elif input_data_format == ChannelDimension.LAST:
            image_to_pad = image[:, :, 0]
        else:
            raise ValueError(f"Invalid channel dimension format: {input_data_format}")

        image_mask = np.pad(np.ones_like(image_to_pad, dtype=bool), mask_padding)

        return padded_image, image_mask

    def find_best_crop_grid_for_image_size(self, image: ImageInput):
        """
        Decide how best to divide an image of size {"width": width, "height": height}]
        in up to max_num_crops of size crop_size
        """
        original_size = np.array(
            [image.shape[0] - self.total_margin_pixels, image.shape[1] - self.total_margin_pixels], dtype=np.float32
        )
        crop_grid = [(i, j) for i in range(1, self.max_num_crops + 1) for j in range(1, (self.max_num_crops // i) + 1)]

        # sort so argmin and argmax favour smaller crop_grid in the event of a tie
        crop_grid.sort(key=lambda x: (x[0] * x[1], x[0]))
        candidate_crop_grid = np.array(crop_grid, dtype=np.int32)  # [n_resolutions, 2]
        candidate_resolutions = candidate_crop_grid * self.crop_window_size  # [n_resolutions, 2]

        required_scale_step = candidate_resolutions.astype(np.float32) / original_size
        required_scale = np.min(required_scale_step, axis=-1, keepdims=True)  # [n_resolutions, 1]

        if np.all(required_scale < 1):
            # min downscaling
            selected_index = np.argmax(required_scale)
        else:
            # same with upscaling
            required_scale = np.where(required_scale < 1.0, np.inf, required_scale)
            selected_index = np.argmin(required_scale)

        return candidate_crop_grid[selected_index]

    def reshape_into_patches(self, global_image, input_data_format):
        if input_data_format == ChannelDimension.FIRST:
            global_image = np.transpose(global_image, (1, 2, 0))
        channels = global_image.shape[-1]

        global_image = global_image.reshape(
            self.patches_per_image_height,
            self.image_patch_size,
            self.patches_per_image_width,
            self.image_patch_size,
            channels,
        )
        global_image = global_image.transpose(0, 2, 1, 3, 4)
        global_image = global_image.reshape(
            self.patches_per_image_width * self.patches_per_image_height,
            self.image_patch_size * self.image_patch_size * channels,
        )
        return global_image

    def split_image_into_crops(
        self,
        image: np.ndarray,
        image_mask: np.ndarray,
        crop_grid: Tuple[int, int],
        input_data_format,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Split the image into crops (patches), while keeping track of the patch ordering and generating masks for each crop.

        Args:
            image: The resized and padded image as a NumPy array.
            image_mask: The mask corresponding to the image, indicating valid pixels.
            crop_grid: Tuple (num_rows, num_cols) representing how the image is divided into crops (crop grid).
            crop_stride: The step size or stride used to move between crops.
            patch_grid_height: The number of patches along the height of the image grid.
            patch_grid_width: The number of patches along the width of the image grid.

        Returns:
            crops: Array of image patches/crops.
            patch_ordering: Array representing the ordering of patches within the original image.
            cropped_masks: Array of masks corresponding to the image crops.
        """
        if input_data_format == ChannelDimension.FIRST:
            image = np.transpose(image, (1, 2, 0))
        crops = []
        cropped_masks = []
        patch_orderings = []

        # Check if patch grid size matches expected dimensions
        if ((self.patches_per_image_height + 1) // 2 != self.tokens_per_image_height) or (
            (self.patches_per_image_width + 1) // 2 != self.tokens_per_image_width
        ):
            raise ValueError("Number of patches per crop does not fit number of tokens per image dimension.")

        patch_index = 0  # Track the index for patch ordering
        for row in range(crop_grid[0]):  # Loop over rows of crops
            crop_y_start = row * self.crop_window_size

            # calculate crop height, accounting for margins (there are overlaps, remember)
            current_crop_height = self.patches_per_image_height - (self.overlap_margins[1] + self.overlap_margins[0])
            if row == 0:  # add left margin for the first row
                current_crop_height += self.overlap_margins[0]
            if row == (crop_grid[0] - 1):  # add right margin for the last row
                current_crop_height += self.overlap_margins[1]

            crop_y_offset = self.overlap_margins[0] // 2 if row > 0 else 0
            for column in range(crop_grid[1]):  # Loop over columns of crops
                crop_x_start = column * self.crop_window_size

                # Calculate crop width, accounting for margins
                current_crop_width = self.patches_per_image_width - (self.overlap_margins[1] + self.overlap_margins[0])
                if column == 0:  # add left margin for the first column
                    current_crop_width += self.overlap_margins[0]
                if column == (crop_grid[1] - 1):  # add right margin for the last column
                    current_crop_width += self.overlap_margins[1]

                pooled_width = (current_crop_width + 1) // 2
                pooled_height = (current_crop_height + 1) // 2

                # Correct padding based on margins and offsets
                crop_x_offset = self.overlap_margins[0] // 2 if column > 0 else 0

                # Track patch ordering: generate an array representing the order of patches (overlaps (on crops))
                reshaped_image = np.reshape(
                    np.arange(patch_index, patch_index + pooled_height * pooled_width, dtype=np.int32),
                    (pooled_height, pooled_width, 1),
                )
                patch_orderings.append(
                    pad_to_bounding_box(
                        reshaped_image,
                        offset_height=crop_y_offset,
                        offset_width=crop_x_offset,
                        target_height=self.tokens_per_image_height,
                        target_width=self.tokens_per_image_width,
                        value=-1,
                    )[:, :, 0]
                )

                # Extract the image crop
                crops.append(
                    image[crop_y_start : crop_y_start + self.crop_size, crop_x_start : crop_x_start + self.crop_size]
                )

                # Extract the corresponding mask for the crop
                cropped_masks.append(
                    image_mask[
                        crop_y_start : crop_y_start + self.crop_size, crop_x_start : crop_x_start + self.crop_size
                    ]
                )
                # Update the patch index for ordering (there are several patches in a crop)
                patch_index += pooled_height * pooled_width
        # Stack the crops, patch orderings, and masks into arrays
        crops = np.stack(crops)
        patch_orderings = np.stack(patch_orderings)
        cropped_masks = np.stack(cropped_masks)
        # rearrange patches
        leading_crops_dim, channels = crops.shape[0], crops.shape[-1]
        crops = crops.reshape(
            leading_crops_dim,
            self.patches_per_image_height,
            self.image_patch_size,
            self.patches_per_image_width,
            self.image_patch_size,
            channels,
        )
        crops = crops.transpose(0, 1, 3, 2, 4, 5)
        crops = crops.reshape(
            leading_crops_dim,
            self.patches_per_image_width * self.patches_per_image_height,
            self.image_patch_size * self.image_patch_size * channels,
        )
        leading_mask_dim = cropped_masks.shape[0]
        cropped_masks = cropped_masks.reshape(
            leading_mask_dim,
            self.patches_per_image_height,
            self.image_patch_size,
            self.patches_per_image_width,
            self.image_patch_size,
        )
        cropped_masks = cropped_masks.transpose(0, 1, 3, 2, 4)
        cropped_masks = cropped_masks.reshape(
            leading_mask_dim,
            self.patches_per_image_width * self.patches_per_image_height,
            self.image_patch_size * self.image_patch_size,
        )

        cropped_masks = cropped_masks.astype(np.float32).mean(axis=-1)
        cropped_masks = np.pad(cropped_masks, [[0, 1], [0, 0]], constant_values=-1)
        patch_orderings = np.reshape(patch_orderings, [-1])
        return crops, patch_orderings, cropped_masks

    def transpose_patch_orderings(self, crop_grid, patch_orderings):
        patch_ordering_left_right = np.reshape(
            patch_orderings, [crop_grid[0], crop_grid[1], self.tokens_per_image_height, self.tokens_per_image_width]
        )
        patch_ordering_left_right = np.transpose(patch_ordering_left_right, [0, 2, 1, 3])
        patch_ordering_left_right = np.reshape(patch_ordering_left_right, [-1])

        # The transpose will mess up which patches are masked, project the
        # new order into sparse structure of `patch_ordering` to fix this
        patch_orderings[patch_orderings >= 0] = patch_ordering_left_right[patch_ordering_left_right >= 0]
        return patch_orderings

    def _prepare_crop_grids(self, data):
        """
        Prepares crop_grids by stacking them into a batch dimension.
        """
        crop_grids = data["crop_grids"]  # List of arrays with shape (2,)
        data["crop_grids"] = np.stack(crop_grids, axis=0)  # Shape: (batch_size, 2)

    def _pad_patch_orderings(self, data):
        """
        Pads patch_orderings to have the same length across the batch.
        """
        patch_orderings = data["patch_orderings"]  # List of arrays with shape (length_i,)
        batch_size = len(patch_orderings)
        max_length = max(ordering.shape[0] for ordering in patch_orderings)

        # use a fill value that doesn't interfere with valid data (e.g., -2)
        fill_value = -2
        batched_patch_orderings = np.full(
            (batch_size, max_length), fill_value=fill_value, dtype=patch_orderings[0].dtype
        )

        patch_orderings_mask = np.zeros((batch_size, max_length), dtype=bool)

        for idx, ordering in enumerate(patch_orderings):
            length = ordering.shape[0]
            batched_patch_orderings[idx, :length] = ordering
            patch_orderings_mask[idx, :length] = True

        # Update the data dictionary
        data["patch_orderings"] = batched_patch_orderings  # Shape: (batch_size, max_length)

    def _pad_for_batching(
        self,
        data: Dict,
    ):
        """
        Pads crops obtained with the largest amount of crops in the batch. Will penalize queries with high
        number of crops. Pads as well the patch orderings and so on.
        """
        crops = data["pixel_values"]
        max_num_crops = max(image.shape[0] for image in crops)
        batch_size = len(crops)
        crop_shape = crops[0].shape[1:]

        batched_crops = np.zeros((batch_size, max_num_crops) + crop_shape, dtype=crops[0].dtype)
        crop_masks = np.zeros((batch_size, max_num_crops), dtype=np.bool_)
        for idx, image in enumerate(crops):
            num_crops = image.shape[0]
            batched_crops[idx, :num_crops, ...] = image
            crop_masks[idx, :num_crops] = True

        data["pixel_values"] = batched_crops

        # pad image_masks with -1
        image_masks = data["image_masks"]
        mask_shape = image_masks[0].shape[1:]
        batched_image_masks = np.full(
            (batch_size, max_num_crops) + mask_shape, fill_value=-1, dtype=image_masks[0].dtype
        )
        for idx, mask in enumerate(image_masks):
            num_crops = mask.shape[0]
            batched_image_masks[idx, :num_crops, ...] = mask

        data["image_masks"] = batched_image_masks
        self._pad_patch_orderings(data)

        self._prepare_crop_grids(data)
        return data

    def preprocess(
        self,
        images: ImageInput,
        do_resize: bool = None,
        size: Dict[str, int] = None,
        resample: PILImageResampling = None,
        do_pad: Optional[bool] = None,
        do_split_into_crops: Optional[bool] = None,
        padding_value: Optional[float] = None,
        padding_mode: Optional[str] = None,
        do_rescale: bool = None,
        rescale_factor: float = None,
        do_normalize: bool = None,
        image_mean: Optional[Union[float, List[float]]] = OPENAI_CLIP_MEAN,
        image_std: Optional[Union[float, List[float]]] = OPENAI_CLIP_STD,
        do_convert_rgb: bool = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        data_format: Optional[ChannelDimension] = ChannelDimension.FIRST,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
        **kwargs,
    ) -> BatchFeature:
        """
        Preprocess images for the Molmo model.

        Args:
            images (ImageInput): Image or batch of images to preprocess.
            image_patch_token_id (int): Token ID for image patches.
            image_col_token_id (int): Token ID for image columns.
            image_start_token_id (int): Token ID for the start of an image.
            image_end_token_id (int): Token ID for the end of an image.

        Returns:
            BatchFeature: A dictionary containing processed image patches, tokens, indices, and masks.
        """
        do_resize = do_resize if do_resize is not None else self.do_resize
        size = size if size is not None else self.size
        size = get_size_dict(size, param_name="size", default_to_square=False)
        resample = resample if resample is not None else self.resample
        do_pad = do_pad if do_pad is not None else self.do_pad
        do_split_into_crops = do_split_into_crops if do_split_into_crops is not None else self.do_split_into_crops
        padding_value = padding_value if padding_value is not None else self.padding_value
        padding_mode = padding_mode if padding_mode is not None else self.padding_mode
        do_rescale = do_rescale if do_rescale is not None else self.do_rescale
        rescale_factor = rescale_factor if rescale_factor is not None else self.rescale_factor
        do_normalize = do_normalize if do_normalize is not None else self.do_normalize
        image_mean = image_mean if image_mean is not None else self.image_mean
        image_std = image_std if image_std is not None else self.image_std
        do_convert_rgb = do_convert_rgb if do_convert_rgb is not None else self.do_convert_rgb

        validate_kwargs(captured_kwargs=kwargs.keys(), valid_processor_keys=self._valid_processor_keys)

        images = make_batched_images(images)

        if not valid_images(images):
            raise ValueError(
                "Invalid image type. Must be of type PIL.Image.Image, numpy.ndarray, "
                "torch.Tensor, tf.Tensor or jax.ndarray."
            )
        validate_preprocess_arguments(
            do_rescale=do_rescale,
            rescale_factor=rescale_factor,
            do_normalize=do_normalize,
            image_mean=image_mean,
            image_std=image_std,
            do_resize=do_resize,
            size=size,
            resample=resample,
        )

        if do_convert_rgb:
            images = [convert_to_rgb(image) for image in images]

        # All transformations expect numpy arrays.
        images = [to_numpy_array(image) for image in images]

        if is_scaled_image(images[0]) and do_rescale:
            logger.warning_once(
                "It looks like you are trying to rescale already rescaled images. If the input"
                " images have pixel values between 0 and 1, set `do_rescale=False` to avoid rescaling them again."
            )

        if input_data_format is None:
            # We assume that all images have the same channel dimension format.
            input_data_format = infer_channel_dimension_format(images[0])

        all_images = []
        all_crop_grids = []
        all_cropped_masks = []
        all_patch_orderings = []
        for image in images:
            # 1. First, for a given image, figure out the best crop grid for the input image.
            # We need to keep track of a few values here.
            crop_grid = self.find_best_crop_grid_for_image_size(image)
            # 2. Then, resize and pad, figure out number of crops (large ones) and patches (small ones)
            if do_resize:
                # we resize both the global image to the wanted size, as well as the crops.
                global_image_size = get_resize_output_image_size(image, size)
                global_image = self.resize(
                    image=image, size=global_image_size, resample=resample, input_data_format=input_data_format
                )
                new_crop_size = {}
                new_crop_size["height"] = crop_grid[0] * self.crop_window_size + self.total_margin_pixels
                new_crop_size["width"] = crop_grid[1] * self.crop_window_size + self.total_margin_pixels
                crop_output_size = get_resize_output_image_size(
                    image,
                    size=new_crop_size,
                )

                image = self.resize(
                    image=image, size=crop_output_size, resample=resample, input_data_format=input_data_format
                )
            # TODO do_pad and do_split_into_crops should not be optional. Removing them will break the processing.
            if do_pad:
                # 2.1 after padding, we also get the image mask
                image, image_mask = self.pad(
                    image=image, size=new_crop_size, input_data_format=input_data_format, constant_values=0
                )
                # 2.2 (from original code) the image mask padding is increased by 1 dim
                global_image, _ = self.pad(
                    image=global_image, size=size, input_data_format=input_data_format, constant_values=0
                )
            if do_rescale:
                image = self.rescale(image=image, scale=rescale_factor, input_data_format=input_data_format)
                global_image = self.rescale(
                    image=global_image, scale=rescale_factor, input_data_format=input_data_format
                )
            if do_normalize:
                image = normalize(image=image, mean=image_mean, std=image_std, input_data_format=input_data_format)
                global_image = normalize(
                    image=global_image, mean=image_mean, std=image_std, input_data_format=input_data_format
                )

            # 3. Then split the padded and rescaled image into crops. Don't touch the global image.
            if do_split_into_crops:
                crops, patch_orderings, cropped_masks = self.split_image_into_crops(
                    image=image, image_mask=image_mask, crop_grid=crop_grid, input_data_format=input_data_format
                )
                # 4. Reorder patches left-to-right instead of crop-by-crop.
                patch_orderings = self.transpose_patch_orderings(crop_grid, patch_orderings)
            global_image = self.reshape_into_patches(global_image, input_data_format=input_data_format)
            # 5. Concatenate patches and the global image
            crops = np.concatenate([np.expand_dims(global_image, 0), crops], 0)

            # 6. Global image goes first, so the order of patches in previous crops gets increased
            # by an amount corresponding to the number of tokens per image
            patch_orderings = np.where(patch_orderings >= 0, patch_orderings + self.tokens_per_image, -1)
            patch_orderings = np.concatenate([np.arange(0, self.tokens_per_image), patch_orderings], 0)
            # 7. Add an extra dim for the image mask padding

            all_images.append(crops)
            all_crop_grids.append(crop_grid)
            all_cropped_masks.append(cropped_masks)
            all_patch_orderings.append(patch_orderings)
        data = {
            "pixel_values": all_images,
            "crop_grids": all_crop_grids,
            "patch_orderings": all_patch_orderings,
            "image_masks": all_cropped_masks,
        }
        if do_pad:
            data = self._pad_for_batching(data)
        return BatchFeature(data=data, tensor_type=return_tensors)
