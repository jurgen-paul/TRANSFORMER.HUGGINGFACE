# coding=utf-8
# Copyright 2025 HuggingFace Inc.
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

import unittest
import numpy as np

from transformers.testing_utils import require_torch, require_vision
from transformers.utils import is_torch_available, is_vision_available

from ...test_image_processing_common import ImageProcessingTestMixin, prepare_image_inputs



if is_torch_available():
    import torch

if is_vision_available():
    from PIL import Image

    from transformers import DeepseekVLImageProcessor

class DeepseekVLImageProcessingTester:
    def __init__(
        self,
        parent,
        batch_size=7,
        num_channels=3,
        image_size=384,
        min_resolution=30,
        max_resolution=200,
        do_resize=True,
        size=None,
        do_normalize=True,
        image_mean=[1.0, 1.0, 1.0],
        image_std=[1.0, 1.0, 1.0],
        do_convert_rgb=True,
    ):
        size = size if size is not None else {"height": 384, "width": 384}
        self.parent = parent
        self.batch_size = batch_size
        self.num_channels = num_channels
        self.image_size = image_size
        self.min_resolution = min_resolution
        self.max_resolution = max_resolution
        self.do_resize = do_resize
        self.size = size
        self.do_normalize = do_normalize
        self.image_mean = image_mean
        self.image_std = image_std
        self.do_convert_rgb = do_convert_rgb

    def prepare_image_processor_dict(self):
        return {
            "do_resize": self.do_resize,
            "size": self.size,
            "do_normalize": self.do_normalize,
            "image_mean": self.image_mean,
            "image_std": self.image_std,
            "do_convert_rgb": self.do_convert_rgb,
        }

    # Copied from tests.models.clip.test_image_processing_clip.CLIPImageProcessingTester.prepare_image_inputs
    def prepare_image_inputs(self, equal_resolution=False, numpify=False, torchify=False):
        return prepare_image_inputs(
            batch_size=self.batch_size,
            num_channels=self.num_channels,
            min_resolution=self.min_resolution,
            max_resolution=self.max_resolution,
            equal_resolution=equal_resolution,
            numpify=numpify,
            torchify=torchify,
        )

@require_torch
@require_vision
class DeepseekVLImageProcessingTest(unittest.TestCase, ImageProcessingTestMixin):
    image_processing_class = DeepseekVLImageProcessor if is_vision_available() else None

    # Copied from tests.models.clip.test_image_processing_clip.CLIPImageProcessingTest.setUp with CLIP->Janus
    def setUp(self):
        super().setUp()
        self.image_processor_tester = DeepseekVLImageProcessingTester(self)

    @property
    # Copied from tests.models.clip.test_image_processing_clip.CLIPImageProcessingTest.image_processor_dict
    def image_processor_dict(self):
        return self.image_processor_tester.prepare_image_processor_dict()

    @property
    # Copied from tests.models.clip.test_image_processing_clip.CLIPImageProcessingTest.image_processor_dict
    def image_processor_dict(self):
        return self.image_processor_tester.prepare_image_processor_dict()
    
    def test_image_processor_properties(self):
        image_processing = self.image_processing_class(**self.image_processor_dict)
        self.assertTrue(hasattr(image_processing, "do_resize"))
        self.assertTrue(hasattr(image_processing, "size"))
        self.assertTrue(hasattr(image_processing, "do_normalize"))
        self.assertTrue(hasattr(image_processing, "image_mean"))
        self.assertTrue(hasattr(image_processing, "image_std"))
        self.assertTrue(hasattr(image_processing, "do_convert_rgb"))
    
    def test_image_processor_from_dict_with_kwargs(self):
        image_processor = self.image_processing_class.from_dict(self.image_processor_dict)
        self.assertEqual(image_processor.size, {"height": 384, "width": 384})
        self.assertEqual(image_processor.image_mean, [1.0, 1.0, 1.0])

        image_processor = self.image_processing_class.from_dict(
            self.image_processor_dict, size = 42, image_mean = [1.0, 2.0, 1.0]
        )
        self.assertEqual(image_processor.size, {"height": 42, "width": 42})
        self.assertEqual(image_processor.image_mean, [1.0, 2.0, 1.0])    
    
    def test_call_pil(self):
        image_processing = self.image_processing_class(**self.image_processor_dict)
        image_inputs = self.image_processor_tester.prepare_image_inputs(equal_resolution=True)
        for image in image_inputs:
            self.assertIsInstance(image, Image.Image)
        
        # Test non batched input
        encoded_images = image_processing(image_inputs[0], return_tensors="pt").pixel_values
        expected_output_image_shape = (1, 3, 384, 384)
        self.assertEqual(tuple(encoded_images.shape), expected_output_image_shape)
        
        # Test batched input
        encoded_images = image_processing(image_inputs, return_tensors="pt").pixel_values
        expected_output_image_shape = (7, 3, 384, 384)
        self.assertEqual(tuple(encoded_images.shape), expected_output_image_shape)
    
    def test_call_numpy(self):
        image_processing = self.image_processing_class(**self.image_processor_dict)
        image_inputs = self.image_processor_tester.prepare_image_inputs(equal_resolution=True, numpify=True)
        for image in image_inputs:
            self.assertIsInstance(image, np.ndarray)
        
        encoded_images = image_processing(image_inputs[0], return_tensors="pt").pixel_values
        expected_output_image_shape = (1, 3, 384, 384)
        self.assertEqual(tuple(encoded_images.shape), expected_output_image_shape)

        encoded_images = image_processing(image_inputs, return_tensors="pt").pixel_values
        expected_output_image_shape = (7, 3, 384, 384)
        self.assertEqual(tuple(encoded_images.shape), expected_output_image_shape)

    
    def test_call_pytorch(self):
        image_processing = self.image_processing_class(**self.image_processor_dict)
        image_inputs = self.image_processor_tester.prepare_image_inputs(equal_resolution=True, torchify=True)
        for image in image_inputs:
            self.assertIsInstance(image, torch.Tensor)
        
        encoded_images = image_processing(image_inputs[0], return_tensors="pt").pixel_values
        expected_output_image_shape = (1, 3, 384, 384)
        self.assertEqual(tuple(encoded_images.shape), expected_output_image_shape)

        encoded_images = image_processing(image_inputs, return_tensors="pt").pixel_values
        expected_output_image_shape = (7, 3, 384, 384)
        self.assertEqual(tuple(encoded_images.shape), expected_output_image_shape)

    def test_nested_input(self):
        image_processing = self.image_processing_class(**self.image_processor_dict)
        image_inputs = self.image_processor_tester.prepare_image_inputs(equal_resolution=True)

        # Test batched as a list of images
        encoded_images = image_processing(image_inputs, return_tensors="pt").pixel_values
        expected_output_image_shape = (7, 3, 384, 384)
        self.assertEqual(tuple(encoded_images.shape), expected_output_image_shape)

        # Test batched as a nested list of images, where each sublist is one batch.
        image_inputs_nested = [image_inputs[:3], image_inputs[3:]]
        encoded_images_nested = image_processing(image_inputs_nested, return_tensors="pt").pixel_values
        expected_output_image_shape = (7, 3, 384, 384)
        self.assertEqual(tuple(encoded_images_nested.shape), expected_output_image_shape)

        # Image processor should return same pixel values, independently of input format.
        self.assertTrue((encoded_images_nested == encoded_images).all())