# Copyright 2022 The HuggingFace Team. All rights reserved.
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

import json
import os
import shutil
import tempfile
import unittest

import pytest

from transformers import CLIPTokenizer, CLIPTokenizerFast
from transformers.models.clip.tokenization_clip import VOCAB_FILES_NAMES
from transformers.testing_utils import require_vision
from transformers.utils import IMAGE_PROCESSOR_NAME, is_vision_available

from ...test_processing_common import ProcessorTesterMixin


if is_vision_available():
    from transformers import CLIPSegProcessor, ViTImageProcessor


@require_vision
class CLIPSegProcessorTest(ProcessorTesterMixin, unittest.TestCase):
    processor_class = CLIPSegProcessor

    def setUp(self):
        self.tmpdirname = tempfile.mkdtemp()

        vocab = ["l", "o", "w", "e", "r", "s", "t", "i", "d", "n", "lo", "l</w>", "w</w>", "r</w>", "t</w>", "low</w>", "er</w>", "lowest</w>", "newer</w>", "wider", "<unk>", "<|startoftext|>", "<|endoftext|>"]  # fmt: skip
        vocab_tokens = dict(zip(vocab, range(len(vocab))))
        merges = ["#version: 0.2", "l o", "lo w</w>", "e r</w>", ""]
        self.special_tokens_map = {"unk_token": "<unk>"}

        self.vocab_file = os.path.join(self.tmpdirname, VOCAB_FILES_NAMES["vocab_file"])
        self.merges_file = os.path.join(
            self.tmpdirname, VOCAB_FILES_NAMES["merges_file"]
        )
        with open(self.vocab_file, "w", encoding="utf-8") as fp:
            fp.write(json.dumps(vocab_tokens) + "\n")
        with open(self.merges_file, "w", encoding="utf-8") as fp:
            fp.write("\n".join(merges))

        image_processor_map = {
            "do_resize": True,
            "size": 20,
            "do_center_crop": True,
            "crop_size": 18,
            "do_normalize": True,
            "image_mean": [0.48145466, 0.4578275, 0.40821073],
            "image_std": [0.26862954, 0.26130258, 0.27577711],
        }
        self.image_processor_file = os.path.join(self.tmpdirname, IMAGE_PROCESSOR_NAME)
        with open(self.image_processor_file, "w", encoding="utf-8") as fp:
            json.dump(image_processor_map, fp)

    def get_tokenizer(self, **kwargs):
        return CLIPTokenizer.from_pretrained(self.tmpdirname, **kwargs)

    def get_rust_tokenizer(self, **kwargs):
        return CLIPTokenizerFast.from_pretrained(self.tmpdirname, **kwargs)

    def get_image_processor(self, **kwargs):
        return ViTImageProcessor.from_pretrained(self.tmpdirname, **kwargs)

    def tearDown(self):
        shutil.rmtree(self.tmpdirname)

    def test_save_load_pretrained_default(self):
        tokenizer_slow = self.get_tokenizer()
        tokenizer_fast = self.get_rust_tokenizer()
        image_processor = self.get_image_processor()

        processor_slow = CLIPSegProcessor(
            tokenizer=tokenizer_slow, image_processor=image_processor
        )
        processor_slow.save_pretrained(self.tmpdirname)
        processor_slow = CLIPSegProcessor.from_pretrained(
            self.tmpdirname, use_fast=False
        )

        processor_fast = CLIPSegProcessor(
            tokenizer=tokenizer_fast, image_processor=image_processor
        )
        processor_fast.save_pretrained(self.tmpdirname)
        processor_fast = CLIPSegProcessor.from_pretrained(self.tmpdirname)

        self.assertEqual(
            processor_slow.tokenizer.get_vocab(), tokenizer_slow.get_vocab()
        )
        self.assertEqual(
            processor_fast.tokenizer.get_vocab(), tokenizer_fast.get_vocab()
        )
        self.assertEqual(tokenizer_slow.get_vocab(), tokenizer_fast.get_vocab())
        self.assertIsInstance(processor_slow.tokenizer, CLIPTokenizer)
        self.assertIsInstance(processor_fast.tokenizer, CLIPTokenizerFast)

        self.assertEqual(
            processor_slow.image_processor.to_json_string(),
            image_processor.to_json_string(),
        )
        self.assertEqual(
            processor_fast.image_processor.to_json_string(),
            image_processor.to_json_string(),
        )
        self.assertIsInstance(processor_slow.image_processor, ViTImageProcessor)
        self.assertIsInstance(processor_fast.image_processor, ViTImageProcessor)

    def test_save_load_pretrained_additional_features(self):
        processor = CLIPSegProcessor(
            tokenizer=self.get_tokenizer(), image_processor=self.get_image_processor()
        )
        processor.save_pretrained(self.tmpdirname)

        tokenizer_add_kwargs = self.get_tokenizer(bos_token="(BOS)", eos_token="(EOS)")
        image_processor_add_kwargs = self.get_image_processor(
            do_normalize=False, padding_value=1.0
        )

        processor = CLIPSegProcessor.from_pretrained(
            self.tmpdirname,
            bos_token="(BOS)",
            eos_token="(EOS)",
            do_normalize=False,
            padding_value=1.0,
        )

        self.assertEqual(
            processor.tokenizer.get_vocab(), tokenizer_add_kwargs.get_vocab()
        )
        self.assertIsInstance(processor.tokenizer, CLIPTokenizerFast)

        self.assertEqual(
            processor.image_processor.to_json_string(),
            image_processor_add_kwargs.to_json_string(),
        )
        self.assertIsInstance(processor.image_processor, ViTImageProcessor)

    def test_image_processor(self):
        image_processor = self.get_image_processor()
        tokenizer = self.get_tokenizer()

        processor = CLIPSegProcessor(
            tokenizer=tokenizer, image_processor=image_processor
        )

        image_input = self.prepare_image_inputs()

        input_feat_extract = image_processor(image_input, return_tensors="np")
        input_processor = processor(images=image_input, return_tensors="np")

        for key in input_feat_extract.keys():
            self.assertAlmostEqual(
                input_feat_extract[key].sum(), input_processor[key].sum(), delta=1e-2
            )

    def test_tokenizer(self):
        image_processor = self.get_image_processor()
        tokenizer = self.get_tokenizer()

        processor = CLIPSegProcessor(
            tokenizer=tokenizer, image_processor=image_processor
        )

        input_str = "lower newer"

        encoded_processor = processor(text=input_str)

        encoded_tok = tokenizer(input_str)

        for key in encoded_tok.keys():
            self.assertListEqual(encoded_tok[key], encoded_processor[key])

    def test_processor_text(self):
        image_processor = self.get_image_processor()
        tokenizer = self.get_tokenizer()

        processor = CLIPSegProcessor(
            tokenizer=tokenizer, image_processor=image_processor
        )

        input_str = "lower newer"
        image_input = self.prepare_image_inputs()

        inputs = processor(text=input_str, images=image_input)

        self.assertListEqual(
            list(inputs.keys()), ["input_ids", "attention_mask", "pixel_values"]
        )

        # test if it raises when no input is passed
        with pytest.raises(ValueError):
            processor()

    def test_processor_visual_prompt(self):
        image_processor = self.get_image_processor()
        tokenizer = self.get_tokenizer()

        processor = CLIPSegProcessor(
            tokenizer=tokenizer, image_processor=image_processor
        )

        image_input = self.prepare_image_inputs()
        visual_prompt_input = self.prepare_image_inputs()

        inputs = processor(images=image_input, visual_prompt=visual_prompt_input)

        self.assertListEqual(
            list(inputs.keys()), ["pixel_values", "conditional_pixel_values"]
        )

        # test if it raises when no input is passed
        with pytest.raises(ValueError):
            processor()

    def test_tokenizer_decode(self):
        image_processor = self.get_image_processor()
        tokenizer = self.get_tokenizer()

        processor = CLIPSegProcessor(
            tokenizer=tokenizer, image_processor=image_processor
        )

        predicted_ids = [[1, 4, 5, 8, 1, 0, 8], [3, 4, 3, 1, 1, 8, 9]]

        decoded_processor = processor.batch_decode(predicted_ids)
        decoded_tok = tokenizer.batch_decode(predicted_ids)

        self.assertListEqual(decoded_tok, decoded_processor)
