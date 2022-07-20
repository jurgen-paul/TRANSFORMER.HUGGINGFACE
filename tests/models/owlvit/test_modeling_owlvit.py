# coding=utf-8
# Copyright 2022 The HuggingFace Inc. team. All rights reserved.
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
""" Testing suite for the PyTorch OwlViT model. """


import inspect
import os
import tempfile
import unittest
from typing import Dict, List, Tuple

import numpy as np

import requests
from transformers import OwlViTConfig, OwlViTTextConfig, OwlViTVisionConfig
from transformers.testing_utils import require_torch, require_vision, slow, torch_device
from transformers.utils import is_torch_available, is_vision_available

from ...test_configuration_common import ConfigTester
from ...test_modeling_common import (
    ModelTesterMixin,
    _config_zero_init,
    floats_tensor,
    ids_tensor,
    random_attention_mask,
)


if is_torch_available():
    import torch
    from torch import nn

    from transformers import OwlViTForObjectDetection, OwlViTModel, OwlViTTextModel, OwlViTVisionModel
    from transformers.models.owlvit.modeling_owlvit import OWLVIT_PRETRAINED_MODEL_ARCHIVE_LIST


if is_vision_available():
    from PIL import Image

    from transformers import OwlViTProcessor


class OwlViTVisionModelTester:
    def __init__(
        self,
        parent,
        batch_size=12,
        image_size=32,
        patch_size=2,
        num_channels=3,
        is_training=True,
        hidden_size=32,
        num_hidden_layers=5,
        num_attention_heads=4,
        intermediate_size=37,
        dropout=0.1,
        attention_dropout=0.1,
        initializer_range=0.02,
        scope=None,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.is_training = is_training
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.initializer_range = initializer_range
        self.scope = scope

        # in ViT, the seq length equals the number of patches + 1 (we add 1 for the [CLS] token)
        num_patches = (image_size // patch_size) ** 2
        self.seq_length = num_patches + 1

    def prepare_config_and_inputs(self):
        pixel_values = floats_tensor([self.batch_size, self.num_channels, self.image_size, self.image_size])
        config = self.get_config()

        return config, pixel_values

    def get_config(self):
        return OwlViTVisionConfig(
            image_size=self.image_size,
            patch_size=self.patch_size,
            num_channels=self.num_channels,
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            intermediate_size=self.intermediate_size,
            dropout=self.dropout,
            attention_dropout=self.attention_dropout,
            initializer_range=self.initializer_range,
        )

    def create_and_check_model(self, config, pixel_values):
        model = OwlViTVisionModel(config=config)
        model.to(torch_device)
        model.eval()

        pixel_values = pixel_values.to(torch.float32)

        with torch.no_grad():
            result = model(pixel_values)
        # expected sequence length = num_patches + 1 (we add 1 for the [CLS] token)
        num_patches = (self.image_size // self.patch_size) ** 2
        self.parent.assertEqual(result.last_hidden_state.shape, (self.batch_size, num_patches + 1, self.hidden_size))
        self.parent.assertEqual(result.pooler_output.shape, (self.batch_size, num_patches + 1, self.hidden_size))

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        config, pixel_values = config_and_inputs
        inputs_dict = {"pixel_values": pixel_values}
        return config, inputs_dict


@require_torch
class OwlViTVisionModelTest(ModelTesterMixin, unittest.TestCase):
    """
    Here we also overwrite some of the tests of test_modeling_common.py, as OWLVIT does not use input_ids, inputs_embeds,
    attention_mask and seq_length.
    """

    all_model_classes = (OwlViTVisionModel,) if is_torch_available() else ()
    fx_compatible = False
    test_pruning = False
    test_resize_embeddings = False
    test_head_masking = False

    def setUp(self):
        self.model_tester = OwlViTVisionModelTester(self)
        self.config_tester = ConfigTester(
            self, config_class=OwlViTVisionConfig, has_text_modality=False, hidden_size=37
        )

    def test_config(self):
        self.config_tester.run_common_tests()

    @unittest.skip(reason="OWLVIT does not use inputs_embeds")
    def test_inputs_embeds(self):
        pass

    def test_model_common_attributes(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config)
            self.assertIsInstance(model.get_input_embeddings(), (nn.Module))
            x = model.get_output_embeddings()
            self.assertTrue(x is None or isinstance(x, nn.Linear))

    def test_forward_signature(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config)
            signature = inspect.signature(model.forward)
            # signature.parameters is an OrderedDict => so arg_names order is deterministic
            arg_names = [*signature.parameters.keys()]

            expected_arg_names = ["pixel_values"]
            self.assertListEqual(arg_names[:1], expected_arg_names)

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    @unittest.skip(reason="OWL-ViT does not support training yet")
    def test_training(self):
        pass

    @unittest.skip(reason="OWL-ViT does not support training yet")
    def test_training_gradient_checkpointing(self):
        pass

    @unittest.skip(reason="OwlViTVisionModel has no base class and is not available in MODEL_MAPPING")
    def test_save_load_fast_init_from_base(self):
        pass

    @unittest.skip(reason="OwlViTVisionModel has no base class and is not available in MODEL_MAPPING")
    def test_save_load_fast_init_to_base(self):
        pass

    @slow
    def test_model_from_pretrained(self):
        for model_name in OWLVIT_PRETRAINED_MODEL_ARCHIVE_LIST[:1]:
            model = OwlViTVisionModel.from_pretrained(model_name)
            self.assertIsNotNone(model)


class OwlViTTextModelTester:
    def __init__(
        self,
        parent,
        batch_size=12,
        num_queries=4,
        seq_length=16,
        is_training=True,
        use_input_mask=True,
        use_labels=True,
        vocab_size=99,
        hidden_size=64,
        num_hidden_layers=12,
        num_attention_heads=4,
        intermediate_size=37,
        dropout=0.1,
        attention_dropout=0.1,
        max_position_embeddings=16,
        initializer_range=0.02,
        scope=None,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.num_queries = num_queries
        self.seq_length = seq_length
        self.is_training = is_training
        self.use_input_mask = use_input_mask
        self.use_labels = use_labels
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.scope = scope

    def prepare_config_and_inputs(self):
        input_ids = ids_tensor([self.batch_size * self.num_queries, self.seq_length], self.vocab_size)
        input_mask = None

        if self.use_input_mask:
            input_mask = random_attention_mask([self.batch_size * self.num_queries, self.seq_length])

        if input_mask is not None:
            num_text, seq_length = input_mask.shape

            rnd_start_indices = np.random.randint(1, seq_length - 1, size=(num_text,))
            for idx, start_index in enumerate(rnd_start_indices):
                input_mask[idx, :start_index] = 1
                input_mask[idx, start_index:] = 0

        config = self.get_config()

        return config, input_ids, input_mask

    def get_config(self):
        return OwlViTTextConfig(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            intermediate_size=self.intermediate_size,
            dropout=self.dropout,
            attention_dropout=self.attention_dropout,
            max_position_embeddings=self.max_position_embeddings,
            initializer_range=self.initializer_range,
        )

    def create_and_check_model(self, config, input_ids, input_mask):
        model = OwlViTTextModel(config=config)
        model.to(torch_device)
        model.eval()
        with torch.no_grad():
            result = model(input_ids=input_ids, attention_mask=input_mask)

        self.parent.assertEqual(
            result.last_hidden_state.shape, (self.batch_size * self.num_queries, self.seq_length, self.hidden_size)
        )
        self.parent.assertEqual(result.pooler_output.shape, (self.batch_size * self.num_queries, self.hidden_size))

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        config, input_ids, input_mask = config_and_inputs
        inputs_dict = {"input_ids": input_ids, "attention_mask": input_mask}
        return config, inputs_dict


@require_torch
class OwlViTTextModelTest(ModelTesterMixin, unittest.TestCase):

    all_model_classes = (OwlViTTextModel,) if is_torch_available() else ()
    fx_compatible = False
    test_pruning = False
    test_head_masking = False

    def setUp(self):
        self.model_tester = OwlViTTextModelTester(self)
        self.config_tester = ConfigTester(self, config_class=OwlViTTextConfig, hidden_size=37)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    @unittest.skip(reason="OWL-ViT does not support training yet")
    def test_training(self):
        pass

    @unittest.skip(reason="OWL-ViT does not support training yet")
    def test_training_gradient_checkpointing(self):
        pass

    @unittest.skip(reason="OWLVIT does not use inputs_embeds")
    def test_inputs_embeds(self):
        pass

    @unittest.skip(reason="OwlViTTextModel has no base class and is not available in MODEL_MAPPING")
    def test_save_load_fast_init_from_base(self):
        pass

    @unittest.skip(reason="OwlViTTextModel has no base class and is not available in MODEL_MAPPING")
    def test_save_load_fast_init_to_base(self):
        pass

    @slow
    def test_model_from_pretrained(self):
        for model_name in OWLVIT_PRETRAINED_MODEL_ARCHIVE_LIST[:1]:
            model = OwlViTTextModel.from_pretrained(model_name)
            self.assertIsNotNone(model)


class OwlViTModelTester:
    def __init__(self, parent, is_training=True):
        self.parent = parent
        self.text_model_tester = OwlViTTextModelTester(parent)
        self.vision_model_tester = OwlViTVisionModelTester(parent)
        self.is_training = is_training
        self.text_config = self.text_model_tester.get_config().to_dict()
        self.vision_config = self.vision_model_tester.get_config().to_dict()

    def prepare_config_and_inputs(self):
        text_config, input_ids, attention_mask = self.text_model_tester.prepare_config_and_inputs()
        vision_config, pixel_values = self.vision_model_tester.prepare_config_and_inputs()
        config = self.get_config()
        return config, input_ids, attention_mask, pixel_values

    def get_config(self):
        return OwlViTConfig.from_text_vision_configs(self.text_config, self.vision_config, projection_dim=64)

    def create_and_check_model(self, config, input_ids, attention_mask, pixel_values):
        model = OwlViTModel(config).to(torch_device).eval()

        with torch.no_grad():
            result = model(
                input_ids=input_ids,
                pixel_values=pixel_values,
                attention_mask=attention_mask,
            )

        image_logits_size = (
            self.vision_model_tester.batch_size,
            self.text_model_tester.batch_size * self.text_model_tester.num_queries,
        )
        text_logits_size = (
            self.text_model_tester.batch_size * self.text_model_tester.num_queries,
            self.vision_model_tester.batch_size,
        )
        self.parent.assertEqual(result.logits_per_image.shape, image_logits_size)
        self.parent.assertEqual(result.logits_per_text.shape, text_logits_size)

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        config, input_ids, attention_mask, pixel_values = config_and_inputs
        inputs_dict = {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "return_loss": False,
        }
        return config, inputs_dict


@require_torch
class OwlViTModelTest(ModelTesterMixin, unittest.TestCase):
    all_model_classes = (OwlViTModel,) if is_torch_available() else ()
    fx_compatible = False
    test_head_masking = False
    test_pruning = False
    test_resize_embeddings = False
    test_attention_outputs = False

    def setUp(self):
        self.model_tester = OwlViTModelTester(self)

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    @unittest.skip(reason="Hidden_states is tested in individual model tests")
    def test_hidden_states_output(self):
        pass

    @unittest.skip(reason="Inputs_embeds is tested in individual model tests")
    def test_inputs_embeds(self):
        pass

    @unittest.skip(reason="Retain_grad is tested in individual model tests")
    def test_retain_grad_hidden_states_attentions(self):
        pass

    @unittest.skip(reason="OwlViTModel does not have input/output embeddings")
    def test_model_common_attributes(self):
        pass

    # override as the `logit_scale` parameter initilization is different for OWLVIT
    def test_initialization(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        configs_no_init = _config_zero_init(config)
        for model_class in self.all_model_classes:
            model = model_class(config=configs_no_init)
            for name, param in model.named_parameters():
                if param.requires_grad:
                    # check if `logit_scale` is initilized as per the original implementation
                    if name == "logit_scale":
                        self.assertAlmostEqual(
                            param.data.item(),
                            np.log(1 / 0.07),
                            delta=1e-3,
                            msg=f"Parameter {name} of model {model_class} seems not properly initialized",
                        )
                    else:
                        self.assertIn(
                            ((param.data.mean() * 1e9).round() / 1e9).item(),
                            [0.0, 1.0],
                            msg=f"Parameter {name} of model {model_class} seems not properly initialized",
                        )

    def _create_and_check_torchscript(self, config, inputs_dict):
        if not self.test_torchscript:
            return

        configs_no_init = _config_zero_init(config)  # To be sure we have no Nan
        configs_no_init.torchscript = True
        configs_no_init.return_dict = False
        for model_class in self.all_model_classes:
            model = model_class(config=configs_no_init)
            model.to(torch_device)
            model.eval()

            try:
                input_ids = inputs_dict["input_ids"]
                pixel_values = inputs_dict["pixel_values"]  # OWLVIT needs pixel_values
                traced_model = torch.jit.trace(model, (input_ids, pixel_values))
            except RuntimeError:
                self.fail("Couldn't trace module.")

            with tempfile.TemporaryDirectory() as tmp_dir_name:
                pt_file_name = os.path.join(tmp_dir_name, "traced_model.pt")

                try:
                    torch.jit.save(traced_model, pt_file_name)
                except Exception:
                    self.fail("Couldn't save module.")

                try:
                    loaded_model = torch.jit.load(pt_file_name)
                except Exception:
                    self.fail("Couldn't load module.")

            model.to(torch_device)
            model.eval()

            loaded_model.to(torch_device)
            loaded_model.eval()

            model_state_dict = model.state_dict()
            loaded_model_state_dict = loaded_model.state_dict()

            self.assertEqual(set(model_state_dict.keys()), set(loaded_model_state_dict.keys()))

            models_equal = True
            for layer_name, p1 in model_state_dict.items():
                p2 = loaded_model_state_dict[layer_name]
                if p1.data.ne(p2.data).sum() > 0:
                    models_equal = False

            self.assertTrue(models_equal)

    def test_load_vision_text_config(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        # Save OwlViTConfig and check if we can load OwlViTVisionConfig from it
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            config.save_pretrained(tmp_dir_name)
            vision_config = OwlViTVisionConfig.from_pretrained(tmp_dir_name)
            self.assertDictEqual(config.vision_config.to_dict(), vision_config.to_dict())

        # Save OwlViTConfig and check if we can load OwlViTTextConfig from it
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            config.save_pretrained(tmp_dir_name)
            text_config = OwlViTTextConfig.from_pretrained(tmp_dir_name)
            self.assertDictEqual(config.text_config.to_dict(), text_config.to_dict())

    @slow
    def test_model_from_pretrained(self):
        for model_name in OWLVIT_PRETRAINED_MODEL_ARCHIVE_LIST[:1]:
            model = OwlViTModel.from_pretrained(model_name)
            self.assertIsNotNone(model)


class OwlViTForObjectDetectionTester:
    def __init__(self, parent, is_training=True):
        self.parent = parent
        self.text_model_tester = OwlViTTextModelTester(parent)
        self.vision_model_tester = OwlViTVisionModelTester(parent)
        self.is_training = is_training
        self.text_config = self.text_model_tester.get_config().to_dict()
        self.vision_config = self.vision_model_tester.get_config().to_dict()

    def prepare_config_and_inputs(self):
        text_config, input_ids, attention_mask = self.text_model_tester.prepare_config_and_inputs()
        vision_config, pixel_values = self.vision_model_tester.prepare_config_and_inputs()
        config = self.get_config()
        return config, pixel_values, input_ids, attention_mask

    def get_config(self):
        return OwlViTConfig.from_text_vision_configs(self.text_config, self.vision_config, projection_dim=64)

    def create_and_check_model(self, config, pixel_values, input_ids, attention_mask):
        model = OwlViTForObjectDetection(config).to(torch_device).eval()
        with torch.no_grad():
            result = model(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True,
            )

        pred_boxes_size = (
            self.vision_model_tester.batch_size,
            (self.vision_model_tester.image_size // self.vision_model_tester.patch_size) ** 2,
            4,
        )
        pred_logits_size = (
            self.vision_model_tester.batch_size,
            (self.vision_model_tester.image_size // self.vision_model_tester.patch_size) ** 2,
            4,
        )
        pred_class_embeds_size = (
            self.vision_model_tester.batch_size,
            (self.vision_model_tester.image_size // self.vision_model_tester.patch_size) ** 2,
            self.text_model_tester.hidden_size,
        )
        self.parent.assertEqual(result.pred_boxes.shape, pred_boxes_size)
        self.parent.assertEqual(result.logits.shape, pred_logits_size)
        self.parent.assertEqual(result.class_embeds.shape, pred_class_embeds_size)

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        config, pixel_values, input_ids, attention_mask = config_and_inputs
        inputs_dict = {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        return config, inputs_dict


@require_torch
class OwlViTForObjectDetectionTest(ModelTesterMixin, unittest.TestCase):
    all_model_classes = (OwlViTForObjectDetection,) if is_torch_available() else ()
    fx_compatible = False
    test_head_masking = False
    test_pruning = False
    test_resize_embeddings = False
    test_attention_outputs = False

    def setUp(self):
        self.model_tester = OwlViTForObjectDetectionTester(self)

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    @unittest.skip(reason="Hidden_states is tested in individual model tests")
    def test_hidden_states_output(self):
        pass

    @unittest.skip(reason="Inputs_embeds is tested in individual model tests")
    def test_inputs_embeds(self):
        pass

    @unittest.skip(reason="Retain_grad is tested in individual model tests")
    def test_retain_grad_hidden_states_attentions(self):
        pass

    @unittest.skip(reason="OwlViTModel does not have input/output embeddings")
    def test_model_common_attributes(self):
        pass

    @unittest.skip(reason="Test_initialization is tested in individual model tests")
    def test_initialization(self):
        pass

    @unittest.skip(reason="Test_forward_signature is tested in individual model tests")
    def test_forward_signature(self):
        pass

    @unittest.skip(reason="Test_save_load_fast_init_from_base is tested in individual model tests")
    def test_save_load_fast_init_from_base(self):
        pass

    @unittest.skip(reason="OWL-ViT does not support training yet")
    def test_training(self):
        pass

    @unittest.skip(reason="OWL-ViT does not support training yet")
    def test_training_gradient_checkpointing(self):
        pass

    def _create_and_check_torchscript(self, config, inputs_dict):
        if not self.test_torchscript:
            return

        configs_no_init = _config_zero_init(config)  # To be sure we have no Nan
        configs_no_init.torchscript = True
        configs_no_init.return_dict = False
        for model_class in self.all_model_classes:
            model = model_class(config=configs_no_init)
            model.to(torch_device)
            model.eval()

            try:
                input_ids = inputs_dict["input_ids"]
                pixel_values = inputs_dict["pixel_values"]  # OWLVIT needs pixel_values
                traced_model = torch.jit.trace(model, (input_ids, pixel_values))
            except RuntimeError:
                self.fail("Couldn't trace module.")

            with tempfile.TemporaryDirectory() as tmp_dir_name:
                pt_file_name = os.path.join(tmp_dir_name, "traced_model.pt")

                try:
                    torch.jit.save(traced_model, pt_file_name)
                except Exception:
                    self.fail("Couldn't save module.")

                try:
                    loaded_model = torch.jit.load(pt_file_name)
                except Exception:
                    self.fail("Couldn't load module.")

            model.to(torch_device)
            model.eval()

            loaded_model.to(torch_device)
            loaded_model.eval()

            model_state_dict = model.state_dict()
            loaded_model_state_dict = loaded_model.state_dict()

            self.assertEqual(set(model_state_dict.keys()), set(loaded_model_state_dict.keys()))

            models_equal = True
            for layer_name, p1 in model_state_dict.items():
                p2 = loaded_model_state_dict[layer_name]
                if p1.data.ne(p2.data).sum() > 0:
                    models_equal = False

            self.assertTrue(models_equal)

    def test_model_outputs_equivalence(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        def set_nan_tensor_to_zero(t):
            t[t != t] = 0
            return t

        def check_equivalence(model, tuple_inputs, dict_inputs, additional_kwargs={}):
            with torch.no_grad():
                tuple_output = model(**tuple_inputs, return_dict=False, **additional_kwargs)
                dict_output = model(**dict_inputs, return_dict=True, **additional_kwargs).to_tuple()

                def recursive_check(tuple_object, dict_object):
                    if isinstance(tuple_object, (List, Tuple)):
                        for tuple_iterable_value, dict_iterable_value in zip(tuple_object, dict_object):
                            recursive_check(tuple_iterable_value, dict_iterable_value)
                    elif isinstance(tuple_object, Dict):
                        for tuple_iterable_value, dict_iterable_value in zip(
                            tuple_object.values(), dict_object.values()
                        ):
                            recursive_check(tuple_iterable_value, dict_iterable_value)
                    elif tuple_object is None:
                        return
                    else:
                        self.assertTrue(
                            torch.allclose(
                                set_nan_tensor_to_zero(tuple_object), set_nan_tensor_to_zero(dict_object), atol=1e-5
                            ),
                            msg=(
                                "Tuple and dict output are not equal. Difference:"
                                f" {torch.max(torch.abs(tuple_object - dict_object))}. Tuple has `nan`:"
                                f" {torch.isnan(tuple_object).any()} and `inf`: {torch.isinf(tuple_object)}. Dict has"
                                f" `nan`: {torch.isnan(dict_object).any()} and `inf`: {torch.isinf(dict_object)}."
                            ),
                        )

                recursive_check(tuple_output, dict_output)

        for model_class in self.all_model_classes:
            model = model_class(config)
            model.to(torch_device)
            model.eval()

            tuple_inputs = self._prepare_for_class(inputs_dict, model_class)
            dict_inputs = self._prepare_for_class(inputs_dict, model_class)
            check_equivalence(model, tuple_inputs, dict_inputs)

    @slow
    def test_model_from_pretrained(self):
        for model_name in OWLVIT_PRETRAINED_MODEL_ARCHIVE_LIST[:1]:
            model = OwlViTForObjectDetection.from_pretrained(model_name)
            self.assertIsNotNone(model)


# We will verify our results on an image of cute cats
def prepare_img():
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    im = Image.open(requests.get(url, stream=True).raw)
    return im


@require_vision
@require_torch
class OwlViTModelIntegrationTest(unittest.TestCase):
    @slow
    def test_inference(self):
        model_name = "adirik/owlvit-base-patch32"
        model = OwlViTModel.from_pretrained(model_name).to(torch_device)
        processor = OwlViTProcessor.from_pretrained(model_name)

        image = prepare_img()
        inputs = processor(
            text=[["a photo of a cat", "a photo of a dog"]],
            images=image,
            max_length=16,
            padding="max_length",
            return_tensors="pt",
        ).to(torch_device)

        # forward pass
        with torch.no_grad():
            outputs = model(**inputs)

        # verify the logits
        self.assertEqual(
            outputs.logits_per_image.shape,
            torch.Size(
                (
                    inputs.pixel_values.shape[0],
                    inputs.input_ids.shape[0] * inputs.input_ids.shape[1] * inputs.pixel_values.shape[0],
                )
            ),
        )
        self.assertEqual(
            outputs.logits_per_text.shape,
            torch.Size(
                (
                    inputs.input_ids.shape[0] * inputs.input_ids.shape[1] * inputs.pixel_values.shape[0],
                    inputs.pixel_values.shape[0],
                )
            ),
        )

        expected_logits = torch.tensor([[1.0115, 0.9982]], device=torch_device)

        self.assertTrue(torch.allclose(outputs.logits_per_image, expected_logits, atol=1e-3))

    @slow
    def test_inference_object_detection(self):
        model_name = "adirik/owlvit-base-patch32"
        model = OwlViTForObjectDetection.from_pretrained(model_name).to(torch_device)

        processor = OwlViTProcessor.from_pretrained(model_name)

        image = prepare_img()
        inputs = processor(
            text=[["a photo of a cat", "a photo of a dog"]],
            images=image,
            max_length=16,
            padding="max_length",
            return_tensors="pt",
        ).to(torch_device)

        with torch.no_grad():
            outputs = model(**inputs)

        num_queries = int((model.config.vision_config.image_size / model.config.vision_config.patch_size) ** 2)
        self.assertEqual(outputs.pred_boxes.shape, torch.Size((1, num_queries, 4)))
        expected_slice_boxes = torch.tensor(
            [[0.0143, 0.0236, 0.0285], [0.0649, 0.0247, 0.0437], [0.0601, 0.0446, 0.0699]]
        ).to(torch_device)
        self.assertTrue(torch.allclose(outputs.pred_boxes[0, :3, :3], expected_slice_boxes, atol=1e-4))
