# coding=utf-8
# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

import gc
import unittest

import requests

from transformers import (
    Phi4MultimodalConfig,
    Phi4MultimodalVisionConfig,
    Phi4MultimodalAudioConfig,
    Phi4MultimodalForCausalLM,
    AutoProcessor,
    AutoModelForCausalLM,
    GenerationConfig,
    is_torch_available,
    is_vision_available,
)
from transformers.testing_utils import (
    require_torch,
    slow,
    torch_device,
)

from ...generation.test_utils import GenerationTesterMixin
from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, floats_tensor, ids_tensor


if is_torch_available():
    import torch


if is_vision_available():
    from PIL import Image


import tempfile
import soundfile

class Phi4MultimodalModelTester:
    def __init__(
        self,
        parent,
        seq_length=7,
        num_hidden_layers=2,
        vocab_size=99,
        hidden_size=32,
        intermediate_size=64,
        num_attention_heads=8,
        num_key_value_heads=4,
        bos_token_id=199999,
        eos_token_id=199999,
        pad_token_id=199999,
        audio_config=Phi4MultimodalAudioConfig(
            num_blocks=2,
            hidden_size=32,
            num_attention_heads=8,
            intermediate_size=48,
        ),
        vision_config=Phi4MultimodalVisionConfig(
            num_hidden_layers=2,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=8,
        ),
    ):
        self.parent = parent
        self.num_hidden_layers = num_hidden_layers
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.bos_token_id = bos_token_id
        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id
        self.audio_config = audio_config
        self.vision_config = vision_config

        self.batch_size = 10
        self.num_channels = 3
        self.image_size = 358
        self.num_image_tokens = 128
        self.seq_length = seq_length + self.num_image_tokens


    def get_config(self):
        return Phi4MultimodalConfig(
            num_hidden_layers=self.num_hidden_layers,
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            intermediate_size=self.intermediate_size,
            num_attention_heads=self.num_attention_heads,
            num_key_value_heads=self.num_key_value_heads,
            bos_token_id=self.bos_token_id,
            eos_token_id=self.eos_token_id,
            pad_token_id=self.pad_token_id,
            vision_config=self.vision_config,
            audio_config=self.audio_config,
        )

    def prepare_config_and_inputs(self):
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)
        attention_mask = torch.tril(torch.ones_like(input_ids))
        config = self.get_config()

        return config, input_ids, attention_mask

    def prepare_config_and_inputs_for_common(self):
        config, input_ids, attention_mask = self.prepare_config_and_inputs()
        inputs_dict = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        return config, inputs_dict

    def create_and_check_model(self, config, input_ids, attention_mask):
        model = Phi4MultimodalForCausalLM(config=config)
        model.to(torch_device)
        model.eval()
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True,
            )["logits"]
        self.parent.assertEqual(logits.shape, (self.batch_size, self.seq_length, self.vocab_size))
        self.parent.assertFalse(torch.isnan(logits).any().item())


@require_torch
class Phi4MultimodalModelTest(ModelTesterMixin, GenerationTesterMixin, unittest.TestCase):
    """
    Model tester for `Phi4Multimodal`.
    """

    all_model_classes = (Phi4MultimodalForCausalLM,) if is_torch_available() else ()
    test_pruning = False
    test_head_masking = False
    _is_composite = True

    def setUp(self):
        self.model_tester = Phi4MultimodalModelTester(self)
        self.config_tester = ConfigTester(self, config_class=Phi4MultimodalConfig)

    # @unittest.skip(
    #     reason="This architecure seem to not compute gradients properly when using GC, check: https://github.com/huggingface/transformers/pull/27124"
    # )
    # def test_training_gradient_checkpointing(self):
    #     pass

    # @unittest.skip(
    #     reason="This architecure seem to not compute gradients properly when using GC, check: https://github.com/huggingface/transformers/pull/27124"
    # )
    # def test_training_gradient_checkpointing_use_reentrant(self):
    #     pass

    # @unittest.skip(
    #     reason="This architecure seem to not compute gradients properly when using GC, check: https://github.com/huggingface/transformers/pull/27124"
    # )
    # def test_training_gradient_checkpointing_use_reentrant_false(self):
    #     pass

    # @unittest.skip(reason="Compile not yet supported because in LLava models")
    # def test_sdpa_can_compile_dynamic(self):
    #     pass

    # @unittest.skip(reason="Compile not yet supported because in LLava models")
    # def test_sdpa_can_dispatch_on_flash(self):
    #     pass

    # @unittest.skip(reason="Feedforward chunking is not yet supported")
    # def test_feed_forward_chunking(self):
    #     pass

    # @unittest.skip(reason="Unstable test")
    # def test_initialization(self):
    #     pass

    # @unittest.skip(reason="Unstable test")
    # def test_dola_decoding_sample(self):
    #     pass

    # @unittest.skip(reason="Unsupported")
    # def test_generate_from_inputs_embeds_0_greedy(self):
    #     pass

    # @unittest.skip(reason="Unsupported")
    # def test_generate_from_inputs_embeds_1_beam_search(self):
    #     pass

    # @unittest.skip(reason="Dynamic control flow due to MoE")
    # def test_generate_with_static_cache(self):
    #     pass

    # @unittest.skip(reason="Dynamic control flow due to MoE")
    # def test_generate_from_inputs_embeds_with_static_cache(self):
    #     pass

    # @unittest.skip(reason="Dynamic control flow due to MoE")
    # def test_generate_compile_model_forward(self):
    #     pass


@require_torch
@slow
class Phi4MultimodalIntegrationTest(unittest.TestCase):

    checkpoint_path = "/raid/cyril/phi4-converted"
    image_url = "https://www.ilankelman.org/stopsigns/australia.jpg"
    audio_url = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2-Audio/audio/f2641_0_throatclearing.wav"

    def setUp(self):
        self.processor = AutoProcessor.from_pretrained(self.checkpoint_path)
        self.generation_config = GenerationConfig(max_new_tokens=20, do_sample=False)
        self.user_token = '<|user|>'
        self.assistant_token = '<|assistant|>'
        self.end_token = '<|end|>'
        self.image = Image.open(requests.get(self.image_url, stream=True).raw)
        with tempfile.NamedTemporaryFile(mode="w+b", suffix=".wav") as tmp:
            tmp.write(requests.get(self.audio_url, stream=True).raw.data)
            tmp.flush()
            tmp.seek(0)
            self.audio = soundfile.read(tmp.name)

    def tearDown(self):
        gc.collect()
        torch.cuda.empty_cache()

    def test_text_only_generation(self):
        model = AutoModelForCausalLM.from_pretrained(self.checkpoint_path, torch_dtype=torch.float16, device_map=torch_device)

        prompt = f"{self.user_token}What is the answer for 1+1? Explain it.{self.end_token}{self.assistant_token}"
        inputs = self.processor(prompt, images=None, return_tensors='pt').to(torch_device)

        output = model.generate(
            **inputs,
            generation_config=self.generation_config,
        )
        output = output[:, inputs['input_ids'].shape[1] :]
        response = self.processor.batch_decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

        EXPECTED_RESPONSE = "The answer for 1+1 is 2. This is a basic arithmetic operation where you are"

        self.assertEqual(response, EXPECTED_RESPONSE)


    def test_vision_text_generation(self):
        model = AutoModelForCausalLM.from_pretrained(self.checkpoint_path, torch_dtype=torch.float16, device_map=torch_device)

        prompt = f"{self.user_token}<|image_1|>What is shown in this image?{self.end_token}{self.assistant_token}"
        inputs = self.processor(prompt, images=self.image, return_tensors='pt').to(torch_device)

        output = model.generate(
            **inputs,
            generation_config=self.generation_config,
        )
        output = output[:, inputs['input_ids'].shape[1] :]
        response = self.processor.batch_decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

        EXPECTED_RESPONSE = "The image shows a vibrant scene in a Chinese-style street, featuring a red and gold Chinese-style arch"

        self.assertEqual(response, EXPECTED_RESPONSE)


    def test_multi_image_vision_text_generation(self):
        model = AutoModelForCausalLM.from_pretrained(self.checkpoint_path, torch_dtype=torch.float16, device_map=torch_device)

        images = []
        placeholder = ""
        for i in range(1, 5):
            url = f'https://image.slidesharecdn.com/azureintroduction-191206101932/75/Introduction-to-Microsoft-Azure-Cloud-{i}-2048.jpg'
            images.append(Image.open(requests.get(url, stream=True).raw))
            placeholder += f"<|image_{i}|>"

        prompt = f"{self.user_token}{placeholder}Summarize the deck of slides.{self.end_token}{self.assistant_token}"
        inputs = self.processor(prompt, images, return_tensors='pt').to(torch_device)

        output = model.generate(
            **inputs,
            generation_config=self.generation_config,
        )
        output = output[:, inputs['input_ids'].shape[1] :]
        response = self.processor.batch_decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

        EXPECTED_RESPONSE = "The presentation provides an overview of Microsoft Azure, a cloud computing platform by Microsoft, and its various services"

        self.assertEqual(response, EXPECTED_RESPONSE)


def test_audio_text_generation(self):
        model = AutoModelForCausalLM.from_pretrained(self.checkpoint_path, torch_dtype=torch.float16, device_map=torch_device)

        prompt = f"{self.user_token}<|audio_1|>What is happening in this audio?{self.end_token}{self.assistant_token}"
        inputs = self.processor(prompt, audios=[self.audio], return_tensors='pt').to(torch_device)

        output = model.generate(
            **inputs,
            generation_config=self.generation_config,
        )
        output = output[:, inputs['input_ids'].shape[1] :]
        response = self.processor.batch_decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

        # Yes, it is truly the expected response... Even though the model correctly treats the audio file
        EXPECTED_RESPONSE = "I'm sorry, but I can't listen to or analyze audio content. However, if you provide a description"

        self.assertEqual(response, EXPECTED_RESPONSE)
