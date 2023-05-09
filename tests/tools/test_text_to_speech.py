# coding=utf-8
# Copyright 2023 HuggingFace Inc.
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

from transformers import load_tool
from transformers.utils import is_torch_available

if is_torch_available():
    import torch

from .test_tools_common import ToolTesterMixin
from transformers.testing_utils import require_torch


@require_torch
class TextToSpeechToolTester(unittest.TestCase, ToolTesterMixin):
    def setUp(self):
        self.tool = load_tool("text-to-speech")
        self.tool.setup()

    def test_exact_match_arg(self):
        # SpeechT5 isn't deterministic
        torch.manual_seed(0)
        result = self.tool("hey")
        self.assertTrue(torch.allclose(result[:3], torch.tensor([-0.00040140701457858086, -0.0002551682700868696, -0.00010294507956132293])))

    def test_exact_match_kwarg(self):
        # SpeechT5 isn't deterministic
        torch.manual_seed(0)
        result = self.tool("hey")
        self.assertTrue(torch.allclose(result[:3], torch.tensor([-0.00040140701457858086, -0.0002551682700868696, -0.00010294507956132293])))
