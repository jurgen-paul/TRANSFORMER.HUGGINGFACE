# Copyright 2021 The HuggingFace Team. All rights reserved.
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


from transformers.testing_utils import require_vision
from transformers.utils import is_vision_available


if is_vision_available():
    from PIL import Image

    from transformers import AutoTokenizer, LlavaProcessor


@require_vision
class LlavaProcessorTest(unittest.TestCase):

    def test_can_load_various_tokenizers(self):
        for checkpoint in ["Intel/llava-gemma-2b", "llava-hf/llava-1.5-7b-hf"]:
            processor = LlavaProcessor.from_pretrained(checkpoint)
            tokenizer = AutoTokenizer.from_pretrained(checkpoint)
            self.assertEqual(processor.tokenizer, tokenizer)