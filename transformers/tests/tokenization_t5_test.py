# coding=utf-8
# Copyright 2018 Google T5 Authors and HuggingFace Inc. team.
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
from __future__ import absolute_import, division, print_function, unicode_literals

import os
import unittest

from transformers.tokenization_t5 import (T5Tokenizer)
from transformers.tokenization_xlnet import SPIECE_UNDERLINE

from .tokenization_tests_commons import CommonTestCases

SAMPLE_VOCAB = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                    'fixtures/test_sentencepiece.model')

class T5TokenizationTest(CommonTestCases.CommonTokenizerTester):

    tokenizer_class = T5Tokenizer

    def setUp(self):
        super(T5TokenizationTest, self).setUp()

        # We have a SentencePiece fixture for testing
        tokenizer = T5Tokenizer(SAMPLE_VOCAB)
        tokenizer.save_pretrained(self.tmpdirname)

    def get_tokenizer(self, **kwargs):
        return T5Tokenizer.from_pretrained(self.tmpdirname, **kwargs)

    def get_input_output_texts(self):
        input_text = u"This is a test"
        output_text = u"This is a test"
        return input_text, output_text

    def test_full_tokenizer(self):
        tokenizer = T5Tokenizer(SAMPLE_VOCAB)

        text = u'This is a test'
        tokens = tokenizer.tokenize(text)
        tokens_wo, offsets = tokenizer.tokenize_with_offsets(text)
        self.assertEqual(len(tokens_wo), len(offsets))
        self.assertListEqual(tokens, tokens_wo)
        self.assertListEqual(tokens, [u'▁This', u'▁is', u'▁a', u'▁t', u'est'])
        self.assertListEqual(offsets, [0, 5, 8, 10, 11])

        self.assertListEqual(
            tokenizer.convert_tokens_to_ids(tokens), [285, 46, 10, 170, 382])

        text = u"I was born in 92000, and this is falsé."
        tokens = tokenizer.tokenize(text)
        tokens_wo, offsets = tokenizer.tokenize_with_offsets(text)
        self.assertEqual(len(tokens_wo), len(offsets))
        self.assertListEqual(tokens, tokens_wo)
        self.assertListEqual(tokens, [SPIECE_UNDERLINE + u'I', SPIECE_UNDERLINE + u'was', SPIECE_UNDERLINE + u'b',
                                    u'or', u'n', SPIECE_UNDERLINE + u'in', SPIECE_UNDERLINE + u'',
                                    u'9', u'2', u'0', u'0', u'0', u',', SPIECE_UNDERLINE + u'and', SPIECE_UNDERLINE + u'this',
                                    SPIECE_UNDERLINE + u'is', SPIECE_UNDERLINE + u'f', u'al', u's', u'é', u'.'])
        self.assertListEqual(offsets, [0, 2, 6, 7, 9, 11, 14, 14, 15, 16, 17, 18, 19, 21, 25, 30, 33, 34, 36, 37, 38])
        ids = tokenizer.convert_tokens_to_ids(tokens)
        self.assertListEqual(
            ids, [8, 21, 84, 55, 24, 19, 7, 0,
                602, 347, 347, 347, 3, 12, 66,
                46, 72, 80, 6, 0, 4])

        back_tokens = tokenizer.convert_ids_to_tokens(ids)
        self.assertListEqual(back_tokens, [SPIECE_UNDERLINE + u'I', SPIECE_UNDERLINE + u'was', SPIECE_UNDERLINE + u'b',
                                        u'or', u'n', SPIECE_UNDERLINE + u'in',
                                        SPIECE_UNDERLINE + u'', u'<unk>', u'2', u'0', u'0', u'0', u',',
                                        SPIECE_UNDERLINE + u'and', SPIECE_UNDERLINE + u'this',
                                        SPIECE_UNDERLINE + u'is', SPIECE_UNDERLINE + u'f', u'al', u's',
                                        u'<unk>', u'.'])


if __name__ == '__main__':
    unittest.main()
