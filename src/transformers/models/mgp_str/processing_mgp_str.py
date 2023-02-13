# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
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
"""Processor class for MGP-STR."""

import warnings

from transformers import BertTokenizer, GPT2Tokenizer
from transformers.utils import is_torch_available
from transformers.utils.generic import ExplicitEnum

from ...processing_utils import ProcessorMixin


if is_torch_available():
    import torch


class DecodeType(ExplicitEnum):
    CHARACTER = "char"
    CHARACTER_EOS_TOKEN = 1
    CHARACTER_EOS_STR = "[s]"
    BPE = "bpe"
    BPE_EOS_TOKEN = 2
    BPE_EOS_STR = "#"
    WORDPIECE = "wp"
    WORDPIECE_EOS_TOKEN = 102
    WORDPIECE_EOS_STR = "[SEP]"


SUPPORTED_ANNOTATION_FORMATS = (DecodeType.CHARACTER, DecodeType.BPE, DecodeType.WORDPIECE)


class MGPSTRProcessor(ProcessorMixin):
    r"""
    Constructs a MGP-STR processor which wraps an image processor and MGP-STR tokenizers into a single

    [`MGPSTRProcessor`] offers all the functionalities of `ViTImageProcessor`] and [`MGPSTRTokenizer`]. See the
    [`~MGPSTRProcessor.__call__`] and [`~MGPSTRProcessor.batch_decode`] for more information.

    Args:
        image_processor (`ViTImageProcessor`):
            An instance of `ViTImageProcessor`. The image extractor is a required input.
        tokenizer ([`MGPSTRTokenizer`]):
            The tokenizer is a required input.
    """
    attributes = ["image_processor", "tokenizer"]
    image_processor_class = "ViTImageProcessor"
    tokenizer_class = "MGPSTRTokenizer"

    def __init__(self, image_processor=None, tokenizer=None, *kwargs):
        if "feature_extractor" in kwargs:
            warnings.warn(
                "The `feature_extractor` argument is deprecated and will be removed in v5, use `image_processor`"
                " instead.",
                FutureWarning,
            )
            feature_extractor = kwargs.pop("feature_extractor")

        image_processor = image_processor if image_processor is not None else feature_extractor
        if image_processor is None:
            raise ValueError("You need to specify an `image_processor`.")
        if tokenizer is None:
            raise ValueError("You need to specify a `tokenizer`.")

        self.char_tokenizer = tokenizer
        self.bpe_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.wp_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

        super().__init__(image_processor, tokenizer)

    def __call__(self, *args, **kwargs):
        """
        When used in normal mode, this method forwards all its arguments to ViTImageProcessor's
        [`~ViTImageProcessor.__call__`] and returns its output. This method also forwards the `text`
        and `kwargs` arguments to MGPSTRTokenizer's [`~MGPSTRTokenizer.__call__`] if `text` is not `None` to encode
        the text. Please refer to the doctsring of the above methods for more information.
        """
        images = kwargs.pop("images", None)
        text = kwargs.pop("text", None)

        if len(args) > 0:
            images = args[0]
            args = args[1:]

        if images is None and text is None:
            raise ValueError("You need to specify either an `images` or `text` input to process.")

        if images is not None:
            inputs = self.image_processor(images, *args, **kwargs)
        if text is not None:
            encodings = self.tokenizer(text, **kwargs)

        if text is None:
            return inputs
        elif images is None:
            return encodings
        else:
            inputs["labels"] = encodings["input_ids"]
            return inputs

    def batch_decode(self, sequences):
        """
        Convert a list of lists of token ids into a list of strings by calling decode.

        Args:
            sequences (`torch.Tensor`):
                List of tokenized input ids.

        Returns:
            `Dict[str, any]`: Dictionary of all the outputs of the decoded results.
                generated_text (`List[str]`): The final results after fusion of char, bpe, and wp. scores
                (`List[float]`): The final scores after fusion of char, bpe, and wp. char_preds (`List[str]`): The list
                of character decoded sentences. bpe_preds (`List[str]`): The list of bpe decoded sentences. wp_preds
                (`List[str]`): The list of wp decoded sentences.

        This method forwards all its arguments to PreTrainedTokenizer's [`~PreTrainedTokenizer.batch_decode`]. Please
        refer to the docstring of this method for more information.
        """
        char_preds, bpe_preds, wp_preds = sequences
        batch_size = char_preds.size(0)

        char_strs, char_scores = self._decode_helper(char_preds, "char")
        bpe_strs, bpe_scores = self._decode_helper(bpe_preds, "bpe")
        wp_strs, wp_scores = self._decode_helper(wp_preds, "wp")

        final_strs = []
        final_scores = []
        for i in range(batch_size):
            scores = [char_scores[i], bpe_scores[i], wp_scores[i]]
            strs = [char_strs[i], bpe_strs[i], wp_strs[i]]
            max_score_index = scores.index(max(scores))
            final_strs.append(strs[max_score_index])
            final_scores.append(scores[max_score_index])

        out = {}
        out["generated_text"] = final_strs
        out["scores"] = final_scores
        out["char_preds"] = char_strs
        out["bpe_preds"] = bpe_strs
        out["wp_preds"] = wp_strs
        return out

    def _decode_helper(self, pred_logits, format):
        """
        Convert a list of lists of bpe token ids into a list of strings by calling bpe tokenizer.

        Args:
            pred_logits (`torch.Tensor`):
                List of model prediction logits.
            format (`Enum`):
                Type of model prediction. Must be one of ['char', 'bpe', 'wp'].
        Returns:
            `tuple`:
                dec_strs(`str`): The decode strings of model prediction. conf_scores(`List[float]`): The confidence
                score of model prediction.
        """
        if format == DecodeType.CHARACTER:
            decoder = self.char_tokenizer.batch_decode
            eos_token = int(DecodeType.CHARACTER_EOS_TOKEN)
            eos_str = DecodeType.CHARACTER_EOS_STR
        elif format == DecodeType.BPE:
            decoder = self.bpe_decode
            eos_token = int(DecodeType.BPE_EOS_TOKEN)
            eos_str = DecodeType.BPE_EOS_STR
        elif format == DecodeType.WORDPIECE:
            decoder = self.wp_decode
            eos_token = int(DecodeType.WORDPIECE_EOS_TOKEN)
            eos_str = DecodeType.WORDPIECE_EOS_STR
        else:
            raise ValueError(f"Format {format} is not supported.")

        dec_strs, conf_scores = [], []
        batch_size = pred_logits.size(0)
        batch_max_length = pred_logits.size(1)
        _, preds_index = pred_logits.topk(1, dim=-1, largest=True, sorted=True)
        preds_index = preds_index.view(-1, batch_max_length)[:, 1:]
        preds_str = decoder(preds_index)
        preds_max_prob, _ = torch.nn.functional.softmax(pred_logits, dim=2).max(dim=2)
        preds_max_prob = preds_max_prob[:, 1:]

        for index in range(batch_size):
            pred_EOS = preds_str[index].find(eos_str)
            pred = preds_str[index][:pred_EOS]
            pred_index = preds_index[index].cpu().tolist()
            pred_EOS_index = pred_index.index(eos_token) if eos_token in pred_index else -1
            pred_max_prob = preds_max_prob[index][: pred_EOS_index + 1]
            confidence_score = pred_max_prob.cumprod(dim=0)[-1] if pred_max_prob.nelement() != 0 else 0.0
            dec_strs.append(pred)
            conf_scores.append(confidence_score)

        return dec_strs, conf_scores

    def bpe_decode(self, sequences):
        """
        Convert a list of lists of bpe token ids into a list of strings by calling bpe tokenizer.

        Args:
            sequences (`torch.Tensor`):
                List of tokenized input ids.
        Returns:
            `List[str]`: The list of bpe decoded sentences.
        """
        decode_strs = []
        for seq in sequences:
            tokenstr = self.bpe_tokenizer.decode(seq)
            decode_strs.append(tokenstr)
        return decode_strs

    def wp_decode(self, sequences):
        """
        Convert a list of lists of word piece token ids into a list of strings by calling word piece tokenizer.

        Args:
            sequences (`torch.Tensor`):
                List of tokenized input ids.
        Returns:
            `List[str]`: The list of wp decoded sentences.
        """
        decode_strs = []
        for seq in sequences:
            tokenstr = self.wp_tokenizer.decode(seq)
            tokenlist = tokenstr.split()
            decode_strs.append("".join(tokenlist))
        return decode_strs
