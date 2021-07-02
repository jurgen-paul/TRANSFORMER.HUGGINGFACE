# Copyright 2020 The HuggingFace Team. All rights reserved.
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

<<<<<<< HEAD
=======
from dataclasses import dataclass
>>>>>>> 184b3b16db95f94e28372ca2d304dc75781c77a4
from typing import Dict, List, Optional, Tuple, Union

import tensorflow as tf

<<<<<<< HEAD
from ..tokenization_utils_base import PreTrainedTokenizerBase


class TFDataCollatorForLanguageModeling:
    """
    Data collator used for language modeling. Encodes sequences for Masked Language Modeling as mentioned in the paper
    'BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding'.

    Labels are -100 for non-masked tokens and the value to predict the masked token.
=======
import tensorflow_probability as tfp

from ..tokenization_utils_base import BatchEncoding, PreTrainedTokenizerBase


def _tf_collate_batch(examples, tokenizer, pad_to_multiple_of: Optional[int] = None):
    """Collate `examples` into a batch, using the information in `tokenizer` for padding if necessary."""
    # Tensorize if necessary.
    if isinstance(examples[0], (list, tuple)):
        examples = [tf.constant(e, dtype=tf.float64) for e in examples]

    # Check if padding is necessary.
    length_of_first = examples[0].shape[0]
    are_tensors_same_length = all(x.shape[0] == length_of_first for x in examples)
    if are_tensors_same_length and (pad_to_multiple_of is None or length_of_first % pad_to_multiple_of == 0):
        return tf.stack(examples, axis=0)

    # If yes, check if we have a `pad_token`.
    if tokenizer._pad_token is None:
        raise ValueError(
            "You are attempting to pad samples but the tokenizer you are using"
            f" ({tokenizer.__class__.__name__}) does not have a pad token."
        )

    # Creating the full tensor and filling it with our data.
    max_length = max(x.shape[0] for x in examples)
    if pad_to_multiple_of is not None and (max_length % pad_to_multiple_of != 0):
        max_length = ((max_length // pad_to_multiple_of) + 1) * pad_to_multiple_of

    result = tf.fill([len(examples), max_length], tokenizer.pad_token_id)
    for i, example in enumerate(examples):
        if tokenizer.padding_side == "right":
            temp_result = result.numpy()
            temp_result[i, : example.shape[0]] = example
            result = tf.convert_to_tensor(temp_result)
        else:
            temp_result = result.numpy()
            result[i, -example.shape[0] :] = example
            result = tf.convert_to_tensor(temp_result)
    return result


@dataclass
class TFDataCollatorForLanguageModeling:
    """
    Data collator used for language modeling. Inputs are dynamically padded to the maximum length of a batch if they
    are not all of the same length.
>>>>>>> 184b3b16db95f94e28372ca2d304dc75781c77a4

    Args:
        tokenizer (:class:`~transformers.PreTrainedTokenizer` or :class:`~transformers.PreTrainedTokenizerFast`):
            The tokenizer used for encoding the data.
<<<<<<< HEAD
        batch_size (:obj:`int`):
            The size of the batch.
        padding_length (:obj:`int`, `optional`):
            The length of the vector concatenated to the largest element.
        mlm_probability (:obj:`float`, `optional`, defaults to 0.15):
            The probability with which to (randomly) mask tokens in the input.
        special_tokens_mask (:obj:`tf.Tensor`, `optional`):
            If set, special tokens will have zero probability of being masked.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        batch_size: int,
        special_tokens_mask: tf.Tensor = None,
        padding_length: int = None,
    ):

        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.padding_length = padding_length
        self.special_tokens_mask = special_tokens_mask
        self.mlm_probability = 0.15

    def __post_init__(self):
        if self.tokenizer.mask_token is None:
=======
        mlm (:obj:`bool`, `optional`, defaults to :obj:`True`):
            Whether or not to use masked language modeling. If set to :obj:`False`, the labels are the same as the
            inputs with the padding tokens ignored (by setting them to -100). Otherwise, the labels are -100 for
            non-masked tokens and the value to predict for the masked token.
        mlm_probability (:obj:`float`, `optional`, defaults to 0.15):
            The probability with which to (randomly) mask tokens in the input, when :obj:`mlm` is set to :obj:`True`.
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.

    .. note::

        For best performance, this data collator should be used with a dataset having items that are dictionaries or
        BatchEncoding, with the :obj:`"special_tokens_mask"` key, as returned by a
        :class:`~transformers.PreTrainedTokenizer` or a :class:`~transformers.PreTrainedTokenizerFast` with the
        argument :obj:`return_special_tokens_mask=True`.
    """

    tokenizer: PreTrainedTokenizerBase
    mlm: bool = True
    mlm_probability: float = 0.15
    pad_to_multiple_of: Optional[int] = None

    def __post_init__(self):
        if self.mlm and self.tokenizer.mask_token is None:
>>>>>>> 184b3b16db95f94e28372ca2d304dc75781c77a4
            raise ValueError(
                "This tokenizer does not have a mask token which is necessary for masked language modeling. "
                "You should pass `mlm=False` to train on causal language modeling instead."
            )

<<<<<<< HEAD
    @tf.function
    def pseudo_bernoulli(self, prob_matrix, labels):
        return tf.cast(prob_matrix - tf.random.uniform(tf.shape(labels), 0, 1) >= 0, tf.bool)

    @tf.function
    def mask_special_tokens(self, labels, special_tokens):
        # Finds all special tokens within labels
        x = tf.map_fn(lambda b: tf.cast(tf.math.equal(labels, b), tf.int32), special_tokens)
        return tf.math.greater(tf.reduce_sum(x, axis=0), 0)

    @tf.function
    def tf_pad_tokens(self, examples, tokenizer, pad_to_multiple_of: Optional[int] = None):
        """Collate `examples` into a batch, using the information in `tokenizer` for padding if necessary."""
        shape = examples.shape
        multiple_tensors = len(tf.shape(examples)) > 1

        # Tensorize if necessary.
        if not isinstance(examples[0], tf.Tensor):
            temporary = []
            for e in examples:
                temporary.append(tf.constant(e, dtype=tf.float64))
            examples = temporary

        # Check if padding is necessary.
        if multiple_tensors:
            length_of_first = tf.shape(examples[0])[0]

            are_tensors_same_length = True
            for x in examples:
                are_tensors_same_length &= tf.shape(x)[0] == length_of_first

            no_padding_necessary = are_tensors_same_length and (
                pad_to_multiple_of is None or length_of_first % pad_to_multiple_of == 0
            )
        else:
            length_of_first = tf.shape(examples)[0] if shape[0] is None else shape[0]
            no_padding_necessary = pad_to_multiple_of is None or length_of_first % pad_to_multiple_of == 0

        if no_padding_necessary:
            return tf.stack(examples, axis=0)

        # Check if we have a `pad_token`.
        if tokenizer._pad_token is None:
            raise ValueError(
                "You are attempting to pad samples but the tokenizer you are using"
                f" ({tokenizer.__class__.__name__}) does not have a pad token."
            )

        # Padding our tensor with the appropriate pad token.
        max_length = length_of_first
        if multiple_tensors:
            for x in examples:
                if x.shape[0] > max_length:
                    max_length = tf.shape(x)[0]

        if pad_to_multiple_of is not None and (max_length % pad_to_multiple_of != 0):
            max_length = ((max_length // pad_to_multiple_of) + 1) * pad_to_multiple_of - length_of_first

        if tokenizer.padding_side == "right":
            if multiple_tensors:
                result = tf.pad(examples, [[0, 0], [0, max_length]], constant_values=tokenizer.pad_token_id)
            else:
                result = tf.pad(examples, [[0, max_length]], constant_values=tokenizer.pad_token_id)
        else:
            if multiple_tensors:
                result = tf.pad(examples, [[0, 0], [max_length, 0]], constant_values=tokenizer.pad_token_id)
            else:
                result = tf.pad(examples, [[max_length, 0]], constant_values=tokenizer.pad_token_id)

        return result

    @tf.function()
    def encode_objects(self, examples: Union[List[int], tf.Tensor, Dict[str, tf.Tensor]]) -> tf.data.Dataset:
        if type(examples) == dict:
            input = examples["input_ids"]
        else:
            input = examples

        padded_output = self.tf_pad_tokens(input, self.tokenizer, self.padding_length)

        encoded_batch = dict()
        if type(examples) == dict:
            encoded_batch.update(examples)
            if self.padding_length is not None:
                encoded_batch["token_type_ids"] = self.tf_pad_tokens(
                    encoded_batch["token_type_ids"], self.tokenizer, self.padding_length
                )

        # Mask example sequences and create their respective labels
        encoded_batch["input_ids"], encoded_batch["labels"] = self.tf_mask_tokens(padded_output)
        return encoded_batch

    @tf.function
    def square_ragged_tensors(self, examples):
        if type(examples) == dict:
            examples["input_ids"] = examples["input_ids"].to_tensor(0)
            return examples
        else:
            return examples.to_tensor(0)

    @tf.function
    def __call__(self, examples: tf.data.Dataset, is_ragged=False) -> tf.data.Dataset:
        if is_ragged:
            examples = examples.batch(self.batch_size).map(self.square_ragged_tensors).unbatch()

        return examples.map(self.encode_objects).batch(self.batch_size)

    @tf.function
    def tf_mask_tokens(self, inputs: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        - Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """
        labels = tf.identity(inputs)

        # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
        probability_matrix = tf.fill(tf.shape(labels), self.mlm_probability)

        if self.special_tokens_mask is None:
            special_tokens_tensor = tf.constant(self.tokenizer.all_special_ids, dtype=tf.int32)
            special_tokens_mask = self.mask_special_tokens(labels, special_tokens_tensor)
        else:
            special_tokens_mask = tf.cast(self.special_tokens_mask, dtype=tf.bool)

        probability_matrix = tf.where(~special_tokens_mask, probability_matrix, 0)
        masked_indices = self.pseudo_bernoulli(probability_matrix, labels)
=======
    def __call__(self, examples: List[Union[List[int], tf.Tensor, Dict[str, tf.Tensor]]]) -> Dict[str, tf.Tensor]:
        # Handle dict or lists with proper padding and conversion to tensor.
        if isinstance(examples[0], (dict, BatchEncoding)):
            batch = self.tokenizer.pad(examples, return_tensors="tf", pad_to_multiple_of=self.pad_to_multiple_of)
        else:
            batch = {
                "input_ids": _tf_collate_batch(examples, self.tokenizer, pad_to_multiple_of=self.pad_to_multiple_of)
            }

        # If special token mask has been preprocessed, pop it from the dict.
        special_tokens_mask = batch.pop("special_tokens_mask", None)
        if self.mlm:
            batch["input_ids"], batch["labels"] = self.tf_mask_tokens(
                batch["input_ids"], special_tokens_mask=special_tokens_mask
            )
        else:
            labels = tf.identity(batch["input_ids"])
            if self.tokenizer.pad_token_id is not None:
                labels = labels.numpy()
                labels[labels == self.tokenizer.pad_token_id] = -100
                labels = tf.constant(labels)
            batch["labels"] = labels
        return batch

    def tf_mask_tokens(
        self, inputs: tf.Tensor, special_tokens_mask: Optional[tf.Tensor] = None
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """
        labels = tf.identity(inputs)
        # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
        probability_matrix = tf.fill(labels.shape, self.mlm_probability)
        if special_tokens_mask is None:
            special_tokens_mask = [
                self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in list(labels)
            ]
            special_tokens_mask = tf.cast(tf.constant(special_tokens_mask, dtype=tf.float16), dtype=tf.bool)
        else:
            special_tokens_mask = tf.cast(special_tokens_mask, dtype=tf.bool)

        probability_matrix = tf.where(~special_tokens_mask, probability_matrix, 0)
        masked_indices = tfp.distributions.Bernoulli(probs=probability_matrix, dtype=tf.bool).sample()
>>>>>>> 184b3b16db95f94e28372ca2d304dc75781c77a4

        labels = tf.where(masked_indices, labels, -100)  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
<<<<<<< HEAD
        indices_replaced = self.pseudo_bernoulli(tf.fill(tf.shape(labels), 0.8), labels) & masked_indices
=======

        indices_replaced = (
            tfp.distributions.Bernoulli(probs=tf.fill(labels.shape, 0.8), dtype=tf.bool).sample() & masked_indices
        )
>>>>>>> 184b3b16db95f94e28372ca2d304dc75781c77a4

        mask_token = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)
        inputs = tf.where(~indices_replaced, inputs, mask_token)

        # 10% of the time, we replace masked input tokens with random word
        indices_random = (
<<<<<<< HEAD
            self.pseudo_bernoulli(tf.fill(tf.shape(labels), 0.5), labels) & masked_indices & ~indices_replaced
        )

        random_words = tf.random.uniform(tf.shape(labels), maxval=len(self.tokenizer), dtype=tf.int32)

=======
            tfp.distributions.Bernoulli(probs=tf.fill(labels.shape, 0.5), dtype=tf.bool).sample()
            & masked_indices
            & ~indices_replaced
        )

        random_words = tf.random.uniform(labels.shape, maxval=len(self.tokenizer), dtype=tf.float64)

        inputs = tf.cast(inputs, dtype=tf.float64)
>>>>>>> 184b3b16db95f94e28372ca2d304dc75781c77a4
        inputs = tf.where(indices_random, random_words, inputs)

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels
