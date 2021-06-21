#!/usr/bin/env python
# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
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
""" Finetuning a 🤗 Flax Transformers model for question-answering on natural-questions."""
import argparse
import logging
import os
import random
import time
from typing import Any, Callable, Dict, Tuple

import datasets
from datasets import load_dataset

import jax
import jax.numpy as jnp
import optax
import transformers
import flax.linen as nn
from flax import struct, traverse_util
from flax.jax_utils import replicate, unreplicate
from flax.metrics import tensorboard
from flax.training import train_state
from flax.training.common_utils import get_metrics, onehot, shard
from transformers import FlaxAutoModelForQuestionAnswering, AutoTokenizer

from transformers import BigBirdConfig
from transformers.models.big_bird.modeling_flax_big_bird import (
    FlaxBigBirdForQuestionAnsweringModule,
    FlaxBigBirdForQuestionAnswering,
)

logger = logging.getLogger(__name__)

Array = Any
Dataset = datasets.arrow_dataset.Dataset
PRNGKey = Any


class FlaxBigBirdForNaturalQuestionsModule(FlaxBigBirdForQuestionAnsweringModule):
    """
    BigBirdForQuestionAnswering with CLS Head over the top for predicting category

    This way we can load its weights with FlaxBigBirdForQuestionAnswering
    """

    config: BigBirdConfig
    dtype: jnp.dtype = jnp.float32
    add_pooling_layer: bool = True

    def setup(self):
        super().setup()
        self.cls = nn.Dense(5, dtype=self.dtype)

    def __call__(self, *args, **kwargs):
        outputs = super().__call__(*args, **kwargs)
        cls_out = self.cls(outputs[2])
        return outputs[:2] + (cls_out, )


class FlaxBigBirdForNaturalQuestions(FlaxBigBirdForQuestionAnswering):
    module_class = FlaxBigBirdForNaturalQuestionsModule


def create_train_state(
    model: FlaxAutoModelForQuestionAnswering,
    learning_rate_fn: Callable[[int], float],
    weight_decay: float,
) -> train_state.TrainState:
    """Create initial training state"""

    class TrainState(train_state.TrainState):
        """Train state with an Optax optimizer.

        Args:
          loss_fn: Function to compute the loss.
        """
        loss_fn: Callable = struct.field(pytree_node=False)

    # We use Optax's "masking" functionality to not apply weight decay
    # to bias and LayerNorm scale parameters. decay_mask_fn returns a
    # mask boolean with the same structure as the parameters.
    # The mask is True for parameters that should be decayed.
    def decay_mask_fn(params):
        flat_params = traverse_util.flatten_dict(params)
        flat_mask = {path: (path[-1] != "bias" and path[-2:] != ("LayerNorm", "scale")) for path in flat_params}
        return traverse_util.unflatten_dict(flat_mask)

    tx = optax.adamw(
        learning_rate=learning_rate_fn, b1=0.9, b2=0.999, eps=1e-6, weight_decay=weight_decay, mask=decay_mask_fn
    )

    def loss_fn(
        start_logits, start_labels, end_logits, end_labels, pooled_logits=None, pooled_labels=None
    ):
        def cross_entropy_loss(logits, labels):
            xentropy = optax.softmax_corss_entropy(logits, onehot(labels, num_classes=vocab_size))
            return jnp.mean(xentropy)

        vocab_size = start_logits.shape[-1]
        start_loss = cross_entropy_loss(start_logits, start_labels)
        end_loss = cross_entropy_loss(end_logits, end_labels)
        if pooled_labels is not None and pooled_logits is not None:
            pooled_loss = cross_entropy_loss(pooled_logits, pooled_labels)
            loss = (start_loss + end_loss + pooled_loss) / 3
        else:
            loss = (start_loss + end_loss) / 2
        return loss

    return TrainState.create(apply_fn=model.__call__, params=model.params, tx=tx, loss_fn=loss_fn)


def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a question answering task")

    parser.add_argument(
        "--train_file", type=str, default=None, help="A jsonl file containing the tokenized training data."
    )
    parser.add_argument(
        "--validation_file", type=str, default=None, help="A jsonl file containing the tokenized validation data."
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=1,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=1,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--lr1",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--lr2",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--block_size",
        type=int,
        default=84,
        help="No. of tokens in each block",
    )
    parser.add_argument(
        "--num_random_blocks",
        type=int,
        default=3,
        help="No. of random blocks",
    )
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=5, help="Total number of training epochs to perform.")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
    parser.add_argument("--seed", type=int, default=3, help="A seed for reproducible training.")
    args = parser.parse_args()

    # Sanity checks
    if args.train_file is None and args.validation_file is None:
        raise ValueError("Need either a task name or a training/validation file.")
    else:
        if args.train_file is not None:
            extension = args.train_file.split(".")[-1]
            assert extension in ["jsonl", "json"], "`train_file` should be a json/jsonl file."
        if args.validation_file is not None:
            extension = args.validation_file.split(".")[-1]
            assert extension in ["jsonl", "json"], "`validation_file` should be a json/jsonl file."

    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)

    return args


def format_inputs(features, pad_id, max_length=4096):
    def _format_inputs(input_ids: list):
        attention_mask = [1 for _ in range(len(input_ids))]
        while len(input_ids) < max_length:
            input_ids.append(pad_id)
            attention_mask.append(0)
        return input_ids, attention_mask

    inputs = [_format_inputs(ids) for ids in features["input_ids"]]
    input_ids, attention_mask = zip(*inputs)

    batch = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "start_labels": features["start_token"],
        "end_labels": features["end_token"],
        "pooled_labels": features["category"],
    }
    return batch


def train_data_collator(rng: PRNGKey, dataset: Dataset, batch_size: int, pad_id: int, max_length=4096):
    """Returns shuffled batches of size `batch_size` from truncated `train dataset`, sharded over all local devices."""
    steps_per_epoch = len(dataset) // batch_size
    perms = jax.random.permutation(rng, len(dataset))
    perms = perms[: steps_per_epoch * batch_size]  # Skip incomplete batch.
    perms = perms.reshape((steps_per_epoch, batch_size))

    for perm in perms:
        batch = dataset[perm]
        batch = format_inputs(dict(batch), pad_id, max_length=max_length)
        batch = {k: jnp.array(v, dtype=jnp.int32) for k, v in batch.items()}
        batch = shard(batch)

        yield batch


def eval_data_collator(dataset: Dataset, batch_size: int, pad_id: int, max_length=4096):
    """Returns batches of size `batch_size` from `eval dataset`, sharded over all local devices."""
    for i in range(len(dataset) // batch_size):
        batch = dataset[i * batch_size : (i + 1) * batch_size]
        batch = format_inputs(dict(batch), pad_id, max_length=max_length)
        batch = {k: jnp.array(v, dtype=jnp.int32) for k, v in batch.items()}
        batch = shard(batch)

        yield batch


def create_learning_rate_fn(
    train_ds_size: int,
    train_batch_size: int,
    num_train_epochs: int,
    lr1: float,
    lr2: float,
) -> Callable[[int], jnp.array]:
    """Returns a linear warmup, linear_decay learning rate function."""

    steps_per_epoch = train_ds_size // train_batch_size
    num_train_steps = steps_per_epoch * num_train_epochs

    # 30% of time, train with lr1
    # rest of time, train with lr2
    transition_steps = int(num_train_steps * 0.3)

    lr1 = optax.linear_schedule(init_value=lr1, end_value=lr1, transition_steps=transition_steps)
    lr2 = optax.linear_schedule(
        init_value=lr2, end_value=lr2, transition_steps=num_train_steps - transition_steps
    )
    schedule_fn = optax.join_schedules(schedules=[lr1, lr2], boundaries=[transition_steps])
    return schedule_fn


def main():
    args = parse_args()

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    # Setup logging, we only want one process per machine to log things on the screen.
    logger.setLevel(logging.INFO if jax.process_index() == 0 else logging.ERROR)
    if jax.process_index() == 0:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    model = FlaxBigBirdForNaturalQuestions.from_pretrained(
        args.model_name_or_path,
        block_size=args.block_size,
        num_random_blocks=args.num_random_blocks
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    # load dataset from files created using `prepare_natural_questions.py` script
    train_dataset = load_dataset("json", data_files=args.train_file, split="train")
    eval_dataset = load_dataset("json", data_files=args.validation_file, split="train")

    # Log a few random samples from the training set:
    for index in random.sample(range(len(train_dataset)), 3):
        logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    # Define a summary writer
    summary_writer = tensorboard.SummaryWriter(args.output_dir)
    summary_writer.hparams(vars(args))

    def write_metric(train_metrics, eval_metrics, train_time, step):
        summary_writer.scalar("train_time", train_time, step)

        train_metrics = get_metrics(train_metrics)
        for key, vals in train_metrics.items():
            tag = f"train_{key}"
            for i, val in enumerate(vals):
                summary_writer.scalar(tag, val, step - len(vals) + i + 1)

        for metric_name, value in eval_metrics.items():
            summary_writer.scalar(f"eval_{metric_name}", value, step)

    num_epochs = int(args.num_train_epochs)
    rng = jax.random.PRNGKey(args.seed)
    dropout_rng = jax.random.split(rng, jax.load_device_count())

    train_batch_size = args.per_device_train_batch_size * jax.local_device_count()
    eval_batch_size = args.per_device_eval_batch_size * jax.local_device_count()

    learning_rate_fn = create_learning_rate_fn(
        len(train_dataset), train_batch_size, args.num_train_epochs, args.lr1, args.lr2
    )

    state = create_train_state(
        model, learning_rate_fn, weight_decay=args.weight_decay
    )

    # define step functions
    def train_step(state: train_state.TrainState, batch: Dict[str, Array], dropout_rng: PRNGKey
    ) -> Tuple[train_state.TrainState, Dict[str, Array], PRNGKey]:
        """Trains model with an optimizer (both in `state`) on `batch`, returning a pair `(new_state, loss)`."""

        dropout_rng, new_dropout_rng = jax.random.split(dropout_rng)

        start_labels = batch.pop("start_labels")
        end_labels = batch.pop("end_labels")
        pooled_labels = batch.pop("pooled_labels", None)

        def loss_fn(params):

            outputs = state.apply_fn(
                **batch, params=params, dropout_rng=dropout_rng, train=True
            )
            start_logits, end_logits = outputs[:2]
            pooled_logits = outputs[2]

            return state.loss_fn(
                start_logits,
                start_labels,
                end_logits,
                end_labels,
                pooled_logits,
                pooled_labels,
            )

        grad_fn = jax.value_and_grad(loss_fn)
        loss, grads = grad_fn(state.params)
        grads = jax.lax.pmean(grads, "batch")
        metrics = jax.lax.pmean({"loss": loss, "learning_rate": learning_rate_fn(state.step)}, axis_name="batch")
        state = state.apply_gradients(grads=grads)
        return state, metrics, new_dropout_rng

    p_train_step = jax.pmap(train_step, axis_name="batch")

    def eval_step(state, batch):
        start_labels = batch.pop("start_labels")
        end_labels = batch.pop("end_labels")
        pooled_labels = batch.pop("pooled_labels", None)

        outputs = state.apply_fn(
            **batch, params=state.params, train=False
        )
        start_logits, end_logits = outputs[:2]
        pooled_logits = outputs[2]

        loss = state.loss_fn(
            start_logits, start_labels, end_logits, end_labels, pooled_logits, pooled_labels
        )
        metrics = jax.lax.pmean({"loss": loss}, axis_name="batch")
        return metrics

    p_eval_step = jax.pmap(eval_step, axis_name="batch")

    logger.info("===== Starting training ({num_epochs} epochs) =====")
    train_time = 0

    # make sure state (params + opt_state) is replicated on each device
    state = replicate(state)

    for epoch in range(1, num_epochs + 1):
        logger.info(f"Epoch {epoch}")
        logger.info("  Training...")

        train_start = time.time()
        train_metrics = []
        rng, input_rng = jax.random.split(rng)

        # train
        for batch in train_data_collator(input_rng, train_dataset, train_batch_size, tokenizer.pad_token_id, max_length=4096):
            state, metrics, dropout_rng = p_train_step(state, batch, dropout_rng)
            train_metrics.append(metrics)
        train_time += time.time() - train_start
        logger.info(f"    Done! Training metrics: {unreplicate(metrics)}")

        logger.info("  Evaluating...")

        losses = []

        # evaluate
        for batch in eval_data_collator(eval_dataset, eval_batch_size, tokenizer.pad_token_id, max_length=4096):
            metric = p_eval_step(state, batch)
            losses.append(unreplicate(metric)["loss"])

        # evaluate also on leftover examples (not divisible by batch_size)
        num_leftover_samples = len(eval_dataset) % eval_batch_size

        # make sure leftover batch is evaluated on one device
        if num_leftover_samples > 0 and jax.process_index() == 0:
            # take leftover samples
            batch = eval_dataset[-num_leftover_samples:]
            batch = {k: jnp.array(v) for k, v in batch.items()}

            metric = eval_step(state, batch)
            losses.append(unreplicate(metric)["loss"])

        eval_metric = {"loss": losses}
        logger.info(f"    Done! Eval metrics: {unreplicate(eval_metric)}")

        cur_step = epoch * (len(train_dataset) // train_batch_size)
        write_metric(train_metrics, eval_metric, train_time, cur_step)


    # save last checkpoint
    if jax.process_index() == 0:
        params = jax.device_get(jax.tree_map(lambda x: x[0], state.params))
        model.save_pretrained(args.output_dir, params=params)


if __name__ == "__main__":
    main()
