#!/usr/bin/env python3
import datasets
import torch
from typing import Dict, List, Union, Optional
from dataclasses import dataclass

import soundfile as sf
from transformers import Trainer, TrainingArguments, Wav2Vec2ForCTC, Wav2Vec2Tokenizer, PreTrainedTokenizer
import numpy as np

from jiwer import wer


model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base")
tokenizer = Wav2Vec2Tokenizer.from_pretrained("facebook/wav2vec2-base")

train_dataset = datasets.load_dataset("librispeech_asr", "clean", split="train.100")
val_dataset = datasets.load_dataset("librispeech_asr", "clean", split="validation")


def map_to_array(batch):
    speech_array, _ = sf.read(batch["file"])
    batch["speech"] = speech_array
    return batch


train_dataset = train_dataset.map(map_to_array, remove_columns=["file"])
val_dataset = val_dataset.map(map_to_array, remove_columns=["file"])


def prepare_dataset(batch):
    batch["input_values"] = tokenizer(batch["speech"]).input_values
    batch["labels"] = tokenizer.encode_labels(batch["text"]).labels
    return batch


train_dataset = train_dataset.map(prepare_dataset, batch_size=16, batched=True, num_proc=32)
val_dataset = val_dataset.map(prepare_dataset, batch_size=16, batched=True, num_proc=32)


@dataclass
class DataCollatorCTCWithPadding:
    """
    Data collator that will dynamically pad the inputs received.
    Args:
        tokenizer (:class:`~transformers.PreTrainedTokenizer` or :class:`~transformers.PreTrainedTokenizerFast`):
            The tokenizer used for encoding the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
        max_length (:obj:`int`, `optional`):
            Maximum length of the returned list and optionally padding length (see above).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.
            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
    """

    tokenizer: PreTrainedTokenizer
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    max_length_labels: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    pad_to_multiple_of_labels: Optional[int] = None
    return_attention_mask: bool = True

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        label_features = [{"input_values": feature["labels"]} for feature in features]
        batch = self.tokenizer.pad(
            input_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_attention_mask=self.return_attention_mask,
            return_tensors="pt",
        )
        labels_batch = self.tokenizer.pad(
            label_features,
            padding=self.padding,
            max_length=self.max_length_labels,
            pad_to_multiple_of=self.pad_to_multiple_of_labels,
            return_tensors="pt",
        )

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_values"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        batch["labels"] = labels

        return batch


data_collator = DataCollatorCTCWithPadding(tokenizer=tokenizer, padding=True, return_attention_mask=False)


def compute_metrics(pred):
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)

    pred.label_ids[pred.label_ids == -100] = 0

    pred_str = tokenizer.batch_decode(pred_ids)
    label_str = tokenizer.batch_decode(pred.label_ids)

    wer_score = wer(label_str, pred_str)

    return {"wer": wer_score}


training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=25,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=16,
    evaluation_strategy="steps",
    gradient_accumulation_steps=2,
    save_total_limit=3,
    save_steps=500,
    eval_steps=100,
    logging_steps=50,
    learning_rate=1e-4,
    warmup_steps=3000,
    group_by_length=True,
    logging_dir="./logs",
    fp16=True,
)

trainer = Trainer(
    model=model,
    data_collator=data_collator,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
)

trainer.train()
