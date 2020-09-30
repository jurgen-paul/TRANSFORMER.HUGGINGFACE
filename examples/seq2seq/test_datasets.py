import os
import argparse
import tempfile
from pathlib import Path

import numpy as np
import pytest
from pack_dataset import pack_data_dir
from save_len_file import save_len_file
from test_seq2seq_examples import ARTICLES, BART_TINY, MARIAN_TINY, MBART_TINY, SUMMARIES, T5_TINY, make_test_data_dir
from torch.utils.data import DataLoader

from transformers import AutoTokenizer
from transformers.modeling_bart import shift_tokens_right
from transformers.testing_utils import slow
from utils import FAIRSEQ_AVAILABLE, DistributedSortishSampler, LegacySeq2SeqDataset, Seq2SeqDataset

BERT_BASE_CASED = "bert-base-cased"
PEGASUS_XSUM = "google/pegasus-xsum"


@slow
@pytest.mark.parametrize(
    "tok_name",
    [
        MBART_TINY,
        MARIAN_TINY,
        T5_TINY,
        BART_TINY,
        PEGASUS_XSUM,
    ],
)
def test_seq2seq_dataset_truncation(tok_name):
    tokenizer = AutoTokenizer.from_pretrained(tok_name)
    tmp_dir = make_test_data_dir()
    max_len_source = max(len(tokenizer.encode(a)) for a in ARTICLES)
    max_len_target = max(len(tokenizer.encode(a)) for a in SUMMARIES)
    max_src_len = 4
    max_tgt_len = 8
    assert max_len_target > max_src_len  # Will be truncated
    assert max_len_source > max_src_len  # Will be truncated
    src_lang, tgt_lang = "ro_RO", "de_DE"  # ignored for all but mbart, but never causes error.
    train_dataset = Seq2SeqDataset(
        tokenizer,
        data_dir=tmp_dir,
        type_path="train",
        max_source_length=max_src_len,
        max_target_length=max_tgt_len,  # ignored
        src_lang=src_lang,
        tgt_lang=tgt_lang,
    )
    dataloader = DataLoader(train_dataset, batch_size=2, collate_fn=train_dataset.collate_fn)
    for batch in dataloader:
        assert isinstance(batch, dict)
        assert batch["attention_mask"].shape == batch["input_ids"].shape
        # show that articles were trimmed.
        assert batch["input_ids"].shape[1] == max_src_len
        # show that targets are the same len
        assert batch["labels"].shape[1] == max_tgt_len
        if tok_name != MBART_TINY:
            continue
        # check language codes in correct place
        batch["decoder_input_ids"] = shift_tokens_right(batch["labels"], tokenizer.pad_token_id)
        assert batch["decoder_input_ids"][0, 0].item() == tokenizer.lang_code_to_id[tgt_lang]
        assert batch["decoder_input_ids"][0, -1].item() == tokenizer.eos_token_id
        assert batch["input_ids"][0, -2].item() == tokenizer.eos_token_id
        assert batch["input_ids"][0, -1].item() == tokenizer.lang_code_to_id[src_lang]

        break  # No need to test every batch


@pytest.mark.parametrize("tok", [BART_TINY, BERT_BASE_CASED])
def test_legacy_dataset_truncation(tok):
    tokenizer = AutoTokenizer.from_pretrained(tok)
    tmp_dir = make_test_data_dir()
    max_len_source = max(len(tokenizer.encode(a)) for a in ARTICLES)
    max_len_target = max(len(tokenizer.encode(a)) for a in SUMMARIES)
    trunc_target = 4
    train_dataset = LegacySeq2SeqDataset(
        tokenizer,
        data_dir=tmp_dir,
        type_path="train",
        max_source_length=20,
        max_target_length=trunc_target,
    )
    dataloader = DataLoader(train_dataset, batch_size=2, collate_fn=train_dataset.collate_fn)
    for batch in dataloader:
        assert batch["attention_mask"].shape == batch["input_ids"].shape
        # show that articles were trimmed.
        assert batch["input_ids"].shape[1] == max_len_source
        assert 20 >= batch["input_ids"].shape[1]  # trimmed significantly
        # show that targets were truncated
        assert batch["labels"].shape[1] == trunc_target  # Truncated
        assert max_len_target > trunc_target  # Truncated
        break  # No need to test every batch


def test_pack_dataset():
    tokenizer = AutoTokenizer.from_pretrained("facebook/mbart-large-cc25")

    tmp_dir = Path(make_test_data_dir())
    orig_examples = tmp_dir.joinpath("train.source").open().readlines()
    save_dir = Path(tempfile.mkdtemp(prefix="packed_"))
    pack_data_dir(tokenizer, tmp_dir, 128, save_dir)
    orig_paths = {x.name for x in tmp_dir.iterdir()}
    new_paths = {x.name for x in save_dir.iterdir()}
    packed_examples = save_dir.joinpath("train.source").open().readlines()
    # orig: [' Sam ate lunch today.\n', 'Sams lunch ingredients.']
    # desired_packed: [' Sam ate lunch today.\n Sams lunch ingredients.']
    assert len(packed_examples) < len(orig_examples)
    assert len(packed_examples) == 1
    assert len(packed_examples[0]) == sum(len(x) for x in orig_examples)
    assert orig_paths == new_paths


@pytest.mark.skipif(not FAIRSEQ_AVAILABLE, reason="This test requires fairseq")
def test_dynamic_batch_size():
    if not FAIRSEQ_AVAILABLE:
        return
    ds, max_tokens, tokenizer = _get_dataset(max_len=64)
    required_batch_size_multiple = 64
    batch_sampler = ds.make_dynamic_sampler(max_tokens, required_batch_size_multiple=required_batch_size_multiple)
    batch_sizes = [len(x) for x in batch_sampler]
    assert len(set(batch_sizes)) > 1  # it's not dynamic batch size if every batch is the same length
    assert sum(batch_sizes) == len(ds)  # no dropped or added examples
    data_loader = DataLoader(ds, batch_sampler=batch_sampler, collate_fn=ds.collate_fn, num_workers=2)
    failures = []
    num_src_per_batch = []
    for batch in data_loader:
        src_shape = batch["input_ids"].shape
        bs = src_shape[0]
        assert bs % required_batch_size_multiple == 0 or bs < required_batch_size_multiple
        num_src_tokens = np.product(batch["input_ids"].shape)
        num_src_per_batch.append(num_src_tokens)
        if num_src_tokens > (max_tokens * 1.1):
            failures.append(num_src_tokens)
    assert num_src_per_batch[0] == max(num_src_per_batch)
    if failures:
        raise AssertionError(f"too many tokens in {len(failures)} batches")


def test_sortish_sampler_reduces_padding():
    ds, _, tokenizer = _get_dataset(max_len=512)
    bs = 2
    sortish_sampler = ds.make_sortish_sampler(bs, shuffle=False)

    naive_dl = DataLoader(ds, batch_size=bs, collate_fn=ds.collate_fn, num_workers=2)
    sortish_dl = DataLoader(ds, batch_size=bs, collate_fn=ds.collate_fn, num_workers=2, sampler=sortish_sampler)

    pad = tokenizer.pad_token_id

    def count_pad_tokens(data_loader, k="input_ids"):
        return [batch[k].eq(pad).sum().item() for batch in data_loader]

    assert sum(count_pad_tokens(sortish_dl, k="labels")) < sum(count_pad_tokens(naive_dl, k="labels"))
    assert sum(count_pad_tokens(sortish_dl)) < sum(count_pad_tokens(naive_dl))
    assert len(sortish_dl) == len(naive_dl)


def _get_dataset(n_obs=1000, max_len=128):
    if os.getenv("USE_REAL_DATA", False):
        data_dir = "examples/seq2seq/wmt_en_ro"
        max_tokens = max_len * 2 * 64
        if not Path(data_dir).joinpath("train.len").exists():
            save_len_file(MARIAN_TINY, data_dir)
    else:
        data_dir = "examples/seq2seq/test_data/wmt_en_ro"
        max_tokens = max_len * 4
        save_len_file(MARIAN_TINY, data_dir)

    tokenizer = AutoTokenizer.from_pretrained(MARIAN_TINY)
    ds = Seq2SeqDataset(
        tokenizer,
        data_dir=data_dir,
        type_path="train",
        max_source_length=max_len,
        max_target_length=max_len,
        n_obs=n_obs,
    )
    return ds, max_tokens, tokenizer


def test_distributed_sortish_sampler_splits_indices_between_procs():
    ds, max_tokens, tokenizer = _get_dataset()
    ids1 = set(DistributedSortishSampler(ds, 256, num_replicas=2, rank=0, add_extra_examples=False))
    ids2 = set(DistributedSortishSampler(ds, 256, num_replicas=2, rank=1, add_extra_examples=False))
    assert ids1.intersection(ids2) == set()


@pytest.mark.parametrize(
    "tok_name",
    [
        MBART_TINY,
        MARIAN_TINY,
        T5_TINY,
        BART_TINY,
        PEGASUS_XSUM,
    ],
)
def test_dataset_kwargs_for_t5_tokenizer(tok_name):
    tokenizer = AutoTokenizer.from_pretrained(tok_name)
    if tok_name == MBART_TINY:
        train_dataset = Seq2SeqDataset(
            tokenizer,
            data_dir=make_test_data_dir(),
            type_path="train",
            max_source_length=4,
            max_target_length=8,
            src_lang='EN',
            tgt_lang='FR'
        )
        kwargs = train_dataset.dataset_kwargs
        assert 'src_lang' in kwargs and 'tgt_lang' in kwargs
    else:
        train_dataset = Seq2SeqDataset(
            tokenizer,
            data_dir=make_test_data_dir(),
            type_path="train",
            max_source_length=4,
            max_target_length=8
        )
        kwargs = train_dataset.dataset_kwargs
        assert 'add_prefix_space' not in  kwargs if tok_name != BART_TINY else 'add_prefix_space' in kwargs
        assert len(kwargs) == 1 if tok_name == BART_TINY else len(kwargs) == 0


