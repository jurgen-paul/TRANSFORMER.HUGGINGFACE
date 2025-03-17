# tokenization_arlow.py
from typing import Optional, Tuple

from transformers.tokenization_utils import AddedToken
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast
from transformers.utils import logging


logger = logging.get_logger(__name__)

VOCAB_FILES_NAMES = {
    "vocab_file": "vocab.json",
    "merges_file": "merges.txt",
    "tokenizer_file": "tokenizer.json",  # If you have a full fast tokenizer JSON
}

class ArlowTokenizer(PreTrainedTokenizerFast):
    """
    Construct a "fast" Arlow tokenizer (backed by HuggingFace's *tokenizers* library).

    This tokenizer is designed for GPT-like usage with BPE merges, plus specialized tokens
    (e.g., <|startoftext|>, <|endoftext|>, <|mask|>, etc.).

    It inherits from `PreTrainedTokenizerFast`, so most methods (encode, decode, etc.)
    are available. If you provide `tokenizer_file`, it will load your full fast tokenizer
    from that JSON. Otherwise, it will fall back to `vocab_file` + `merges_file`.

    Args:
        vocab_file (str, *optional*):
            Path to the BPE vocabulary JSON file.
        merges_file (str, *optional*):
            Path to the BPE merges file.
        tokenizer_file (str, *optional*):
            Path to a *tokenizers* library JSON file (usually named "tokenizer.json").
        bos_token (str, *optional*, defaults to "<|startoftext|>"):
            Beginning-of-sequence token.
        eos_token (str, *optional*, defaults to "<|endoftext|>"):
            End-of-sequence token.
        unk_token (str, *optional*, defaults to "<|unk|>"):
            Unknown token.
        pad_token (str, *optional*, defaults to "<|pad|>"):
            Padding token.
        mask_token (str, *optional*, defaults to "<|mask|>"):
            Mask token for masked language modeling.
        additional_special_tokens (list of str, *optional*):
            Additional special tokens, e.g. ["<|im_start|>", "<|im_end|>"].
    """

    vocab_files_names = VOCAB_FILES_NAMES
    # This indicates there's no slow (Python-based) tokenizer available:
    slow_tokenizer_class = None
    # Typical model inputs for a GPT-like tokenizer:
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(
        self,
        vocab_file=None,
        merges_file=None,
        tokenizer_file=None,
        bos_token="<|startoftext|>",
        eos_token="<|endoftext|>",
        unk_token="<|unk|>",
        pad_token="<|pad|>",
        mask_token="<|mask|>",
        additional_special_tokens=None,
        **kwargs,
    ):
        # Convert str tokens to `AddedToken` for consistency, if needed
        if isinstance(bos_token, str):
            bos_token = AddedToken(bos_token, special=True)
        if isinstance(eos_token, str):
            eos_token = AddedToken(eos_token, special=True)
        if isinstance(unk_token, str):
            unk_token = AddedToken(unk_token, special=True)
        if isinstance(pad_token, str):
            pad_token = AddedToken(pad_token, special=True)
        if isinstance(mask_token, str):
            mask_token = AddedToken(mask_token, special=True)

        # Pass along to PreTrainedTokenizerFast __init__.
        super().__init__(
            vocab_file=vocab_file,
            merges_file=merges_file,
            tokenizer_file=tokenizer_file,
            bos_token=bos_token,
            eos_token=eos_token,
            unk_token=unk_token,
            pad_token=pad_token,
            mask_token=mask_token,
            additional_special_tokens=additional_special_tokens,
            **kwargs,
        )

    def save_vocabulary(
        self,
        save_directory: str,
        filename_prefix: Optional[str] = None
    ) -> Tuple[str]:
        """
        Save the tokenizer vocabulary + merges to `save_directory`.
        Returns a tuple of (vocab_file_path, merges_file_path).
        """
        files = self._tokenizer.model.save(save_directory, name=filename_prefix)
        return tuple(files)
