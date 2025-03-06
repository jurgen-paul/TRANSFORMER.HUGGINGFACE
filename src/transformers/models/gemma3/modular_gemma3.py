# coding=utf-8
# Copyright 2025 Google Inc. HuggingFace Inc. team. All rights reserved.
#
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
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import itertools
import math
import re
from typing import Any, Literal, Optional, Union, cast

import PIL
import PIL.Image
import torch
import torch.nn as nn
import torch.utils.checkpoint

from ...activations import ACT2FN
from ...cache_utils import Cache, HybridCache, StaticCache
from ...configuration_utils import PretrainedConfig
from ...feature_extraction_utils import BatchFeature
from ...generation import GenerationMixin
from ...modeling_flash_attention_utils import FlashAttentionKwargs
from ...modeling_rope_utils import ROPE_INIT_FUNCTIONS
from ...modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from ...processing_utils import (
    ImagesKwargs,
    ProcessingKwargs,
    ProcessorMixin,
    TextKwargs,
    Unpack,
    _validate_images_text_input_order,
)
from ...tokenization_utils_base import (
    AddedToken,
    TextInput,
)
from ...utils import (
    ModelOutput,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_torchdynamo_compiling,
    logging,
    replace_return_docstrings,
)
from ...utils.deprecation import deprecate_kwarg
from ..gemma import GemmaPreTrainedModel, GemmaTokenizer, GemmaTokenizerFast
from ..gemma.modeling_gemma import (
    GemmaMLP,
    GemmaRMSNorm,
    GemmaRotaryEmbedding,
    apply_rotary_pos_emb,
    repeat_kv,
)
from ..gemma2.modeling_gemma2 import Gemma2DecoderLayer
from ..siglip import SiglipImageProcessor, SiglipVisionConfig, SiglipVisionModel


_CHECKPOINT_FOR_DOC = "google/gemma-3-4b"
_CONFIG_FOR_DOC = "Gemma3Config"

logger = logging.get_logger(__name__)

GEMMA3_INPUTS_DOCSTRING = ""

BatchedImageInput = Sequence[PIL.Image.Image]
BatchedMultiImageInput = Sequence[BatchedImageInput]
Gemma3ProcessorImageInput = Union[
    PIL.Image.Image, BatchedImageInput, BatchedMultiImageInput
]

PanAndScannedImage = tuple[PIL.Image.Image, Sequence[PIL.Image.Image]]
BatchedPanAndScannedImage = Sequence[Sequence[PanAndScannedImage]]
MutablePanAndScannedImage = tuple[PIL.Image.Image, list[PIL.Image.Image]]
MutableBatchedPanAndScannedImage = list[list[MutablePanAndScannedImage]]

ATTENTION_TYPE_GLOBAL = "global"
ATTENTION_TYPE_LOCAL = "local_sliding"
AttentionType = Literal["global", "local_sliding"]
AttentionPattern = Sequence[AttentionType]
DEFAULT_ATTENION_PATTERN = cast(
    AttentionPattern,
    (
        ATTENTION_TYPE_LOCAL,
        ATTENTION_TYPE_LOCAL,
        ATTENTION_TYPE_LOCAL,
        ATTENTION_TYPE_LOCAL,
        ATTENTION_TYPE_LOCAL,
        ATTENTION_TYPE_GLOBAL,
    ),
)

TextInputTypes = Union[TextInput, Sequence[TextInput]]


class Gemma3RotaryEmbeddingConfig(PretrainedConfig):

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        rope_theta: float,
        head_dim: Optional[int] = None,
        max_position_embeddings: int = 131_072,
        partial_rotary_factor: Optional[float] = None,
        rope_scaling: Optional[Mapping[str, Union[int, float]]] = None,
        torch_dtype: str = "bfloat16",
        **kwargs,
    ):
        super().__init__(torch_dtype=torch_dtype, **kwargs)
        self.hidden_size = hidden_size
        self.max_position_embeddings = max_position_embeddings
        self.num_attention_heads = num_attention_heads
        self.rope_theta = rope_theta

        if head_dim is not None:
            self.head_dim = head_dim
        if partial_rotary_factor is not None:
            self.partial_rotary_factor = partial_rotary_factor
        if rope_scaling is not None:
            self.rope_scaling = rope_scaling


class Gemma3TextConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`Gemma3Model`]. It is used to instantiate a Gemma3
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the Gemma3-7B.
    e.g. [google/gemma-3-4b](https://huggingface.co/google/gemma-3-4b)
    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        vocab_size (`int`, *optional*, defaults to 256000):
            Vocabulary size of the Gemma3 model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`Gemma3Model`]
        num_hidden_layers (`int`, *optional*, defaults to 26):
            Number of hidden layers in the Transformer decoder.
        max_position_embeddings (`int`, *optional*, defaults to 8192):
            The maximum sequence length that this model might ever be used with.
        hidden_size (`int`, *optional*, defaults to 2304):
            Dimension of the hidden representations.
        intermediate_size (`int`, *optional*, defaults to 9216):
            Dimension of the MLP representations.
        num_attention_heads (`int`, *optional*, defaults to 8):
            Number of attention heads for each attention layer in the Transformer decoder.
        num_key_value_heads (`int`, *optional*, defaults to 4):
            This is the number of key_value heads that should be used to implement Grouped Query Attention. If
            `num_key_value_heads=num_attention_heads`, the model will use Multi Head Attention (MHA), if
            `num_key_value_heads=1` the model will use Multi Query Attention (MQA) otherwise GQA is used. When
            converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed
            by meanpooling all the original heads within that group. For more details checkout [this
            paper](https://arxiv.org/pdf/2305.13245.pdf). If it is not specified, will default to
            `num_attention_heads`.
        head_dim (`int`, *optional*, defaults to 256):
            The attention head dimension.
        hidden_activation (`str` or `function`, *optional*, defaults to `"gelu_pytorch_tanh"`):
            The non-linear activation function (function or string) in the decoder. Will default to
            `"gelu_pytorch_tanh"` if not specified. `"gelu_pytorch_tanh"` uses an approximation of the `"gelu"`
            activation function.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        rms_norm_eps (`float`, *optional*, defaults to 1e-06):
            The epsilon used by the rms normalization layers.
        pad_token_id (`int`, *optional*, defaults to 0):
            Padding token id.
        eos_token_id (`int`, *optional*, defaults to 1):
            End of stream token id.
        bos_token_id (`int`, *optional*, defaults to 2):
            Beginning of stream token id.
        tie_word_embeddings (`bool`, *optional*, defaults to `True`):
            Whether to tie weight embeddings
        rope_theta (`float`, *optional*, defaults to 10000.0):
            The base period of the RoPE embeddings.
        rope_global_base_freq (float, *optional*, defaults to `rope_theta`):
            The base period of the RoPE embeddings for global attention.
        rope_local_base_freq (float, *optional*, defaults to `rope_theta`):
            The base period of the RoPE embeddings for local attention.
        attention_pattern (Sequence[AttentionTypes], defaults to (5 * local, global)):
            The attention pattern to apply
        attention_bias (`bool`, *optional*, defaults to `False`):
            Whether to use a bias in the query, key, value and output projection layers during self-attention.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        query_pre_attn_scalar (`float`, *optional*, defaults to None):
            The scaling factor used on the attention scores, not that
        sliding_window (`int`, *optional*, defaults to 4096): in Gemma3, every other layer uses sliding window
            attention. This is the size of the sliding window.
        attn_logit_softcapping (`float`, *optional*, defaults to 50.0): scaling factor when applying tanh soft-capping
            on the attention scorexs.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.
        cache_implementation (`str`, *optional*, defaults to `"hybrid"`): the cache type to be used with `generate`.

    ```python
    >>> from transformers import Gemma3Model, Gemma3Config
    >>> # Initializing a Gemma3 gemma3-7b style configuration
    >>> configuration = Gemma3Config()
    >>> # Initializing a model from the gemma3-7b style configuration
    >>> model = Gemma3Model(configuration)
    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "gemma3_text"
    keys_to_ignore_at_inference = ["past_key_values"]
    base_model_tp_plan = {
        "layers.*.self_attn.q_proj": "colwise",
        "layers.*.self_attn.k_proj": "colwise",
        "layers.*.self_attn.v_proj": "colwise",
        "layers.*.self_attn.o_proj": "rowwise",
        "layers.*.mlp.gate_proj": "colwise",
        "layers.*.mlp.up_proj": "colwise",
        "layers.*.mlp.down_proj": "rowwise",
    }

    def __init__(
        self,
        # Config parameters found in all implementations, name differences noted
        vocab_size: int = 262_144,  # num_embed in FLAX
        mm_tokens_per_image: int = 256,
        hidden_size: int = 2304,  # embed_dim in FLAX
        intermediate_size: int = 9216,  # hidden_dim in FLAX
        num_hidden_layers: int = 26,  # num_layers in FLAX
        num_attention_heads: int = 8,  # num_heads in FLAX
        num_key_value_heads: int = 4,  # num_kv_heads in FLAX
        head_dim: int = 256,
        sliding_window: int = 4096,  # sliding_window_size in FLAX
        query_pre_attn_scalar: Optional[float] = None,
        attention_pattern: AttentionPattern = DEFAULT_ATTENION_PATTERN,
        rope_global_base_freq: float = 1_000_000.0,
        rope_local_base_freq: float = 10_000.0,
        rms_norm_eps: float = 1e-6,
        hidden_activation: str = "gelu_pytorch_tanh",
        pad_token_id: int = 0,
        eos_token_id: int = 1,
        bos_token_id: int = 2,
        boi_token_id: int = 255_999,
        eoi_token_id: int = 256_000,
        image_token_id: int = 262_144,
        tie_word_embeddings: bool = True,
        max_position_embeddings: int = 131_072,
        initializer_range: float = 0.02,
        attention_bias: bool = False,
        attention_dropout: float = 0.0,
        use_cache: bool = True,
        cache_implementation: str = "hybrid",
        torch_dtype: str = "bfloat16",
        **kwargs,
    ):
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            torch_dtype=torch_dtype,
            **kwargs,
        )

        self.boi_token_id = boi_token_id
        self.eoi_token_id = eoi_token_id
        self.image_token_id = image_token_id
        self.vocab_size = vocab_size
        self.mm_tokens_per_image = mm_tokens_per_image
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.head_dim = head_dim
        self.num_key_value_heads = num_key_value_heads
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_global_base_freq = rope_global_base_freq
        self.rope_local_base_freq = rope_local_base_freq
        self.attention_pattern = attention_pattern
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.hidden_activation = hidden_activation
        self.query_pre_attn_scalar = query_pre_attn_scalar
        self.sliding_window = sliding_window
        self.cache_implementation = cache_implementation


class Gemma3VisionConfig(SiglipVisionConfig):

    model_type = "gemma3_vision"

    def __init__(
        self,
        hidden_size: int = 1152,
        intermediate_size: int = 4304,
        num_hidden_layers: int = 27,
        num_attention_heads: int = 16,
        num_channels: int = 3,
        image_size: int = 896,
        attention_dropout: float = 0.0,
        patch_size: int = 14,
        hidden_act: str = "gelu_pytorch_tanh",
        layer_norm_eps: float = 0.000001,
        vision_use_head: bool = False,
        torch_dtype: str = "bfloat16",
        **kwargs,
    ):
        super().__init__(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            num_channels=num_channels,
            image_size=image_size,
            patch_size=patch_size,
            hidden_act=hidden_act,
            layer_norm_eps=layer_norm_eps,
            attention_dropout=attention_dropout,
            torch_dtype=torch_dtype,
            **kwargs,
        )

        self.vision_use_head = vision_use_head


class Gemma3Config(PretrainedConfig):
    model_type = "gemma3"
    sub_configs = {
        "text_config": Gemma3TextConfig,
        "vision_config": Gemma3VisionConfig,
    }

    def __init__(
        self,
        text_config: Optional[Gemma3TextConfig] = None,
        vision_config: Optional[Gemma3VisionConfig] = None,
        torch_dtype: str = "bfloat16",
        **kwargs,
    ):
        if text_config is None:
            self.text_config = Gemma3TextConfig()
            logger.info(
                "text_config is None, using default Gemma3TextConfig vision config."
            )
        elif isinstance(text_config, dict):
            self.text_config = Gemma3TextConfig(**text_config)
        elif isinstance(text_config, Gemma3TextConfig):
            self.text_config = text_config
        else:
            raise ValueError(
                "text_config must be None or compatible with initializing a Gemma3TextConfig."
            )

        if isinstance(vision_config, dict):
            self.vision_config = Gemma3VisionConfig(**vision_config)
        elif isinstance(vision_config, Gemma3VisionConfig):
            self.vision_config = cast(Gemma3VisionConfig, vision_config)
        else:
            self.vision_config = None
            logger.info(
                "vision_config is None or incompatible with Gemma3VisionConfig intialization. Gemma3 will be limited "
                "to text tasks."
            )

        super().__init__(torch_dtype=torch_dtype, **kwargs)


class Gemma3TextKwargs(TextKwargs):
    pass


class Gemma3ImagesKwargs(ImagesKwargs):
    do_pan_and_scan: Optional[bool]
    pan_and_scan_min_crop_size: Optional[int]
    pan_and_scan_max_num_crops: Optional[int]
    pan_and_scan_min_ratio_to_activate: Optional[float]
    do_convert_rgb: Optional[bool]


class Gemma3ProcessorKwargs(ProcessingKwargs, total=False):
    text_kwargs: Gemma3TextKwargs
    images_kwargs: Gemma3ImagesKwargs
    _defaults = {
        "text_kwargs": {
            "padding": False,
        },
        "images_kwargs": {
            "data_format": "channels_first",
            "do_pan_and_scan": False,
            "pan_and_scan_min_crop_size": 256,
            "pan_and_scan_max_num_crops": 4,
            "pan_and_scan_min_ratio_to_activate": 1.2,
            "do_resize": True,
            "size": {"height": 896, "width": 896},
            "resample": PIL.Image.Resampling.BICUBIC,
            "do_rescale": True,
            "rescale_factor": 1 / 255,
            "do_normalize": True,
            "image_mean": (127.5,) * 3,
            "image_std": (127.5,) * 3,
            "do_convert_rgb": None,
        },
    }


def pan_and_scan(
    image: PIL.Image.Image,
    pan_and_scan_min_crop_size: int,
    pan_and_scan_max_num_crops: int,
    pan_and_scan_min_ratio_to_activate: float,
    **unused_kwargs,
) -> Sequence[PIL.Image.Image]:
    w, h = image.size

    # Square or landscape image.
    if w >= h:
        # Only apply PaS if the image is sufficiently exaggerated
        if w / h < pan_and_scan_min_ratio_to_activate:
            return []

        # Select ideal number of crops close to the image aspect ratio and such that crop_size > min_crop_size.
        num_crops_w = int(math.floor(w / h + 0.5))  # Half round up rounding.
        num_crops_w = min(int(math.floor(w / pan_and_scan_min_crop_size)), num_crops_w)

        # Make sure the number of crops is in range [2, pan_and_scan_max_num_crops].
        num_crops_w = max(2, num_crops_w)
        num_crops_w = min(pan_and_scan_max_num_crops, num_crops_w)
        num_crops_h = 1

    # Portrait image.
    else:
        # Only apply PaS if the image is sufficiently exaggerated
        if h / w < pan_and_scan_min_ratio_to_activate:
            return []

        # Select ideal number of crops close to the image aspect ratio and such that crop_size > min_crop_size.
        num_crops_h = int(math.floor(h / w + 0.5))
        num_crops_h = min(int(math.floor(h / pan_and_scan_min_crop_size)), num_crops_h)

        # Make sure the number of crops is in range [2, pan_and_scan_max_num_crops].
        num_crops_h = max(2, num_crops_h)
        num_crops_h = min(pan_and_scan_max_num_crops, num_crops_h)
        num_crops_w = 1

    crop_size_w = int(math.ceil(w / num_crops_w))
    crop_size_h = int(math.ceil(h / num_crops_h))

    # Don't apply PaS if crop size is too small.
    if min(crop_size_w, crop_size_h) < pan_and_scan_min_crop_size:
        return []

    crop_positions_w = [crop_size_w * i for i in range(num_crops_w)]
    crop_positions_h = [crop_size_h * i for i in range(num_crops_h)]

    # Generate crops.
    return [
        image.crop((pos_w, pos_h, pos_w + crop_size_w, pos_h + crop_size_h))
        for pos_h, pos_w in itertools.product(crop_positions_h, crop_positions_w)
    ]


class Gemma3Processor(ProcessorMixin):

    attributes = ["image_processor", "tokenizer"]
    valid_kwargs = ["chat_template"]
    image_processor_class = "SiglipImageProcessor"
    tokenizer_class = ("GemmaTokenizer", "GemmaTokenizerFast")

    def __init__(
        self,
        image_processor: SiglipImageProcessor,
        tokenizer: Union[GemmaTokenizer, GemmaTokenizerFast],
        chat_template: Optional[str] = None,
        num_mm_soft_tokens_per_image: int = 256,
        **kwargs,
    ):
        try:
            self.image_seq_length = getattr(image_processor, "image_seq_length")
        except AttributeError as e:
            raise ValueError(
                "`image_processor` is missing the required `image_seq_length` attribute."
            ) from e

        self.image_token_id = tokenizer.convert_tokens_to_ids("<image_soft_token>")
        self.full_image_sequence = (
            "\n\n<start_of_image>" +
            "".join(["<image_soft_token>"] * num_mm_soft_tokens_per_image) +
            "<end_of_image>\n\n"
        )

        super().__init__(
            image_processor=image_processor,
            tokenizer=tokenizer,
            chat_template=chat_template,
            **kwargs,
        )

    def __call__(
        self,
        images: Optional[Gemma3ProcessorImageInput] = None,
        text: Optional[TextInputTypes] = None,
        videos: Optional[Any] = None,
        audio: Optional[Any] = None,
        **kwargs: Unpack[Gemma3ProcessorKwargs],
    ) -> BatchFeature:
        del videos, audio   # Unsupported modalities for Gemma 3

        if text is None and images is None:
            raise ValueError("Provide at least one of `text` or `images`.")

        # Check if images and text inputs are reversed for backward compatibility
        images, text = _validate_images_text_input_order(images, text)

        output_kwargs = self._merge_kwargs(
            Gemma3ProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )

        if images is not None:
            batched_images = self._process_images(
                images=images, **output_kwargs["images_kwargs"]
            )
            batch_flattened_images = self._batch_flatten_pas_images(batched_images=batched_images)
            # The Hugging Face implementation of the SigLIP Vision Model expects a single Tensor of shape [F, C, W, H]
            # where:
            #
            # - F is the number of images to encode
            # - C is the number of channels in each image
            # - W and H are the width and height of the image, which in this case are the same since Gemma 3 only
            #   supports 896x896 images.
            #
            # So we concat all images across all batches into a single flat list prior to sending it to the
            # `Gemma3ForConditionalGeneration.vision_model` for ecnoding and use `torch.masked_scatter()` in that the
            # `Gemma3ForConditionalGeneration` model class to sequentially update the language model embeddings wth the
            # pooled vision embdeddings.
            pixel_values = torch.cat([
                self.image_processor(prompt_images, **output_kwargs["images_kwargs"])["pixel_values"]
                for prompt_images in batch_flattened_images
            ])
        else:
            batched_images = None
            pixel_values = None

        batched_input = self._process_text(
            text=text, batched_images=batched_images, **output_kwargs["text_kwargs"]
        )

        if pixel_values is not None:
            batched_input.update(pixel_values=pixel_values)

        return batched_input

    def _process_images(
        self, images: Gemma3ProcessorImageInput, **kwargs: Unpack[Gemma3ImagesKwargs]
    ) -> BatchedPanAndScannedImage:
        if isinstance(images, PIL.Image.Image):
            images_lists: MutableBatchedPanAndScannedImage = [[(images, [])]]
        elif isinstance(images[0], PIL.Image.Image):
            images = cast(BatchedImageInput, images)
            images_lists: MutableBatchedPanAndScannedImage = [[(i, [])] for i in images]
        else:
            images = cast(BatchedMultiImageInput, images)
            images_lists: MutableBatchedPanAndScannedImage = [
                [(i, []) for i in il] for il in images
            ]

        if getattr(kwargs, "do_pan_and_scan", False):
            for images_list in images_lists:
                for image, crops in images_list:
                    crops.extend(pan_and_scan(image=image, **kwargs))

        return images_lists

    def _process_text(
        self,
        text: Optional[TextInputTypes] = None,
        batched_images: Optional[BatchedPanAndScannedImage] = None,
        **kwargs: Unpack[Gemma3TextKwargs],
    ) -> BatchFeature:
        if batched_images and not text:
            text = [
                " ".join(["<image>"] * len(images))
                for images in batched_images
            ]

        if batched_images and text:
            if isinstance(text, str):
                text = [text]

            if (bi_l := len(batched_images)) != (t_l := len(text)):
                raise ValueError(
                    f"Received inconsistently sized batches of images ({bi_l}) and text ({t_l})."
                )

            for prompt, images in zip(text, batched_images):
                image_indexes = [m.start() for m in re.finditer("<image>", prompt)]

                if (i_l := len(images)) != (idx_l := len(image_indexes)):
                    raise ValueError(f"Prompt contained {idx_l} image tokens but received {i_l} images.")

                # Insert additional image tokens for Pan-and-Scan crops
                for (_, pas_images), idx in reversed(list(zip(images, image_indexes))):
                    if pas_images:
                        formatted_image_text = (
                            "Here is the original image <image> and here are some crops to help you see better " +
                            " ".join(["<image>"] * len(pas_images))
                        )
                        prompt = prompt[:idx] + formatted_image_text + prompt[idx + len("<image>") :]

            # Expand placeholder image tokens to the full image token sequence
            text = [
                prompt.replace("<image>", self.full_image_sequence)
                for prompt in text
            ]

        inputs = self.tokenizer(text=text, **kwargs)
        return BatchFeature({**inputs})

    def _batch_flatten_pas_images(
        self, batched_images: BatchedPanAndScannedImage,
    ) -> Sequence[Sequence[PIL.Image.Image]]:
        """Converts the Sequence[tuple[Image, Sequence[Image]]] into a Sequence[Image]"""
        batch_flattened: list[list[PIL.Image.Image]] = []

        for images in batched_images:
            prompt_flattened: list[PIL.Image.Image] = []
            for (image, pas_images) in images:
                prompt_flattened.extend([image] + pas_images)
            batch_flattened.append(prompt_flattened)

        return batch_flattened


    # Copied from transformers.models.clip.processing_clip.CLIPProcessor.batch_decode with CLIP->Gemma
    def batch_decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to GemmaTokenizerFast's [`~PreTrainedTokenizer.batch_decode`]. Please
        refer to the docstring of this method for more information.
        """
        return self.tokenizer.batch_decode(*args, **kwargs)

    # Copied from transformers.models.clip.processing_clip.CLIPProcessor.decode with CLIP->Gemma
    def decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to GemmaTokenizerFast's [`~PreTrainedTokenizer.decode`]. Please refer to
        the docstring of this method for more information.
        """
        return self.tokenizer.decode(*args, **kwargs)

    @property
    # Copied from transformers.models.clip.processing_clip.CLIPProcessor.model_input_names with CLIP->PaliGemma
    def model_input_names(self):
        tokenizer_input_names = self.tokenizer.model_input_names
        image_processor_input_names = self.image_processor.model_input_names
        return list(dict.fromkeys(tokenizer_input_names + image_processor_input_names))


class Gemma3RMSNorm(GemmaRMSNorm):
    pass


class Gemma3MultimodalInputProjection(nn.Module):

    def __init__(self, config: Gemma3Config):
        super().__init__()

        self.mm_input_projection_weight = nn.Parameter(torch.zeros(
            config.vision_config.hidden_size, config.text_config.hidden_size
        ))

        self.mm_soft_emb_norm = Gemma3RMSNorm(
            config.vision_config.hidden_size, eps=config.vision_config.layer_norm_eps
        )

        self.patches_per_image = int(config.vision_config.image_size // config.vision_config.patch_size)
        self.tokens_per_side = int(config.text_config.mm_tokens_per_image ** 0.5)
        self.kernel_size = self.patches_per_image // self.tokens_per_side
        self.avg_pool = nn.AvgPool2d(kernel_size=self.kernel_size, stride=self.kernel_size)

    def forward(self, vision_outputs: torch.Tensor):
        b, _, l = vision_outputs.shape

        reshaped_vision_outputs = vision_outputs.transpose(1, 2)
        reshaped_vision_outputs = reshaped_vision_outputs.reshape(
            b, l, self.patches_per_image, self.patches_per_image
        )
        reshaped_vision_outputs = reshaped_vision_outputs.contiguous()

        pooled_vision_outputs = self.avg_pool(reshaped_vision_outputs)
        pooled_vision_outputs = pooled_vision_outputs.flatten(2)
        pooled_vision_outputs = pooled_vision_outputs.transpose(1, 2)

        normed_vision_outputs = self.mm_soft_emb_norm(pooled_vision_outputs)

        projected_vision_outputs = torch.einsum(
            'btm,md->btd', normed_vision_outputs, self.mm_input_projection_weight
        )
        return projected_vision_outputs.type_as(vision_outputs)


class Gemma3MLP(GemmaMLP):

    def __init__(self, config: Gemma3TextConfig):
        super().__init__(config)
        self.act_fn = ACT2FN[config.hidden_activation]


def create_sliding_window_mask(
    position_ids: torch.LongTensor,
    cache_position: int,
    cache_len: int,
    sliding_window_size: int,
) -> torch.Tensor:
    """Creates mask for sliding window attention."""
    total_tokens = cache_position + position_ids.shape[1]  # cached + processing tokens

    def _reconstruct_rotated_cache_positions():
        cache_positions = torch.arange(cache_len) + total_tokens - cache_len
        rotated_cache_positions = torch.zeros_like(cache_positions)
        rotated_cache_positions[cache_positions % cache_len] = cache_positions
        return rotated_cache_positions

    # Reconstruct position_ids for cached kv.
    if total_tokens <= cache_len:
        cache_positions = torch.arange(cache_len)
    else:
        cache_positions = _reconstruct_rotated_cache_positions()

    cache_positions = cache_positions.unsqueeze(0).unsqueeze(0).to(position_ids.device)  # [1, 1, cache_len]
    position_ids = position_ids.unsqueeze(-1)  # [B, seq_len, 1]
    sliding_mask = cache_positions > position_ids - sliding_window_size
    sliding_mask *= cache_positions < position_ids + sliding_window_size
    return sliding_mask.unsqueeze(1)


def eager_attention_forward(
    module: "Gemma3Attention",
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    dropout: float = 0.0,
    scaling: Optional[float] = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    if scaling is None:
        scaling = module.head_dim**-0.5

    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling

    if attention_mask is not None:  # no matter the length, we just slice it
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    # upcast attention to fp32
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
        query.dtype
    )
    attn_weights = nn.functional.dropout(
        attn_weights, p=dropout, training=module.training
    )
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output, attn_weights


class Gemma3Attention(nn.Module):

    def __init__(self, config: Gemma3TextConfig, layer_idx: int):

        super().__init__()
        self.attention_dropout = config.attention_dropout
        self.attention_type: AttentionType = config.attention_pattern[
            layer_idx % len(config.attention_pattern)
        ]
        self.config = config
        self.head_dim = getattr(
            config, "head_dim", config.hidden_size // config.num_attention_heads
        )
        self.is_causal = True
        self.layer_idx = layer_idx
        self.num_key_value_groups = (
            config.num_attention_heads // config.num_key_value_heads
        )
        self.scaling = config.query_pre_attn_scalar
        self.is_sliding = self.attention_type == ATTENTION_TYPE_LOCAL
        self.sliding_window = config.sliding_window if self.is_sliding else None

        self.q_proj = nn.Linear(
            config.hidden_size,
            config.num_attention_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.k_proj = nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.v_proj = nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim,
            config.hidden_size,
            bias=config.attention_bias,
        )
        self.q_norm = Gemma3RMSNorm(dim=config.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Gemma3RMSNorm(dim=config.head_dim, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings_global: torch.Tensor,
        position_embeddings_local: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        last_cache_position: int = 0,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[tuple[torch.Tensor]]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        query_states = self.q_norm(query_states)
        key_states = self.k_norm(key_states)

        if self.attention_type == ATTENTION_TYPE_GLOBAL:
            cos, sin = position_embeddings_global
        else:
            cos, sin = position_embeddings_local

        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin
        )

        if (
            self.sliding_window is not None
            and attention_mask is not None
            and position_ids is not None
        ):
            sliding_mask = create_sliding_window_mask(
                position_ids=position_ids,
                cache_position=last_cache_position,
                cache_len=attention_mask.shape[-1],
                sliding_window_size=self.sliding_window,
            )
            attention_mask *= sliding_mask

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {
                "sin": sin,
                "cos": cos,
                # Why is this a torch.LongTensor? Feels like last_cache_position should be in here somehow?
                "cache_position": cache_position,
                "sliding_window": self.sliding_window,
            }
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )

            # Here we need to slice as we use a static cache by default, but FA2 does not support it
            if (
                attention_mask is not None
                and self.config._attn_implementation == "flash_attention_2"
            ):
                seq_len = attention_mask.shape[-1]
                key_states, value_states = (
                    key_states[:, :, :seq_len, :],
                    value_states[:, :, :seq_len, :],
                )

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            if self.config._attn_implementation == "sdpa" and kwargs.get(
                "output_attentions", False
            ):
                logger.warning_once(
                    "`torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. "
                    "Falling back to eager attention. This warning can be removed using the argument "
                    '`attn_implementation="eager"` when loading the model.'
                )
            else:
                attention_interface = cast(
                    Callable[..., tuple[torch.Tensor, torch.Tensor]],
                    ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation],
                )

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=self.attention_dropout if self.training else 0.0,
            scaling=self.scaling,
            sliding_window=self.sliding_window,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class Gemma3DecoderLayer(nn.Module):

    def __init__(self, config: Gemma3TextConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx
        self.self_attn = Gemma3Attention(config=config, layer_idx=layer_idx)
        self.mlp = Gemma3MLP(config)
        self.input_layernorm = Gemma3RMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Gemma3RMSNorm(
            self.hidden_size, eps=config.rms_norm_eps
        )
        self.pre_feedforward_layernorm = Gemma3RMSNorm(
            self.hidden_size, eps=config.rms_norm_eps
        )
        self.post_feedforward_layernorm = Gemma3RMSNorm(
            self.hidden_size, eps=config.rms_norm_eps
        )
        self.is_sliding = self.self_attn.is_sliding
        self.sliding_window = config.sliding_window

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings_global: torch.Tensor,
        position_embeddings_local: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        last_cache_position: int = 0,
        **kwargs,
    ) -> tuple[
        torch.FloatTensor, Optional[tuple[torch.FloatTensor, torch.FloatTensor]]
    ]:
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings_global=position_embeddings_global,
            position_embeddings_local=position_embeddings_local,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            last_cache_position=last_cache_position,
            **kwargs,
        )
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.pre_feedforward_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.post_feedforward_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        return outputs


class Gemma3RotaryEmbedding(nn.Module):

    def __init__(
        self, config: Gemma3RotaryEmbeddingConfig, device: torch.device = None
    ):
        super().__init__()
        # BC: "rope_type" was originally "type"
        if (rope_scaling := getattr(config, "rope_scaling", None)) is not None:
            self.rope_type = rope_scaling.get("rope_type", rope_scaling.get("type"))
        else:
            self.rope_type = "default"

        self.config = config
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings
        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]

        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

    def _dynamic_frequency_update(self, position_ids, device):
        """
        dynamic RoPE layers should recompute `inv_freq` in the following situations:
        1 - growing beyond the cached sequence length (allow scaling)
        2 - the current sequence length is in the original scale (avoid losing precision with small sequences)
        """
        seq_len = torch.max(position_ids) + 1
        if seq_len > self.max_seq_len_cached:  # growth
            inv_freq, self.attention_scaling = self.rope_init_fn(
                self.config, device, seq_len=seq_len
            )
            self.register_buffer(
                "inv_freq", inv_freq, persistent=False
            )  # TODO joao: may break with compilation
            self.max_seq_len_cached = seq_len

        if (
            seq_len < self.original_max_seq_len
            and self.max_seq_len_cached > self.original_max_seq_len
        ):  # reset
            # This .to() is needed if the model has been moved to a device after being initialized (because
            # the buffer is automatically moved, but not the original copy)
            self.original_inv_freq = self.original_inv_freq.to(device)
            self.register_buffer("inv_freq", self.original_inv_freq, persistent=False)
            self.max_seq_len_cached = self.original_max_seq_len

    @torch.no_grad()
    def forward(self, x, position_ids):
        if "dynamic" in self.rope_type:
            self._dynamic_frequency_update(position_ids, device=x.device)

        # Core RoPE block
        inv_freq_expanded = (
            self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        )
        position_ids_expanded = position_ids[:, None, :].float()
        # Force float32 (see https://github.com/huggingface/transformers/pull/29285)
        device_type = x.device.type
        device_type = (
            device_type
            if isinstance(device_type, str) and device_type != "mps"
            else "cpu"
        )
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (
                inv_freq_expanded.float() @ position_ids_expanded.float()
            ).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()

        # Advanced RoPE types (e.g. yarn) apply a post-processing scaling factor, equivalent to scaling attention
        cos = cos * self.attention_scaling
        sin = sin * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class Gemma3PreTrainedModel(GemmaPreTrainedModel):
    base_model_prefix = "model"
    config_class = Gemma3TextConfig
    supports_gradient_checkpointing = True
    _no_split_modules = ["Gemma3DecoderLayer"]


class Gemma3Model(Gemma3PreTrainedModel):

    def __init__(self, config: Gemma3TextConfig):
        super().__init__(config)
        self.config = config

        self.embed_tokens = nn.Embedding(
            config.vocab_size,
            config.hidden_size,
            padding_idx=config.pad_token_id,
        )
        self.layers = nn.ModuleList(
            [
                Gemma3DecoderLayer(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = Gemma3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb_global = Gemma3RotaryEmbedding(
            config=Gemma3RotaryEmbeddingConfig(
                hidden_size=config.hidden_size,
                num_attention_heads=config.num_attention_heads,
                rope_theta=config.rope_global_base_freq,
                head_dim=config.head_dim,
                max_position_embeddings=config.max_position_embeddings,
            )
        )
        self.rotary_emb_local = Gemma3RotaryEmbedding(
            config=Gemma3RotaryEmbeddingConfig(
                hidden_size=config.hidden_size,
                num_attention_heads=config.num_attention_heads,
                rope_theta=config.rope_local_base_freq,
                head_dim=config.head_dim,
                max_position_embeddings=config.max_position_embeddings,
            )
        )
        self.gradient_checkpointing = False
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[HybridCache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        last_cache_position: Optional[int] = None,
        **flash_attn_kwargs: Unpack[FlashAttentionKwargs],
    ) -> Union[tuple, BaseModelOutputWithPast]:
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You must specify exactly one of input_ids or inputs_embeds"
            )

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None and not self.training:
            batch_size, seq_len, _ = inputs_embeds.shape
            past_key_values = HybridCache(
                self.config,
                max_batch_size=batch_size,
                max_cache_len=seq_len,
                dtype=inputs_embeds.dtype,
            )

        if cache_position is None:
            past_seen_tokens = (
                past_key_values.get_seq_length() if past_key_values is not None else 0
            )
            cache_position = torch.arange(
                past_seen_tokens,
                past_seen_tokens + inputs_embeds.shape[1],
                device=inputs_embeds.device,
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        # This is needed to correctly slice the mask without data-dependent slicing later on if using dynamo tracing
        # (retrieving the same value from `cache_position` later on would crash dynamo)
        if last_cache_position is None:
            last_cache_position = 0
            if attention_mask is not None:
                # In case a 4d mask is passed directly without using `generate`, we have to rely on cache_position
                # It will break dynamo tracing but there are no way around it (and it should never happen in practice)
                last_cache_position = (
                    attention_mask.shape[-1]
                    if attention_mask.dim() == 2
                    else cache_position[-1].item()
                )
        causal_mask = self._update_causal_mask(
            attention_mask,
            inputs_embeds,
            cache_position,
            past_key_values,
            output_attentions,
        )

        # embed positions
        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings_global = self.rotary_emb_global(hidden_states, position_ids)
        position_embeddings_local = self.rotary_emb_local(hidden_states, position_ids)

        # normalized
        # Gemma3 downcasts the below to float16, causing sqrt(3072)=55.4256 to become 55.5
        # See https://github.com/huggingface/transformers/pull/29402
        normalizer = torch.tensor(
            self.config.hidden_size**0.5, dtype=hidden_states.dtype
        )
        hidden_states = hidden_states * normalizer

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    position_embeddings_global,
                    position_embeddings_local,
                    causal_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                    cache_position,
                    last_cache_position,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    position_embeddings_global=position_embeddings_global,
                    position_embeddings_local=position_embeddings_local,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    last_cache_position=last_cache_position,
                    **flash_attn_kwargs,
                )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        output = BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )
        return output if return_dict else output.to_tuple()

    @torch.no_grad()
    def _update_causal_mask(
        self,
        attention_mask: torch.Tensor,
        input_tensor: torch.Tensor,
        cache_position: torch.Tensor,
        past_key_values: HybridCache,
        output_attentions: bool,
    ):
        # Flash Attention currently doesn't support static cache but Gemma3 work only with static cache.
        # So we will pass in attention mask as is in any case, not only when ther's padding. Then we'll use its shape
        # to cut out keys/values trailing 0 used in static cache. This workaround should be compile compatible
        # as it doesn't cause dynamic control issues.
        if self.config._attn_implementation == "flash_attention_2":
            return attention_mask

        dtype, device = input_tensor.dtype, input_tensor.device
        sequence_length = input_tensor.shape[1]
        if isinstance(past_key_values, HybridCache):
            target_length = past_key_values.get_max_cache_shape()
        else:
            target_length = (
                attention_mask.shape[-1]
                if attention_mask is not None
                else input_tensor.shape[1]
            )

        # We generate a lower triangular causal mask here (4D).
        causal_mask = self._prepare_4d_causal_attention_mask_with_cache_position(
            attention_mask,
            sequence_length=sequence_length,
            target_length=target_length,
            dtype=dtype,
            device=device,
            cache_position=cache_position,
            batch_size=input_tensor.shape[0],
        )
        return causal_mask

    @staticmethod
    def _prepare_4d_causal_attention_mask_with_cache_position(
        attention_mask: torch.Tensor,
        sequence_length: int,
        target_length: int,
        dtype: torch.dtype,
        device: torch.device,
        cache_position: torch.Tensor,
        batch_size: int,
        **kwargs,
    ):
        """
        Creates a causal 4D mask of shape `(batch_size, 1, query_length, key_value_length)` from a 2D mask of shape
        `(batch_size, key_value_length)`, or if the input `attention_mask` is already 4D, do nothing.

        Args:
            attention_mask (`torch.Tensor`):
                A 2D attention mask of shape `(batch_size, key_value_length)` or a 4D attention mask of shape
                `(batch_size, 1, query_length, key_value_length)`.
            sequence_length (`int`):
                The sequence length being processed.
            target_length (`int`):
                The target length: when generating with static cache, the mask should be as long as the static cache,
                to account for the 0 padding, the part of the cache that is not filled yet.
            dtype (`torch.dtype`):
                The dtype to use for the 4D attention mask.
            device (`torch.device`):
                The device to plcae the 4D attention mask on.
            cache_position (`torch.Tensor`):
                Indices depicting the position of the input sequence tokens in the sequence.
            batch_size (`torch.Tensor`):
                Batch size.
        """
        if attention_mask is not None and attention_mask.dim() == 4 and attention_mask.shape[2] == 1:
            # In this case that the mask comes already in inverted form and requires no inversion or slicing.
            causal_mask = attention_mask
        else:
            min_dtype = torch.finfo(dtype).min
            causal_mask = torch.full(
                (sequence_length, target_length),
                fill_value=min_dtype,
                dtype=dtype,
                device=device,
            )
            if sequence_length != 1:
                causal_mask = torch.triu(causal_mask, diagonal=1)
            causal_mask *= torch.arange(
                target_length, device=device
            ) > cache_position.reshape(-1, 1)
            causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, -1, -1)
            if attention_mask is not None:
                causal_mask = (
                    causal_mask.clone()
                )  # copy to contiguous memory for in-place edit
                mask_length = attention_mask.shape[-1]
                padding_mask = (
                    causal_mask[:, :, :, :mask_length]
                    + attention_mask[:, None, None, :]
                )
                padding_mask = padding_mask == 0
                causal_mask[:, :, :, :mask_length] = causal_mask[
                    :, :, :, :mask_length
                ].masked_fill(padding_mask, min_dtype)
        return causal_mask


class Gemma3ForCausalLM(Gemma3PreTrainedModel, GenerationMixin):

    base_model_prefix = "language_model"
_tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config: Gemma3TextConfig):
        super().__init__(config)
        self.config = config
        self.vocab_size = config.vocab_size
        self.model = Gemma3Model(config)
        self.lm_head = nn.Linear(config.hidden_size, self.vocab_size, bias=False)
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    @deprecate_kwarg("num_logits_to_keep", version="4.50", new_name="logits_to_keep")
    @add_start_docstrings_to_model_forward(GEMMA3_INPUTS_DOCSTRING)
    @replace_return_docstrings(
        output_type=CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC
    )
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[HybridCache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **loss_kwargs,
    ) -> Union[tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

            logits_to_keep (`int` or `torch.Tensor`, *optional*):
                If an `int`, compute logits for the last `logits_to_keep` tokens. If `0`, calculate logits for all
                `input_ids` (special case). Only last token logits are needed for generation, and calculating them only for that
                token can save memory, which becomes pretty significant for long sequences or large vocabulary size.
                If a `torch.Tensor`, must be 1D corresponding to the indices to keep in the sequence length dimension.
                This is useful when using packed tensor format (single dimension for batch and sequence length).

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, GemmaForCausalLM

        >>> model = GemmaForCausalLM.from_pretrained("google/gemma-2-9b")
        >>> tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-9b")

        >>> prompt = "What is your favorite condiment?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "What is your favorite condiment?"
        ```"""

        if self.training and self.config._attn_implementation != "eager":
            logger.warning_once(
                "It is strongly recommended to train Gemma3 models with the `eager` attention implementation "
                f"instead of `{self.config._attn_implementation}`. Use `eager` with `AutoModelForCausalLM.from_pretrained('<path-to-checkpoint>', attn_implementation='eager')`."
            )
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            **loss_kwargs,
        )

        hidden_states = outputs[0]
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = (
            slice(-logits_to_keep, None)
            if isinstance(logits_to_keep, int)
            else logits_to_keep
        )
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(logits, labels, self.vocab_size, **loss_kwargs)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        position_ids=None,
        use_cache=True,
        logits_to_keep=None,
        **kwargs,
    ):
        # Overwritten: has a special cache type, `HybridCache`

        # If we have cache: let's slice `input_ids` through `cache_position`, to keep only the unprocessed tokens
        # Exception 1: when passing input_embeds, input_ids may be missing entries
        # Exception 2: some generation methods do special slicing of input_ids, so we don't need to do it here
        # Exception 3: with synced GPUs cache_position may go out of bounds, but we only want dummy token in that case.
        #              (we can't check exception 3 while compiling)
        if past_key_values is not None:
            if inputs_embeds is not None or (  # Exception 1
                is_torchdynamo_compiling() or cache_position[-1] >= input_ids.shape[1]
            ):  # Exception 3
                input_ids = input_ids[:, -cache_position.shape[0] :]
            elif (
                input_ids.shape[1] != cache_position.shape[0]
            ):  # Default case (the "else", a no op, is Exception 2)
                input_ids = input_ids[:, cache_position]
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]
                # This `clone` call is needed to avoid recapturing cuda graphs with `torch.compile`'s
                # `mode="reduce-overhead`, as otherwise the input `position_ids` would have various stride
                # during the decoding. Here, simply using `.contiguous()` is not sufficient as in the
                # batch size = 1 case, `position_ids` is already contiguous but with varying stride
                # which retriggers a capture.
                position_ids = position_ids.clone(memory_format=torch.contiguous_format)

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and cache_position[0] == 0:
            model_inputs = {"inputs_embeds": inputs_embeds, "input_ids": None}
        else:
            # The clone here is for the same reason as for `position_ids`.
            model_inputs = {
                "input_ids": input_ids.clone(memory_format=torch.contiguous_format),
                "inputs_embeds": None,
            }

        # This is needed to correctly slice the mask without data-dependent slicing later on if using dynamo tracing
        # (retrieving the same value from `cache_position` later on would crash dynamo)
        model_inputs["last_cache_position"] = (
            attention_mask.shape[-1] if attention_mask is not None else 0
        )

        if (
            isinstance(past_key_values, HybridCache)
            and attention_mask.ndim == 2
            and not self.config._attn_implementation == "flash_attention_2"
        ):
            if model_inputs["inputs_embeds"] is not None:
                batch_size, sequence_length, _ = model_inputs["inputs_embeds"].shape
                device = model_inputs["inputs_embeds"].device
            else:
                batch_size, sequence_length = model_inputs["input_ids"].shape
                device = model_inputs["input_ids"].device

            attention_mask = (
                self.model._prepare_4d_causal_attention_mask_with_cache_position(
                    attention_mask,
                    sequence_length=sequence_length,
                    target_length=past_key_values.get_max_cache_shape(),
                    dtype=self.lm_head.weight.dtype,
                    device=device,
                    cache_position=cache_position,
                    batch_size=batch_size,
                )
            )

        if logits_to_keep is not None:
            model_inputs["logits_to_keep"] = logits_to_keep

        model_inputs.update(
            {
                "position_ids": position_ids,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "use_cache": use_cache,
                "attention_mask": attention_mask,
            }
        )
        return model_inputs


@dataclass
class Gemma3CausalLMOutputWithPast(ModelOutput):
    """
    Base class for PaliGemmacausal language model (or autoregressive) outputs.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Language modeling loss (for next-token prediction).
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.text_config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or
            when `config.use_cache=True`): Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each
            tuple having 2 tensors of shape `(batch_size, num_heads, sequence_length, embed_size_per_head)`)

            Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
            `past_key_values` input) to speed up sequential decoding.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or
            when `config.output_hidden_states=True`): Tuple of `torch.FloatTensor` (one for the output of the
            embeddings, if the model has an embedding layer, + one for the output of each layer) of shape
            `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when
            `config.output_attentions=True`): Tuple of `torch.FloatTensor` (one for each layer) of shape
            `(batch_size, num_heads, sequence_length, sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        image_hidden_states (`torch.FloatTensor`, *optional*):
            A `torch.FloatTensor` of size `(batch_size, num_images, sequence_length, hidden_size)`.
            image_hidden_states of the model produced by the vision encoder after projecting last hidden state.
    """

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[Union[list[torch.FloatTensor], Cache]] = None
    hidden_states: Optional[tuple[torch.FloatTensor]] = None
    attentions: Optional[tuple[torch.FloatTensor]] = None
    image_hidden_states: Optional[torch.FloatTensor] = None


class Gemma3ForConditionalGeneration(PreTrainedModel, GenerationMixin):
    config_class = Gemma3Config
    supports_gradient_checkpointing = True

    _skip_keys_device_placement = "past_key_values"
    _supports_cache_class = True
    _supports_flash_attn_2 = True
    _supports_quantized_cache = True
    _supports_sdpa = True
    _supports_static_cache = True

    def __init__(self, config: Gemma3Config):
        super().__init__(config)

        self.config = config
        text_config = self.config.text_config
        vision_config = self.config.vision_config
        if vision_config is None:
            raise ValueError(
                "Atempted to initialize a `Gemma3ForConditionalGeneration` instance without a `Gemma3VisionConfig`; "
                "either provide a vision config or use a `Gemma3ForCausalLM` instace for text-only generation."
            )

        self.language_model = Gemma3ForCausalLM(config=text_config)
        self.vision_model = SiglipVisionModel(config=vision_config)
        self.multimodal_projector = Gemma3MultimodalInputProjection(config=config)

        self.vocab_size = text_config.vocab_size
        self.pad_token_id = (
            pad_token_id
            if (pad_token_id := text_config.pad_token_id) is not None
            else -1
        )
        self.post_init()

    # Copied from transformers.models.paligemma.modeling_paligemma.PaliGemmaForConditionalGeneration.get_input_embeddings with PaliGemma->Gema3
    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    # Copied from transformers.models.paligemma.modeling_paligemma.PaliGemmaForConditionalGeneration.set_input_embeddings with PaliGemma->Gema3
    def set_input_embeddings(self, value):
        self.language_model.set_input_embeddings(value)

    # Copied from transformers.models.paligemma.modeling_paligemma.PaliGemmaForConditionalGeneration.get_output_embeddings with PaliGemma->Gema3
    def get_output_embeddings(self):
        return self.language_model.get_output_embeddings()

    # Copied from transformers.models.paligemma.modeling_paligemma.PaliGemmaForConditionalGeneration.set_output_embeddings with PaliGemma->Gema3
    def set_output_embeddings(self, new_embeddings):
        self.language_model.set_output_embeddings(new_embeddings)

    # Copied from transformers.models.paligemma.modeling_paligemma.PaliGemmaForConditionalGeneration.set_decoder with PaliGemma->Gema3
    def set_decoder(self, decoder):
        self.language_model.set_decoder(decoder)

    # Copied from transformers.models.paligemma.modeling_paligemma.PaliGemmaForConditionalGeneration.get_decoder with PaliGemma->Gema3
    def get_decoder(self):
        return self.language_model.get_decoder()

    def get_image_features(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Projects the last hidden state from the vision model into language model space.

        Args:
            pixel_values (`torch.FloatTensor]` of shape `(batch_size, channels, height, width)`)
               The tensors corresponding to the input images.
        Returns:
            image_features (`torch.Tensor`): Image feature tensor of shape `(num_images, image_length, embed_dim)`).
        """
        vision_outputs = self.vision_model(pixel_values=pixel_values).last_hidden_state
        image_features = self.multimodal_projector(vision_outputs)
        return image_features

    @deprecate_kwarg("num_logits_to_keep", version="4.50", new_name="logits_to_keep")
    @add_start_docstrings_to_model_forward(GEMMA3_INPUTS_DOCSTRING)
    @replace_return_docstrings(
        output_type=Gemma3CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC
    )
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[list[torch.FloatTensor], Cache]] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **lm_kwargs,
    ) -> Union[tuple, Gemma3CausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.text_config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.text_config.vocab_size]`.

            logits_to_keep (`int` or `torch.Tensor`, *optional*):
                If an `int`, compute logits for the last `logits_to_keep` tokens. If `0`, calculate logits for all
                `input_ids` (special case). Only last token logits are needed for generation, and calculating them only for that
                token can save memory, which becomes pretty significant for long sequences or large vocabulary size.
                If a `torch.Tensor`, must be 1D corresponding to the indices to keep in the sequence length dimension.
                This is useful when using packed tensor format (single dimension for batch and sequence length).

        Returns:

        Example:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, PaliGemmaForConditionalGeneration

        >>> model = PaliGemmaForConditionalGeneration.from_pretrained("google/PaliGemma-test-224px-hf")
        >>> processor = AutoProcessor.from_pretrained("google/PaliGemma-test-224px-hf")

        >>> prompt = "answer en Where is the cow standing?"
        >>> url = "https://huggingface.co/gv-hf/PaliGemma-test-224px-hf/resolve/main/cow_beach_1.png"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> inputs = processor(images=image, text=prompt,  return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(**inputs, max_length=30)
        >>> processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "answer en Where is the cow standing?\nbeach"
        ```"""

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You must specify exactly one of input_ids or inputs_embeds"
            )

        if pixel_values is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both pixel_values and inputs_embeds at the same time, and must specify either one"
            )

        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        is_training = token_type_ids is not None and labels is not None

        if inputs_embeds is None:
            if self.config.text_config.image_token_id >= self.get_input_embeddings().num_embeddings:
                image_token_mask = input_ids == self.config.text_config.image_token_id
                llm_input_ids = input_ids.clone()
                llm_input_ids[image_token_mask] = 0
            else:
                llm_input_ids = input_ids

            inputs_embeds = self.get_input_embeddings()(llm_input_ids)

        if cache_position is None:
            past_seen_tokens = (
                past_key_values.get_seq_length() if past_key_values is not None else 0
            )
            cache_position = torch.arange(
                past_seen_tokens,
                past_seen_tokens + inputs_embeds.shape[1],
                device=inputs_embeds.device,
            )

        if position_ids is None:
            position_ids = (cache_position.unsqueeze(0) + 1)

        # Merge text and images
        if pixel_values is not None:
            image_features = self.get_image_features(pixel_values).to(inputs_embeds.device, inputs_embeds.dtype)

            image_mask = input_ids == self.config.text_config.image_token_id
            image_mask = image_mask.unsqueeze(-1)
            image_mask = image_mask.expand_as(inputs_embeds).to(inputs_embeds.device)

            if (
                not is_torchdynamo_compiling() and
                (emb_s := inputs_embeds[image_mask].numel()) != (img_s := image_features.numel())
            ):
                raise ValueError(
                    f"Number of image features ({img_s}) does not match number of special image tokens in the input "
                    f"text ({emb_s}). "
                )

            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_features)

        causal_mask = self._update_causal_mask(
            attention_mask,
            token_type_ids,
            past_key_values,
            cache_position,
            inputs_embeds,
            is_training,
        )
        outputs = self.language_model(
            attention_mask=causal_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            logits_to_keep=logits_to_keep,
            **lm_kwargs,
        )

        logits = outputs.logits
        loss = None
        if labels is not None:
            # Upcast to float if we need to compute the loss to avoid potential precision issues
            logits = logits.float()
            shift_logits = logits[..., :-1, :]
            shift_labels = labels[..., 1:]
            if attention_mask is not None:
                # we use the input attention mask to shift the logits and labels, because it is 2D.
                # we also crop attn mask in case it is longer, which happens in PrefixTuning with peft
                shift_attention_mask = attention_mask[:, -shift_logits.shape[1] :].to(
                    logits.device
                )
                shift_logits = shift_logits[
                    shift_attention_mask.to(logits.device) != 0
                ].contiguous()
                shift_labels = shift_labels[
                    shift_attention_mask.to(shift_labels.device) != 0
                ].contiguous()
            else:
                shift_logits = shift_logits.contiguous()
                shift_labels = shift_labels.contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()

            flat_logits = shift_logits.view(-1, self.config.text_config.vocab_size)
            flat_labels = shift_labels.view(-1).to(shift_logits.device)
            loss = loss_fct(flat_logits, flat_labels)
        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return Gemma3CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            image_hidden_states=image_features if pixel_values is not None else None,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        inputs_embeds=None,
        cache_position=None,
        position_ids=None,
        pixel_values=None,
        attention_mask=None,
        token_type_ids=None,
        use_cache=True,
        logits_to_keep=None,
        labels=None,
        **kwargs,
    ):
        # Overwritten -- custom `position_ids` and `pixel_values` handling
        model_inputs = self.language_model.prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            cache_position=cache_position,
            use_cache=use_cache,
            logits_to_keep=logits_to_keep,
            token_type_ids=token_type_ids,
            **kwargs,
        )

        # If we're in cached decoding stage, pixel values should be None because
        # input ids do not contain special image tokens anymore. Otherwise we
        # need pixel values to be passed to model.
        # NOTE: use_cache=False needs pixel_values always
        if cache_position[0] == 0:
            model_inputs["pixel_values"] = pixel_values

        is_training = token_type_ids is not None and labels is not None
        if cache_position[0] == 0 and isinstance(past_key_values, HybridCache):
            input_tensor = inputs_embeds if inputs_embeds is not None else input_ids
            causal_mask = self._update_causal_mask(
                attention_mask,
                token_type_ids,
                past_key_values,
                cache_position,
                input_tensor,
                is_training,
            )
            model_inputs["attention_mask"] = causal_mask

        return model_inputs

    def _update_causal_mask(
        self,
        attention_mask,
        token_type_ids,
        past_key_values,
        cache_position,
        input_tensor,
        is_training: bool = False,
    ):
        if self.config.text_config._attn_implementation == "flash_attention_2":
            if attention_mask is not None and 0.0 in attention_mask:
                return attention_mask
            return None

        using_static_cache = isinstance(past_key_values, StaticCache)
        min_dtype = torch.finfo(self.dtype).min
        inputs_lead_dim, sequence_length = input_tensor.shape[:2]
        if using_static_cache:
            target_length = past_key_values.get_max_cache_shape()
        elif isinstance(past_key_values, HybridCache):
            target_length = past_key_values.get_max_cache_shape()
        else:
            target_length = (
                attention_mask.shape[-1]
                if isinstance(attention_mask, torch.Tensor)
                else cache_position[0] + sequence_length + 1
            )

        if attention_mask is not None and attention_mask.dim() == 4:
            # In this case we assume that the mask comes already in inverted form and requires no inversion or slicing.
            return attention_mask

        causal_mask = torch.full(
            (sequence_length, target_length), fill_value=min_dtype, dtype=self.dtype, device=cache_position.device
        )
        # Causal diagonal mask only if training, otherwise attend to the whole prefix. Training-specific attn for prefix is handled below
        if sequence_length != 1:
            if is_training:
                causal_mask = torch.triu(causal_mask, diagonal=1)
            else:
                causal_mask[:, :sequence_length] = 0.0

        causal_mask *= torch.arange(target_length, device=cache_position.device) > cache_position.reshape(-1, 1)
        causal_mask = causal_mask[None, None, :, :].expand(inputs_lead_dim, 1, -1, -1)
        if attention_mask is not None:
            causal_mask = causal_mask.clone()  # copy to contiguous memory for in-place edit
            mask_length = attention_mask.shape[-1]
            padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :].to(causal_mask.device)
            padding_mask = padding_mask == 0
            causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
                padding_mask, min_dtype
            )
            # we are training thus we need to create a full mask on the image + prefix but causal on suffix
            if is_training:
                causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
                    token_type_ids[:, None, None, :].to(causal_mask.device) == 0, 0
                )
        return causal_mask


__all__ = [
    "Gemma3Config",
    "Gemma3TextConfig",
    "Gemma3VisionConfig",
    "Gemma3Processor",
    "Gemma3PreTrainedModel",  # noqa: F822
    "Gemma3Model",
    "Gemma3ForCausalLM",
    "Gemma3ForConditionalGeneration",
]
