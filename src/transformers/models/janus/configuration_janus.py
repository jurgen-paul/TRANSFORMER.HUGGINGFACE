# coding=utf-8
# Copyright 2025 Deepseek AI and The HuggingFace Team. All rights reserved.
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
"""Janus model configuration"""

from typing import List

from ...configuration_utils import PretrainedConfig
from ...utils import logging
from ..auto import CONFIG_MAPPING, AutoConfig


logger = logging.get_logger(__name__)


class JanusVisionConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`JanusVisionModel`]. It is used to instantiate a
    `JanusVisionModel` according to the specified arguments, defining the model architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        hidden_size (`int`, *optional*, defaults to 1024):
            Dimensionality of the encoder layers and the pooler layer.
        mlp_ratio (`float`, *optional*, defaults to 4.0):
            Ratio of the hidden size of the MLPs relative to `hidden_size`.
        projection_dim (`int`, *optional*, defaults to 2048):
            Dimensionality of the MLP projection head.
        num_hidden_layers (`int`, *optional*, defaults to 24):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 16):
            Number of attention heads for each attention layer in the Transformer encoder.
        num_channels (`int`, *optional*, defaults to 3):
            The number of input channels.
        image_size (`int`, *optional*, defaults to 384):
            The size (resolution) of each image.
        patch_size (`int`, *optional*, defaults to 16):
            The size (resolution) of each patch.
        hidden_act (`str` or `function`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"selu"`, and `"gelu_new"` are supported.
        layer_norm_eps (`float`, *optional*, defaults to 1e-06):
            The epsilon used by the layer normalization layers.
        attention_bias (`bool`, *optional*, defaults to `True`):
            Whether to add a bias to the queries, keys, and values in the attention layers.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            Dropout probability for attention weights.

        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated normal initializer for initializing all weight matrices.

        hidden_dropout_rate (`float`, *optional*, defaults to 0.0):
            The dropout probability for fully connected layers in the encoder.
        projection_dropout (`float`, *optional*, defaults to 0.0):
            Dropout probability for the projection layer.
        use_qk_norm (`bool`, *optional*, defaults to `False`):
            Whether to normalize the query and key matrices.
        use_vision_head (`bool`, *optional*, defaults to `True`):
            Whether to use a Vision specific Attention Pool head.
        depth (`int`, *optional*, defaults to 2):
            Number of hidden layers in the aligner module.
        num_image_tokens (`int`, *optional*, defaults to 576):
            Number of image tokens.
    """

    model_type = "siglip_vision_model"
    base_config_key = "vision_config"

    def __init__(
        self,
        hidden_size=1024,
        mlp_ratio=4.0,
        projection_dim=2048,
        num_hidden_layers=24,
        num_attention_heads=16,
        num_channels=3,
        image_size=384,
        patch_size=16,
        hidden_act="gelu",
        layer_norm_eps=1e-6,
        attention_bias=True,
        attention_dropout=0.0,
        initializer_range=0.02,
        hidden_dropout_rate=0.0,
        projection_dropout=0.0,
        use_qk_norm=False,
        use_vision_head=True,
        depth=2,
        num_image_tokens=576,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.hidden_size = hidden_size
        self.mlp_ratio = mlp_ratio
        self.projection_dim = projection_dim
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_channels = num_channels
        self.patch_size = patch_size
        self.image_size = image_size
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.layer_norm_eps = layer_norm_eps
        self.hidden_act = hidden_act
        self.intermediate_size = int(hidden_size * mlp_ratio)
        self.hidden_dropout_rate = hidden_dropout_rate
        self.projection_dropout = projection_dropout
        self.use_qk_norm = use_qk_norm
        self.initializer_range = initializer_range
        self.use_vision_head = use_vision_head
        self.depth = depth
        self.num_image_tokens = num_image_tokens
        self.use_special_tokens = kwargs.get("use_special_tokens", False)


class JanusVQVAEConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`JanusVQVAEModel`]. It is used to instantiate a
    `JanusVQVAEModel` according to the specified arguments, defining the model architecture.
    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information. Instantiating a
    configuration with the defaults will yield a similar configuration to the VQModel of the
    [yaswanthgali/Janus-Pro-1B-HF](https://huggingface.co/yaswanthgali/Janus-Pro-1B-HF).

    Args:
        embed_dim (`int`, *optional*, defaults to 8):
            Dimensionality of each embedding vector.
        num_embeddings (`int`, *optional*, defaults to 16384):
            Number of codebook embeddings.
        double_latent (`bool`, *optional*, defaults to `False`):
            Whether to use double z channels.
        latent_channels (`int`, *optional*, defaults to 256):
            Number of channels for the latent space.
        num_patches (`int`, *optional*, defaults to 32):
            Num of patches the input images can be divided into.
        in_channels (`int`, *optional*, defaults to 3):
            Number of input channels.
        out_channels (`int`, *optional*, defaults to 3):
            Number of out channels.
        base_channels (`int`, *optional*, defaults to 128):
            Base channel count.
        channel_multiplier (`List[int]`, *optional*, defaults to `[1, 1, 2, 2, 4]`):
            Channel multipliers for each resolution.
        num_res_blocks (`int`, *optional*, defaults to 2):
            Number of residual blocks.
        dropout (`float`, *optional*, defaults to 0.0):
            Dropout rate.
        attn_type (`str`, *optional*, defaults to `"vanilla"`):
            Attention type used in VQ-GAN encoder. Can be "vanilla" or None.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        projection_dim (`int`, *optional*, defaults to 2048):
            Dimensionality of the MLP projection head.
        num_hidden_layers (`int`, *optional*, defaults to 2):
            Number of hidden layers in VAVAE MLP Connecter module.
        hidden_act (`str` or `Callable`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"silu"` and `"gelu_new"` are supported.
        image_token_embed_dim (`int`, *optional*, defaults to 2048):
            Dimension of image embeddings. It should be same as the dimensionality of text embeddings.
    """

    model_type = "janus_vqgan"
    base_config_key = "vq_config"

    def __init__(
        self,
        embed_dim: int = 8,
        num_embeddings: int = 16384,
        double_latent: bool = False,
        latent_channels: int = 256,
        num_patches: int = 32,
        in_channels: int = 3,
        out_channels: int = 3,
        base_channels: int = 128,
        channel_multiplier: List[int] = [1, 1, 2, 2, 4],
        num_res_blocks: int = 2,
        dropout: float = 0.0,
        attn_type: str = "vanilla",
        initializer_range=0.02,
        projection_dim=2048,
        num_hidden_layers=2,
        hidden_act="gelu",
        image_token_embed_dim=2048,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_embeddings = num_embeddings
        self.double_latent = double_latent
        self.latent_channels = latent_channels
        self.num_patches = num_patches
        self.in_channels = in_channels
        self.base_channels = base_channels
        self.channel_multiplier = channel_multiplier
        self.num_res_blocks = num_res_blocks
        self.dropout = dropout
        self.attn_type = attn_type
        self.initializer_range = initializer_range
        self.out_channels = out_channels
        self.projection_dim = projection_dim
        self.num_hidden_layers = num_hidden_layers
        self.hidden_act = hidden_act
        self.image_token_embed_dim = image_token_embed_dim


class JanusConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`JanusModel`]. It is used to instantiate an
    Janus model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the Janus-1B or Janus-7B models.

    e.g. [yaswanthgali/Janus-Pro-1B-HF](https://huggingface.co/yaswanthgali/Janus-Pro-1B-HF) or
    [hugosilva664/Janus-Pro-7B-HF](https://huggingface.co/hugosilva664/Janus-Pro-7B-HF)

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        text_config (`Union[AutoConfig, dict]`, *optional*, defaults to `LlamaConfig`):
            The config object or dictionary of the text backbone.
        vision_config (`Union[AutoConfig, dict]`,  *optional*, defaults to `JanusVisionConfig`):
            The config object or dictionary of the vision backbone.
        vq_config (`Union[AutoConfig, dict]`,  *optional*, defaults to `JanusVQVAEConfig`):
            The config object or dictionary of the VQVAE backbone.

    Example:

    ```python
    >>> from transformers import JanusForConditionalGeneration, JanusConfig, JanusVisionConfig, JanusVQVAEConfig, LlamaConfig

    >>> # Initializing a Janus vision config
    >>> vision_config = JanusVisionConfig()

    >>> # Initializing a Llama config
    >>> text_config = LlamaConfig()

    >>> # Initializing a VQ config
    >>> vq_config = JanusVQVAEConfig()

    >>> # Initializing a Janus Pro 1B style configuration
    >>> configuration = JanusConfig(vision_config=vision_config, text_config=text_config, vq_config=vq_config)

    >>> # Initializing a model from the Janus Pro 1B style configuration
    >>> model = JanusForConditionalGeneration(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "janus"
    sub_configs = {
        "text_config": AutoConfig,
        "vision_config": JanusVisionConfig,
        "vq_config": JanusVQVAEConfig,
    }

    def __init__(self, text_config=None, vision_config=None, vq_config=None, **kwargs):
        if isinstance(text_config, dict):
            text_config["model_type"] = text_config.get("model_type", "llama")
            self.text_config = CONFIG_MAPPING[text_config["model_type"]](**text_config)

        elif text_config is None:
            logger.info("`text_config` is None. Initializing with default values")
            self.text_config = CONFIG_MAPPING["llama"](
                hidden_size=2048,
                num_hidden_layers=24,
                num_attention_heads=16,
                intermediate_size=5632,
                max_position_embeddings=16384,
                num_key_value_heads=16,
                vocab_size=102400,
            )
        elif isinstance(text_config, PretrainedConfig):
            self.text_config = text_config
        else:
            raise ValueError(
                f"Invalid type for `text_config`. Must be either `dict` or `LlamaConfig`."
                f" Type found: {type(text_config)}"
            )

        if vision_config is None:
            logger.info("`vision_config` is None. Initializing with default JanusVisionConfig values")
            self.vision_config = JanusVisionConfig()
        elif isinstance(vision_config, dict):
            self.vision_config = JanusVisionConfig(**vision_config)
        elif isinstance(vision_config, JanusVisionConfig):
            self.vision_config = vision_config
        else:
            raise ValueError(
                f"Invalid type for `vision_config`. Must be either `dict` or `JanusVisionConfig`."
                f" Type found: {type(vision_config)}"
            )

        if vq_config is None:
            logger.info("`vq_config` is None. Initializing with default JanusVQVAEConfig values")
            self.vq_config = JanusVQVAEConfig()
        elif isinstance(vq_config, dict):
            self.vq_config = JanusVQVAEConfig(**vq_config)
        elif isinstance(vq_config, JanusVQVAEConfig):
            self.vq_config = vq_config
        else:
            raise ValueError(
                f"Invalid type for `vq_config`. Must be either `dict` or `JanusVQVAEConfig`."
                f" Type found: {type(vq_config)}"
            )

        # This dimension is required when decoding discrete image tokens to continuous input.
        self.vq_config.num_patches = self.vision_config.image_size // self.vision_config.patch_size
        # The default is only the index for the 1B model, 7B uses a different one
        self.image_token_index = kwargs.get("image_token_index", 100581)
        super().__init__(**kwargs)


__all__ = ["JanusVQVAEConfig", "JanusVisionConfig", "JanusConfig"]
