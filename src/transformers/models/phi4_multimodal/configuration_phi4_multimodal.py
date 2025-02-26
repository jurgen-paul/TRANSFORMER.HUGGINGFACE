#                🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨
#           This file was automatically generated from src/transformers/models/phi4_multimodal/modular_phi4_multimodal.py.
#               Do NOT edit this file manually as any edits will be overwritten by the generation of
#             the file from the modular. If any change should be done, please apply the change to the
#                          modular_phi4_multimodal.py file directly. One of our CI enforces this.
#                🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨

from ...configuration_utils import PretrainedConfig


class Phi4MultimodalVisionConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`Phi4MultimodalVisionModel`]. It is used to instantiate a
    Phi4Multimodal vision encoder according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the vision encoder of the Phi4Multimodal
    [google/phi4_multimodal-base-patch16-224](https://huggingface.co/google/phi4_multimodal-base-patch16-224) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        hidden_size (`int`, *optional*, defaults to 768):
            Dimensionality of the encoder layers and the pooler layer.
        intermediate_size (`int`, *optional*, defaults to 3072):
            Dimensionality of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
        num_hidden_layers (`int`, *optional*, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        num_channels (`int`, *optional*, defaults to 3):
            Number of channels in the input images.
        image_size (`int`, *optional*, defaults to 224):
            The size (resolution) of each image.
        patch_size (`int`, *optional*, defaults to 16):
            The size (resolution) of each patch.
        hidden_act (`str` or `function`, *optional*, defaults to `"gelu_pytorch_tanh"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"selu"` and `"gelu_new"` `"quick_gelu"` are supported.
        layer_norm_eps (`float`, *optional*, defaults to 1e-06):
            The epsilon used by the layer normalization layers.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.

    Example:

    ```python
    >>> from transformers import Phi4MultimodalVisionConfig, Phi4MultimodalVisionModel

    >>> # Initializing a Phi4MultimodalVisionConfig with google/phi4_multimodal-base-patch16-224 style configuration
    >>> configuration = Phi4MultimodalVisionConfig()

    >>> # Initializing a Phi4MultimodalVisionModel (with random weights) from the google/phi4_multimodal-base-patch16-224 style configuration
    >>> model = Phi4MultimodalVisionModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "phi4_multimodal_vision_model"
    base_config_key = "vision_config"

    def __init__(
        self,
        hidden_size=1152,
        intermediate_size=4304,
        num_hidden_layers=27,
        num_attention_heads=16,
        num_channels=3,
        image_size=448,
        patch_size=14,
        hidden_act="gelu_pytorch_tanh",
        layer_norm_eps=1e-6,
        attention_dropout=0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_channels = num_channels
        self.patch_size = patch_size
        self.image_size = image_size
        self.attention_dropout = attention_dropout
        self.layer_norm_eps = layer_norm_eps
        self.hidden_act = hidden_act

        self.image_embd_layer = {
            "crop_size": 448,
            "embedding_cls": "tune_image",
            "enable_gradient_checkpointing": True,
            "hd_transform_order": "sub_glb",
            "projection_cls": "mlp",
            "use_hd_transform": True,
            "with_learnable_separator": True,
        }


class Phi4MultimodalAudioConfig(PretrainedConfig):
    def __init__(
        self,
        hidden_size: int = 1024,  # attention_dim
        num_attention_heads: int = 16,  # attention_heads
        intermediate_size: int = 2048,
        activation: str = "swish",
        chunk_size: int = None,
        left_chunk: int = None,
        num_lang: int = None,
        num_blocks: int = 6,
        dropout_rate: float = 0.0,
        input_layer: str = "nemo_conv",
        causal: bool = True,
        batch_norm: bool = False,
        ext_pw_out_channel: int = 0,
        ext_pw_kernel_size: int = 1,
        depthwise_seperable_out_channel: int = 256,
        depthwise_multiplier: int = 1,
        chunk_se: int = 0,
        kernel_size: int = 3,
        conv_activation: str = "relu",
        conv_glu_type: str = "sigmoid",
        bias_in_glu: bool = True,
        linear_glu_in_convm: bool = False,
        attention_glu_type: str = "swish",
        extra_layer_output_idx: int = -1,
        extra_multi_layer_output_idxs: list = [],
        activation_checkpointing: str = "",
        relative_attention_bias_args: dict = None,
        time_reduction: int = 4,
        replication_pad_for_subsample_embedding: bool = False,
        attention_group_size: int = 1,
        encoder_embedding_config: dict = None,
        positional_dropout_rate: float = 0.0,
    ):
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.activation = activation
        self.chunk_size = chunk_size
        self.left_chunk = left_chunk
        self.num_lang = num_lang
        self.num_blocks = num_blocks
        self.dropout_rate = dropout_rate
        self.input_layer = input_layer
        self.causal = causal
        self.batch_norm = batch_norm
        self.ext_pw_out_channel = ext_pw_out_channel
        self.ext_pw_kernel_size = ext_pw_kernel_size
        self.depthwise_seperable_out_channel = depthwise_seperable_out_channel
        self.depthwise_multiplier = depthwise_multiplier
        self.chunk_se = chunk_se
        self.kernel_size = kernel_size
        self.conv_activation = conv_activation
        self.conv_glu_type = conv_glu_type
        self.bias_in_glu = bias_in_glu
        self.linear_glu_in_convm = linear_glu_in_convm
        self.attention_glu_type = attention_glu_type
        self.extra_layer_output_idx = extra_layer_output_idx
        self.extra_multi_layer_output_idxs = extra_multi_layer_output_idxs
        self.activation_checkpointing = activation_checkpointing
        self.relative_attention_bias_args = relative_attention_bias_args
        self.time_reduction = time_reduction
        self.replication_pad_for_subsample_embedding = replication_pad_for_subsample_embedding
        self.attention_group_size = attention_group_size
        self.encoder_embedding_config = encoder_embedding_config
        self.positional_dropout_rate = positional_dropout_rate

        self.nemo_conv_settings = {
            "subsampling": "dw_striding",
            "subsampling_factor": self.time_reduction,
            "conv_channels": 1024,
            "activation": "relu",
            "is_causal": False,
        }
        self.encoder_embedding_config = {
            "input_size": 80,
        }


class Phi4MultimodalConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`Phi4MultimodalModel`]. It is used to instantiate a Phi-3
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the
    [microsoft/Phi-3-mini-4k-instruct](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct).

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        vocab_size (`int`, *optional*, defaults to 32064):
            Vocabulary size of the Phi-3 model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`Phi4MultimodalModel`].
        hidden_size (`int`, *optional*, defaults to 3072):
            Dimension of the hidden representations.
        intermediate_size (`int`, *optional*, defaults to 8192):
            Dimension of the MLP representations.
        num_hidden_layers (`int`, *optional*, defaults to 32):
            Number of hidden layers in the Transformer decoder.
        num_attention_heads (`int`, *optional*, defaults to 32):
            Number of attention heads for each attention layer in the Transformer decoder.
        num_key_value_heads (`int`, *optional*):
            This is the number of key_value heads that should be used to implement Grouped Query Attention. If
            `num_key_value_heads=num_attention_heads`, the model will use Multi Head Attention (MHA), if
            `num_key_value_heads=1` the model will use Multi Query Attention (MQA) otherwise GQA is used. When
            converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed
            by meanpooling all the original heads within that group. For more details checkout [this
            paper](https://arxiv.org/pdf/2305.13245.pdf). If it is not specified, will default to
            `num_attention_heads`.
        resid_pdrop (`float`, *optional*, defaults to 0.0):
            Dropout probability for mlp outputs.
        embd_pdrop (`int`, *optional*, defaults to 0.0):
            The dropout ratio for the embeddings.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio after computing the attention scores.
        hidden_act (`str` or `function`, *optional*, defaults to `"silu"`):
            The non-linear activation function (function or string) in the decoder.
        max_position_embeddings (`int`, *optional*, defaults to 4096):
            The maximum sequence length that this model might ever be used with.
        original_max_position_embeddings (`int`, *optional*, defaults to 4096):
            The maximum sequence length that this model was trained with. This is used to determine the size of the
            original RoPE embeddings when using long scaling.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        rms_norm_eps (`float`, *optional*, defaults to 1e-05):
            The epsilon value used for the RMSNorm.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`. Whether to tie weight embeddings or not.
        tie_word_embeddings (`bool`, *optional*, defaults to `False`):
            Whether to tie weight embeddings
        rope_theta (`float`, *optional*, defaults to 10000.0):
            The base period of the RoPE embeddings.
        rope_scaling (`dict`, *optional*):
            The scaling strategy for the RoPE embeddings. If `None`, no scaling is applied. If a dictionary, it must
            contain the following keys: `type`, `short_factor` and `long_factor`. The `type` must be `longrope` and
            the `short_factor` and `long_factor` must be lists of numbers with the same length as the hidden size
            divided by the number of attention heads divided by 2.
        partial_rotary_factor (`float`, *optional*, defaults to 1.0):
            Percentage of the query and keys which will have rotary embedding. Must be between 0.0 and 1.0.
        bos_token_id (`int`, *optional*, defaults to 1):
            The id of the "beginning-of-sequence" token.
        eos_token_id (`int`, *optional*, defaults to 32000):
            The id of the "end-of-sequence" token.
        pad_token_id (`int`, *optional*, defaults to 32000):
            The id of the padding token.
        sliding_window (`int`, *optional*):
            Sliding window attention window size. If `None`, no sliding window is applied.

    Example:

    ```python
    >>> from transformers import Phi4MultimodalModel, Phi4MultimodalConfig

    >>> # Initializing a Phi-3 style configuration
    >>> configuration = Phi4MultimodalConfig.from_pretrained("microsoft/Phi-3-mini-4k-instruct")

    >>> # Initializing a model from the configuration
    >>> model = Phi4MultimodalModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "phi4_multimodal"
    keys_to_ignore_at_inference = ["past_key_values"]
    base_model_tp_plan = {
        "layers.*.self_attn.qkv_proj": "colwise_rep",  # we need to replicate here due to the slicing of qkv
        "layers.*.self_attn.o_proj": "rowwise_rep",  # we need to replicate here due to the slicing of qkv
        "layers.*.mlp.gate_up_proj": "colwise_rep",  # we need to replicate here due to the `chunk` operation
        "layers.*.mlp.down_proj": "rowwise_rep",  # we need to replicate here due to the `chunk` operation
    }
    base_model_pp_plan = {
        "embed_tokens": (["input_ids"], ["inputs_embeds"]),
        "layers": (["hidden_states", "attention_mask"], ["hidden_states"]),
        "norm": (["hidden_states"], ["hidden_states"]),
    }

    def __init__(
        self,
        vocab_size=200064,
        hidden_size=3072,
        intermediate_size=8192,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=None,
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        attention_dropout=0.0,
        hidden_act="silu",
        max_position_embeddings=4096,
        initializer_range=0.02,
        rms_norm_eps=1e-5,
        use_cache=True,
        tie_word_embeddings=False,
        rope_theta=10000.0,
        rope_scaling=None,
        partial_rotary_factor=1,
        bos_token_id=199999,
        eos_token_id=199999,
        pad_token_id=199999,
        sliding_window=None,
        embd_layer: str = "default",
        img_processor=None,
        audio_processor=None,
        vision_lora=None,
        speech_lora=None,
        **kwargs,
    ):
        super().__init__(
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads

        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.num_key_value_heads = num_key_value_heads
        self.resid_pdrop = resid_pdrop
        self.embd_pdrop = embd_pdrop
        self.attention_dropout = attention_dropout
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.partial_rotary_factor = partial_rotary_factor
        self._rope_scaling_adjustment()
        self._rope_scaling_validation()
        self.sliding_window = sliding_window
        self.embd_layer = embd_layer
        self.img_processor = img_processor
        self.audio_processor = audio_processor
        self.vision_lora = vision_lora
        self.speech_lora = speech_lora

    def _rope_scaling_adjustment(self):
        """
        Adjust the `type` of the `rope_scaling` configuration for backward compatibility.
        """
        if self.rope_scaling is None:
            return

        rope_scaling_type = self.rope_scaling.get("type", None)

        # For backward compatibility if previous version used "su" or "yarn"
        if rope_scaling_type is not None and rope_scaling_type in ["su", "yarn"]:
            self.rope_scaling["type"] = "longrope"

    def _rope_scaling_validation(self):
        """
        Validate the `rope_scaling` configuration.
        """
        if self.rope_scaling is None:
            return

        if not isinstance(self.rope_scaling, dict) or len(self.rope_scaling) != 3:
            raise ValueError(
                "`rope_scaling` must be a dictionary with three fields, `type`, `short_factor` and `long_factor`, "
                f"got {self.rope_scaling}"
            )
        rope_scaling_type = self.rope_scaling.get("type", None)
        rope_scaling_short_factor = self.rope_scaling.get("short_factor", None)
        rope_scaling_long_factor = self.rope_scaling.get("long_factor", None)
        if rope_scaling_type is None or rope_scaling_type not in ["longrope"]:
            raise ValueError(f"`rope_scaling`'s type field must be one of ['longrope'], got {rope_scaling_type}")
        if not (
            isinstance(rope_scaling_short_factor, list)
            and all(isinstance(x, (int, float)) for x in rope_scaling_short_factor)
        ):
            raise ValueError(
                f"`rope_scaling`'s short_factor field must be a list of numbers, got {rope_scaling_short_factor}"
            )
        rotary_ndims = int(self.hidden_size // self.num_attention_heads * self.partial_rotary_factor)
        if not len(rope_scaling_short_factor) == rotary_ndims // 2:
            raise ValueError(
                f"`rope_scaling`'s short_factor field must have length {rotary_ndims // 2}, got {len(rope_scaling_short_factor)}"
            )
        if not (
            isinstance(rope_scaling_long_factor, list)
            and all(isinstance(x, (int, float)) for x in rope_scaling_long_factor)
        ):
            raise ValueError(
                f"`rope_scaling`'s long_factor field must be a list of numbers, got {rope_scaling_long_factor}"
            )
        if not len(rope_scaling_long_factor) == rotary_ndims // 2:
            raise ValueError(
                f"`rope_scaling`'s long_factor field must have length {rotary_ndims // 2}, got {len(rope_scaling_long_factor)}"
            )
