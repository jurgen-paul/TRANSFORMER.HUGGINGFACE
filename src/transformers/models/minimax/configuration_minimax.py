#                🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨
#           This file was automatically generated from src/transformers/models/minimax/modular_minimax.py.
#               Do NOT edit this file manually as any edits will be overwritten by the generation of
#             the file from the modular. If any change should be done, please apply the change to the
#                          modular_minimax.py file directly. One of our CI enforces this.
#                🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨
# coding=utf-8
# Copyright 2025 MiniMaxAI and HuggingFace Inc. teams. All rights reserved.
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

from ...configuration_utils import PretrainedConfig


class MiniMaxConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`MiniMaxModel`]. It is used to instantiate an
    MiniMax model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the MiniMax.

    [MiniMaxAI/MiniMax-Text-01](https://huggingface.co/MiniMaxAI/MiniMax-Text-01)

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        vocab_size (`int`, *optional*, defaults to 32000):
            Vocabulary size of the MiniMax model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`MiniMaxModel`]
        hidden_size (`int`, *optional*, defaults to 4096):
            Dimension of the hidden representations.
        intermediate_size (`int`, *optional*, defaults to 14336):
            Dimension of the MLP representations.
        num_hidden_layers (`int`, *optional*, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 32):
            Number of attention heads for each attention layer in the Transformer encoder.
        num_key_value_heads (`int`, *optional*, defaults to 8):
            This is the number of key_value heads that should be used to implement Grouped Query Attention. If
            `num_key_value_heads=num_attention_heads`, the model will use Multi Head Attention (MHA), if
            `num_key_value_heads=1 the model will use Multi Query Attention (MQA) otherwise GQA is used. When
            converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed
            by meanpooling all the original heads within that group. For more details checkout [this
            paper](https://arxiv.org/pdf/2305.13245.pdf). If it is not specified, will default to `8`.
        head_dim (`int`, *optional*, defaults to `hidden_size // num_attention_heads`):
            The attention head dimension.
        hidden_act (`str` or `function`, *optional*, defaults to `"silu"`):
            The non-linear activation function (function or string) in the decoder.
        max_position_embeddings (`int`, *optional*, defaults to `4096*32`):
            The maximum sequence length that this model might ever be used with. MiniMax's sliding window attention
            allows sequence of up to 4096*32 tokens.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        rms_norm_eps (`float`, *optional*, defaults to 1e-05):
            The epsilon used by the rms normalization layers.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.
        pad_token_id (`int`, *optional*):
            The id of the padding token.
        bos_token_id (`int`, *optional*, defaults to 1):
            The id of the "beginning-of-sequence" token.
        eos_token_id (`int`, *optional*, defaults to 2):
            The id of the "end-of-sequence" token.
        tie_word_embeddings (`bool`, *optional*, defaults to `False`):
            Whether the model's input and output word embeddings should be tied.
        rope_theta (`float`, *optional*, defaults to 1000000.0):
            The base period of the RoPE embeddings.
        sliding_window (`int`, *optional*):
            Sliding window attention window size. If not specified, will default to `4096`.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        num_experts_per_tok (`int`, *optional*, defaults to 2):
            The number of experts to route per-token, can be also interpreted as the `top-k` routing
            parameter
        num_local_experts (`int`, *optional*, defaults to 8):
            Number of experts per Sparse MLP layer.
        output_router_logits (`bool`, *optional*, defaults to `False`):
            Whether or not the router logits should be returned by the model. Enabeling this will also
            allow the model to output the auxiliary loss. See [here]() for more details
        router_aux_loss_coef (`float`, *optional*, defaults to 0.001):
            The aux loss factor for the total loss.
        router_jitter_noise (`float`, *optional*, defaults to 0.0):
            Amount of noise to add to the router.
        attn_type_list (`List[int]`, *optional*, defaults to `[0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1]`):
            List of attention types for each layer. `0` for linear (lightning) attention
            and `1` for full (normal) attention.
        block_size (`int`, *optional*, defaults to 256):
            The length of each attention block, determining how queries, keys, and values
            are grouped and processed for intra- and inter-block attention.
        postnorm (`bool`, *optional*, defaults to `False`):
            Use residual connections post-normalization.
        layernorm_full_attention_alpha (`float`, *optional*, defaults to 1):
            Weight for residual value in residual connection after normal attention.
        layernorm_full_attention_beta (`float`, *optional*, defaults to 1):
            Weight for hidden state value in residual connection after normal attention.
        layernorm_linear_attention_alpha (`float`, *optional*, defaults to 1):
            Weight for residual value in residual connection after lightning attention.
        layernorm_linear_attention_beta (`float`, *optional*, defaults to 1):
            Weight for hidden state value in residual connection after lightning attention.
        layernorm_mlp_alpha (`float`, *optional*, defaults to 1):
            Weight for residual value in residual connection after MLP.
        layernorm_mlp_beta (`float`, *optional*, defaults to 1):
            Weight for hidden state value in residual connection after MLP.

    ```python
    >>> from transformers import MiniMaxModel, MiniMaxConfig

    >>> # Initializing a MiniMax style configuration
    >>> configuration = MiniMaxConfig()

    >>> # Initializing a model from the MiniMax style configuration
    >>> model = MiniMaxModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "minimax"
    keys_to_ignore_at_inference = ["past_key_values"]
    base_model_tp_plan = {
        "layers.*.self_attn.q_proj": "colwise",
        "layers.*.self_attn.k_proj": "colwise",
        "layers.*.self_attn.v_proj": "colwise",
        "layers.*.self_attn.o_proj": "rowwise",
        "layers.*.block_sparse_moe.gate": "colwise_rep",  # we need to replicate here to correctly route experts
        "layers.*.block_sparse_moe.experts.*.w1": "colwise",
        "layers.*.block_sparse_moe.experts.*.w2": "rowwise",
        "layers.*.block_sparse_moe.experts.*.w3": "colwise",
    }
    base_model_pp_plan = {
        "embed_tokens": (["input_ids"], ["inputs_embeds"]),
        "layers": (["hidden_states", "attention_mask"], ["hidden_states"]),
        "norm": (["hidden_states"], ["hidden_states"]),
    }

    def __init__(
        self,
        vocab_size=32000,
        hidden_size=4096,
        intermediate_size=14336,
        num_hidden_layers=12,
        num_attention_heads=32,
        num_key_value_heads=8,
        head_dim=None,
        hidden_act="silu",
        max_position_embeddings=4096 * 32,
        initializer_range=0.02,
        rms_norm_eps=1e-5,
        use_cache=True,
        pad_token_id=None,
        bos_token_id=None,
        eos_token_id=None,
        tie_word_embeddings=False,
        rope_theta=1e6,
        sliding_window=None,
        attention_dropout=0.0,
        num_experts_per_tok=2,
        num_local_experts=8,
        output_router_logits=False,
        router_aux_loss_coef=0.001,
        router_jitter_noise=0.0,
        attn_type_list=[0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1],
        block_size=256,
        postnorm=False,
        layernorm_full_attention_alpha=1,
        layernorm_full_attention_beta=1,
        layernorm_linear_attention_alpha=1,
        layernorm_linear_attention_beta=1,
        layernorm_mlp_alpha=1,
        layernorm_mlp_beta=1,
        **kwargs,
    ):
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.sliding_window = sliding_window

        # for backward compatibility
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.attention_dropout = attention_dropout
        self.head_dim = head_dim if head_dim is not None else self.hidden_size // self.num_attention_heads

        self.num_experts_per_tok = num_experts_per_tok
        self.num_local_experts = num_local_experts
        self.output_router_logits = output_router_logits
        self.router_aux_loss_coef = router_aux_loss_coef
        self.router_jitter_noise = router_jitter_noise

        self.attn_type_list = attn_type_list
        self.block_size = block_size
        self.postnorm = postnorm
        self.layernorm_full_attention_alpha = layernorm_full_attention_alpha
        self.layernorm_full_attention_beta = layernorm_full_attention_beta
        self.layernorm_linear_attention_alpha = layernorm_linear_attention_alpha
        self.layernorm_linear_attention_beta = layernorm_linear_attention_beta
        self.layernorm_mlp_alpha = layernorm_mlp_alpha
        self.layernorm_mlp_beta = layernorm_mlp_beta
