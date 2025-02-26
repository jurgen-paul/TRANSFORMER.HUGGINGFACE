import math
from typing import Callable, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn

from transformers.modeling_attn_mask_utils import _prepare_4d_attention_mask

from ...activations import ACT2CLS, ACT2FN
from ...cache_utils import DynamicCache
from ...configuration_utils import PretrainedConfig
from ...modeling_outputs import (
    BaseModelOutputWithPast,
    BaseModelOutputWithPooling,
    CausalLMOutputWithPast,
)
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from ...utils import logging
from ..phi3.configuration_phi3 import Phi3Config
from ..phi3.modeling_phi3 import Phi3DecoderLayer, Phi3ForCausalLM, Phi3Model, Phi3RMSNorm
from ..siglip.configuration_siglip import SiglipVisionConfig
from ..siglip.modeling_siglip import (
    SiglipEncoder,
    SiglipEncoderLayer,
    SiglipMLP,
    SiglipMultiheadAttentionPoolingHead,
    SiglipVisionEmbeddings,
    SiglipVisionTransformer,
    default_flax_embed_init,
    lecun_normal_,
)


logger = logging.get_logger(__name__)


class Phi4MultimodalVisionConfig(SiglipVisionConfig):
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
            **kwargs,
        )

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


class Phi4MultimodalConfig(Phi3Config):
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
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            resid_pdrop=resid_pdrop,
            embd_pdrop=embd_pdrop,
            attention_dropout=attention_dropout,
            hidden_act=hidden_act,
            max_position_embeddings=max_position_embeddings,
            initializer_range=initializer_range,
            rms_norm_eps=rms_norm_eps,
            use_cache=use_cache,
            tie_word_embeddings=tie_word_embeddings,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            partial_rotary_factor=partial_rotary_factor,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            sliding_window=sliding_window,
            **kwargs,
        )
        del self.original_max_position_embeddings
        self.embd_layer = embd_layer
        self.img_processor = img_processor
        self.audio_processor = audio_processor
        self.vision_lora = vision_lora
        self.speech_lora = speech_lora


# Special token ids
_IMAGE_SPECIAL_TOKEN_ID = 200010  # '<|endoftext10|>', or we can better name it (in `tokenizer_config.json`)
_AUDIO_SPECIAL_TOKEN_ID = 200011  # '<|endoftext11|>'
_COMPATIBLE_IMAGE_SPECIAL_TOKEN_ID_RANGE = [-9999, -1]  # For backward compatibility
_COMPATIBLE_AUDIO_SPECIAL_TOKEN_ID_RANGE = [float("-inf"), -10000]  # For backward compatibility


class Phi4MultimodalVisionMLP(SiglipMLP):
    pass


def vision_eager_attention_forward(
    module: nn.Module,
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
):
    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


class Phi4MultimodalVisionAttention(nn.Module):
    def __init__(self, config: Phi4MultimodalVisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {self.num_heads})."
            )
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout

        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Input shape: Batch x Time x Channel"""
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        attention_interface: Callable = vision_eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1)
        attn_output = self.out_proj(attn_output)
        return attn_output, attn_weights


class Phi4MultimodalVisionEncoderLayer(SiglipEncoderLayer):
    def __init__(self, config: Phi4MultimodalVisionConfig):
        super().__init__(config)
        self.self_attn = Phi4MultimodalVisionAttention(config)
        self.mlp = Phi4MultimodalVisionMLP(config)


class Phi4MultimodalVisionEncoder(SiglipEncoder):
    def __init__(self, config: Phi4MultimodalVisionConfig):
        super().__init__()
        self.layers = nn.ModuleList(
            [Phi4MultimodalVisionEncoderLayer(config) for _ in range(config.num_hidden_layers)]
        )


class Phi4MultimodalVisionPreTrainedModel(PreTrainedModel):
    config_class = Phi4MultimodalVisionConfig
    base_model_prefix = "phi4_vision"
    supports_gradient_checkpointing = True

    _no_split_modules = ["Phi4MultimodalVisionEncoderLayer"]
    _supports_flash_attn_2 = True
    _supports_sdpa = True

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, Phi4MultimodalVisionEmbeddings):
            width = (
                self.config.vision_config.hidden_size
                if isinstance(self.config, Phi4MultimodalVisionConfig)
                else self.config.hidden_size
            )
            nn.init.normal_(module.position_embedding.weight, std=1 / np.sqrt(width))
        elif isinstance(module, nn.Embedding):
            default_flax_embed_init(module.weight)
        elif isinstance(module, Phi4MultimodalVisionAttention):
            nn.init.normal_(module.q_proj.weight)
            nn.init.normal_(module.k_proj.weight)
            nn.init.normal_(module.v_proj.weight)
            nn.init.normal_(module.out_proj.weight)
            nn.init.zeros_(module.q_proj.bias)
            nn.init.zeros_(module.k_proj.bias)
            nn.init.zeros_(module.v_proj.bias)
            nn.init.zeros_(module.out_proj.bias)
        elif isinstance(module, Phi4MultimodalVisionMLP):
            nn.init.normal_(module.fc1.weight)
            nn.init.normal_(module.fc2.weight)
            nn.init.normal_(module.fc1.bias, std=1e-6)
            nn.init.normal_(module.fc2.bias, std=1e-6)
        elif isinstance(module, (nn.Linear, nn.Conv2d)):
            lecun_normal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


class Phi4MultimodalVisionEmbeddings(SiglipVisionEmbeddings, nn.Module):
    def __init__(self, config: Phi4MultimodalVisionConfig):
        nn.Module.__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size

        self.patch_embedding = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            padding="valid",
        )

        self.num_patches_per_side = self.image_size // self.patch_size
        self.num_patches = self.num_patches_per_side**2
        self.num_positions = self.num_patches
        self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim)

    def forward(self, pixel_values: torch.FloatTensor, patch_attention_mask: torch.BoolTensor) -> torch.Tensor:
        batch_size = pixel_values.size(0)

        patch_embeds = self.patch_embedding(pixel_values)
        embeddings = patch_embeds.flatten(2).transpose(1, 2)

        max_im_h, max_im_w = pixel_values.size(2), pixel_values.size(3)
        max_nb_patches_h, max_nb_patches_w = max_im_h // self.patch_size, max_im_w // self.patch_size
        boundaries = torch.arange(1 / self.num_patches_per_side, 1.0, 1 / self.num_patches_per_side)
        position_ids = torch.full(
            size=(
                batch_size,
                max_nb_patches_h * max_nb_patches_w,
            ),
            fill_value=0,
        )

        for batch_idx, p_attn_mask in enumerate(patch_attention_mask):
            nb_patches_h = p_attn_mask[:, 0].sum()
            nb_patches_w = p_attn_mask[0].sum()

            fractional_coords_h = torch.arange(0, 1 - 1e-6, 1 / nb_patches_h)
            fractional_coords_w = torch.arange(0, 1 - 1e-6, 1 / nb_patches_w)

            bucket_coords_h = torch.bucketize(fractional_coords_h, boundaries, right=True)
            bucket_coords_w = torch.bucketize(fractional_coords_w, boundaries, right=True)

            pos_ids = (bucket_coords_h[:, None] * self.num_patches_per_side + bucket_coords_w).flatten()
            position_ids[batch_idx][p_attn_mask.view(-1).cpu()] = pos_ids

        position_ids = position_ids.to(self.position_embedding.weight.device)

        embeddings = embeddings + self.position_embedding(position_ids)
        return embeddings


class Phi4MultimodalVisionMultiheadAttentionPoolingHead(SiglipMultiheadAttentionPoolingHead):
    def forward(self, hidden_state, attention_mask):
        batch_size = hidden_state.shape[0]
        probe = self.probe.repeat(batch_size, 1, 1)

        hidden_state = self.attention(
            query=probe, key=hidden_state, value=hidden_state, key_padding_mask=~attention_mask
        )[0]

        residual = hidden_state
        hidden_state = self.layernorm(hidden_state)
        hidden_state = residual + self.mlp(hidden_state)

        return hidden_state[:, 0]


class Phi4MultimodalVisionModel(SiglipVisionTransformer, Phi4MultimodalVisionPreTrainedModel):
    config_class = Phi4MultimodalVisionConfig
    main_input_name = "pixel_values"

    def __init__(self, config: Phi4MultimodalVisionConfig):
        Phi4MultimodalVisionPreTrainedModel.__init__(config)
        self.config = config
        embed_dim = config.hidden_size

        self.embeddings = Phi4MultimodalVisionEmbeddings(config)
        self.encoder = Phi4MultimodalVisionEncoder(config)
        self.post_layernorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)
        self.head = Phi4MultimodalVisionMultiheadAttentionPoolingHead(config)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self) -> nn.Module:
        return self.embeddings.patch_embedding

    def forward(
        self,
        pixel_values,
        patch_attention_mask: Optional[torch.BoolTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        batch_size = pixel_values.size(0)
        if patch_attention_mask is None:
            patch_attention_mask = torch.ones(
                size=(
                    batch_size,
                    pixel_values.size(2) // self.config.patch_size,
                    pixel_values.size(3) // self.config.patch_size,
                ),
                dtype=torch.bool,
                device=pixel_values.device,
            )

        hidden_states = self.embeddings(pixel_values=pixel_values, patch_attention_mask=patch_attention_mask)

        patch_attention_mask = patch_attention_mask.view(batch_size, -1)
        # The call to `_upad_input` in `_flash_attention_forward` is expensive
        # So when the `patch_attention_mask` is full of 1s (i.e. attending to the whole sequence),
        # avoiding passing the attention_mask, which is equivalent to attending to the full sequence
        if not torch.any(~patch_attention_mask):
            attention_mask = None
        else:
            attention_mask = (
                _prepare_4d_attention_mask(patch_attention_mask, hidden_states.dtype)
                if not self.config._flash_attn_2_enabled
                else patch_attention_mask
            )

        encoder_outputs = self.encoder(
            inputs_embeds=hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        last_hidden_state = encoder_outputs[0]
        last_hidden_state = self.post_layernorm(last_hidden_state)

        pooled_output = self.head(
            hidden_state=last_hidden_state,
            attention_mask=patch_attention_mask,
        )

        if not return_dict:
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


class Phi4MultimodalImageEmbedding(nn.Module):
    """Image embedding."""

    def __init__(self, config: PretrainedConfig):
        super().__init__()

        # n_embed or hidden_size
        hidden_size = config.hidden_size
        self.drop = nn.Dropout(config.embd_pdrop)

        self.img_processor = Phi4MultimodalVisionModel(config.vision_config)

        pe_weight = self.img_processor.embeddings.position_embedding.weight
        L, D = pe_weight.size()
        H = int(math.sqrt(L))
        assert H**2 == L
        if H % 2 != 0:
            self.img_processor_padding = nn.ReflectionPad2d((0, 1, 0, 1))
            H += 1
        self.num_img_tokens = (H // 2) ** 2
        self.base_feat_height_target = H

        self.image_dim_out = D
        self.img_sizes = None
        self.image_attention_mask = None

        # global_gn and sub_gn for hd transform, serves as line separator
        self.use_hd_transform = config.vision_config.image_embd_layer["use_hd_transform"]
        self.with_learnable_separator = config.vision_config.image_embd_layer["with_learnable_separator"]
        self.hd_transform_order = config.vision_config.image_embd_layer["hd_transform_order"]
        self.freeze_img_processor = False
        self.crop_size = config.vision_config.image_embd_layer["crop_size"]
        assert (
            self.use_hd_transform == self.with_learnable_separator
        ), "use_hd_transform and with_learnable_separator should have same value"

        # image token compression
        self.image_token_compression = nn.AvgPool2d(kernel_size=2, stride=2)
        self.base_feat_height_reduction = 1
        self.base_feat_height_target = self.base_feat_height_target // 2

        if self.with_learnable_separator:
            # 1024 * 4, merge spatial to channel dimension
            self.glb_GN = nn.Parameter(torch.zeros([1, 1, self.image_dim_out * self.base_feat_height_reduction**2]))
            self.sub_GN = nn.Parameter(torch.zeros([1, 1, 1, self.image_dim_out * self.base_feat_height_reduction**2]))

        projection_cls = config.vision_config.image_embd_layer["projection_cls"]
        if projection_cls == "linear":
            self.img_projection = nn.Linear(self.image_dim_out, hidden_size)
        elif projection_cls == "mlp":
            dim_projection = hidden_size
            first_dim = (
                self.image_dim_out * self.base_feat_height_reduction**2
                if self.use_hd_transform
                else self.image_dim_out
            )
            layers = [nn.Linear(first_dim, dim_projection)]
            layers.extend([nn.GELU(), nn.Linear(dim_projection, dim_projection)])
            self.img_projection = nn.Sequential(*layers)
        else:
            raise NotImplementedError(f"projection_cls = {projection_cls}, not implemented")

        self.vocab_size = config.vocab_size
        self.img_features = None

        self.layer_idx = -2
        self.type_feature = "patch"

    def set_img_features(self, img_features: torch.FloatTensor) -> None:
        self.img_features = img_features

    def set_img_sizes(self, img_sizes: torch.LongTensor) -> None:
        self.img_sizes = img_sizes

    def set_img_attn_mask(self, image_attention_mask: torch.FloatTensor) -> None:
        self.image_attention_mask = image_attention_mask

    def get_img_features(self, img_embeds: torch.FloatTensor, attention_mask=None) -> torch.FloatTensor:
        img_processor_output = self.img_processor(
            img_embeds, patch_attention_mask=attention_mask, output_hidden_states=True
        )
        img_feature = img_processor_output.hidden_states[self.layer_idx]

        patch_feature = img_feature
        # reshape to 2D tensor
        width = int(math.sqrt(patch_feature.size(1)))
        patch_feature = patch_feature.view(-1, width, width, patch_feature.size(-1))
        # convert to NCHW
        patch_feature = patch_feature.permute(0, 3, 1, 2)
        if getattr(self, "img_processor_padding", None) is not None:
            patch_feature = self.img_processor_padding(patch_feature)
        patch_feature = self.image_token_compression(patch_feature)
        # convert to NHWC
        patch_feature = patch_feature.permute(0, 2, 3, 1)
        patch_feature = patch_feature.view(-1, patch_feature.size(1) * patch_feature.size(2), patch_feature.size(-1))
        return patch_feature

    def forward(
        self, input_ids: torch.LongTensor, input_embeds: torch.FloatTensor, image_sizes=None, **kwargs
    ) -> torch.FloatTensor:
        img_embeds = input_embeds
        img_sizes = image_sizes

        if self.img_features is not None:
            img_embeds = self.img_features.clone()
            self.img_features = None

        if self.img_sizes is not None:
            img_sizes = self.img_sizes

        dtype = self.img_processor.embeddings.patch_embedding.weight.dtype
        if img_embeds is not None:
            img_embeds = img_embeds.to(dtype)

        if self.image_attention_mask is not None:
            image_attention_mask = self.image_attention_mask.clone()
            self.image_attention_mask = None
        elif "image_attention_mask" in kwargs:
            image_attention_mask = kwargs["image_attention_mask"]
        else:
            image_attention_mask = None
        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_shape[-1])

        with torch.no_grad():
            positions = torch.nonzero(input_ids == _IMAGE_SPECIAL_TOKEN_ID, as_tuple=False)
            positions_tuple = torch.nonzero(input_ids == _IMAGE_SPECIAL_TOKEN_ID, as_tuple=True)

        fake_image_forward = False
        select = False
        hd_transform = False

        if isinstance(self.img_projection, nn.Sequential):
            target_device = self.img_projection[0].bias.device
            target_dtype = self.img_projection[0].bias.dtype
        else:  # It's a single nn.Linear layer
            target_device = self.img_projection.bias.device
            target_dtype = self.img_projection.bias.dtype

        if len(positions.tolist()) > 0:
            select = True
            hd_transform = True
            assert (
                img_embeds.ndim == 5
            ), f"(branch 1) img_embeds size: {img_embeds.size()}, expect 5D tensor for hd transform"

            bs = img_embeds.shape[0]
            # Nx(HW)xC
            img_features = self.get_img_features(
                img_embeds.flatten(0, 1),
                attention_mask=image_attention_mask.type(torch.BoolTensor).flatten(0, 1).to(target_device),
            )

            base_feat_height_target = self.base_feat_height_target
            base_resolution = self.crop_size
            base_feat_height_reduction = self.base_feat_height_reduction

            base_feat_height = base_feat_width = int(np.sqrt(img_features.shape[1]))

            assert (
                base_feat_height == base_feat_height_target and base_feat_width == base_feat_height_target
            ), f"base_feat_height: {base_feat_height}, base_feat_width: {base_feat_width}, expect {base_feat_height_target} features for hd transform"

            # bs x max_num_crops x (24x24) x C
            img_features = img_features.view(bs, -1, base_feat_height * base_feat_width, self.image_dim_out)
            C = self.image_dim_out
            H = base_feat_height

            output_imgs = []
            output_len = []
            # training is tensor, inference is list
            if isinstance(img_sizes, torch.Tensor):
                img_sizes = img_sizes.view(-1, 2)
            for _bs in range(bs):
                h, w = img_sizes[_bs]
                h = h // base_resolution
                w = w // base_resolution
                B_ = h * w

                # 1 x (24x24) x 1024
                global_img_feature = img_features[_bs, :1]

                # 1 x 12 x 12 x 4096
                glb_img = (
                    global_img_feature.reshape(1, H, H, C)
                    .reshape(
                        1,
                        H // base_feat_height_reduction,
                        base_feat_height_reduction,
                        H // base_feat_height_reduction,
                        base_feat_height_reduction,
                        C,
                    )
                    .contiguous()
                    .permute(0, 1, 3, 2, 4, 5)
                    .reshape(
                        1,
                        H // base_feat_height_reduction,
                        H // base_feat_height_reduction,
                        base_feat_height_reduction * base_feat_height_reduction * C,
                    )
                    .contiguous()
                )
                temp_glb_GN = self.sub_GN.repeat(1, H // base_feat_height_reduction, 1, 1)

                # 1 x 156 x 4096
                glb_img = torch.cat([glb_img, temp_glb_GN], dim=2).reshape(
                    1, -1, base_feat_height_reduction * base_feat_height_reduction * C
                )

                # (max_num_crops-1) x (12x12) x C
                sub_img = img_features[_bs, 1:]
                # 16x574x1024
                # get rid of padding sub_img
                sub_img = sub_img[:B_]

                # (num_crops, 12, 2, 12, 2, 1024) -> (num_crops, 12, 12, 2, 2, 1024) -> (num_crops, 12*12, 4*1024)
                sub_img = (
                    sub_img.reshape(B_, H, H, C)
                    .reshape(
                        B_,
                        H // base_feat_height_reduction,
                        base_feat_height_reduction,
                        H // base_feat_height_reduction,
                        base_feat_height_reduction,
                        C,
                    )
                    .contiguous()
                    .permute(0, 1, 3, 2, 4, 5)
                    .reshape(B_, -1, base_feat_height_reduction * base_feat_height_reduction * C)
                    .contiguous()
                )
                sub_img = (
                    sub_img.reshape(
                        1,
                        h,
                        w,
                        base_feat_height // base_feat_height_reduction,
                        base_feat_width // base_feat_height_reduction,
                        -1,
                    )
                    .permute(0, 1, 3, 2, 4, 5)
                    .reshape(
                        1,
                        h * base_feat_height // base_feat_height_reduction,
                        w * base_feat_width // base_feat_height_reduction,
                        base_feat_height_reduction * base_feat_height_reduction * C,
                    )
                )

                if image_attention_mask is not None and len(image_attention_mask) > 0:
                    reshaped_image_attention_mask = (
                        image_attention_mask[_bs, 1 : B_ + 1, 0::2, 0::2]
                        .reshape(
                            1,
                            h,
                            w,
                            base_feat_height // base_feat_height_reduction,
                            base_feat_width // base_feat_height_reduction,
                        )
                        .permute(0, 1, 3, 2, 4)
                        .reshape(
                            1,
                            h * base_feat_height // base_feat_height_reduction,
                            w * base_feat_width // base_feat_height_reduction,
                        )
                    )
                    useful_height = int(reshaped_image_attention_mask[0, :, 0].sum().item())
                    useful_width = int(reshaped_image_attention_mask[0, 0, :].sum().item())
                    sub_img = sub_img[:, :useful_height, :useful_width]
                    temp_sub_GN = self.sub_GN.repeat(1, useful_height, 1, 1)
                    temp_len = (
                        int(image_attention_mask[_bs, : B_ + 1, 0::2, 0::2].sum().item())
                        + (useful_height + 1)
                        + base_feat_height // base_feat_height_reduction
                    )
                else:
                    temp_sub_GN = self.sub_GN.repeat(1, h * base_feat_height // base_feat_height_reduction, 1, 1)
                    temp_len = int(
                        (h * w + 1) * self.num_img_tokens
                        + 1
                        + (h + 1) * base_feat_height // base_feat_height_reduction
                    )

                sub_img = torch.cat([sub_img, temp_sub_GN], dim=2).reshape(
                    1, -1, base_feat_height_reduction * base_feat_height_reduction * C
                )
                # (1, num_img_tokens, 1024*4)

                # glb + sub
                if self.hd_transform_order == "glb_sub":
                    output_imgs.append(torch.cat([glb_img, self.glb_GN, sub_img], dim=1))
                elif self.hd_transform_order == "sub_glb":
                    output_imgs.append(torch.cat([sub_img, self.glb_GN, glb_img], dim=1))
                else:
                    raise NotImplementedError(f"hd_transform_order = {self.hd_transform_order}, not implemented")

                # temp_len = int((h*w+1)*144 + 1 + (h+1)*12)
                assert (
                    temp_len == output_imgs[-1].shape[1]
                ), f"temp_len: {temp_len}, output_imgs[-1].shape[1]: {output_imgs[-1].shape[1]}"
                output_len.append(temp_len)

            img_set_tensor = []
            for _output_img in output_imgs:
                img_feature_proj = self.img_projection(_output_img.to(target_device).to(target_dtype))
                img_set_tensor.append(img_feature_proj)
            # logger.info(f'img_embeds size: {img_embeds.size()}, image sizes: {img_sizes} loading time {datetime.now() - start_time}')
            # assert sum(num_img_tokens) == len(g_values), f'(branch 1) sum(num_img_tokens): {sum(num_img_tokens)}, g_values size: {len(g_values)}, g_values {g_values}'

        else:
            # create a fake image tensor
            if self.training:
                img_embeds = torch.zeros(
                    1, 3, self.crop_size, self.crop_size, dtype=target_dtype, device=input_ids.device
                )

                tt = self.get_img_features(img_embeds).to(target_device).to(target_dtype).reshape(-1, 1024)
                if self.use_hd_transform:
                    img_set_tensor = self.img_projection(
                        tt.reshape(-1, self.image_dim_out * self.base_feat_height_reduction**2)
                        * self.glb_GN[0]
                        * self.sub_GN[0, 0]
                    )
                else:
                    img_set_tensor = self.img_projection(tt)  # adapted visual features.
                fake_image_forward = True

        # we use the token embedding layer from the huggingface model, this is REQUIRED to make sure we are using the loaded weights.
        hidden_states = kwargs["wte"](input_ids)

        if select:
            # img_set_tensor: a list of tensors, each tensor has shape (1, N_tokens, C)
            assert all(
                [_img_set_tensor.shape[0] == 1 for _img_set_tensor in img_set_tensor]
            ), "img_set_tensor should have shape (1, N_tokens, C)"
            # Shape: (merged_N_tokens, C)
            merged_img_set_tensor = torch.cat(img_set_tensor, dim=1).squeeze(0)
            merged_img_set_tensor = merged_img_set_tensor.to(hidden_states.dtype).to(hidden_states.device)
            # Temporarily disable autocast to avoid issue on bf16 tensors
            # Ref: https://github.com/pytorch/pytorch/issues/132715
            with torch.autocast(device_type=hidden_states.device.type, enabled=False):
                new_hidden_states = hidden_states.index_put(
                    indices=positions_tuple, values=merged_img_set_tensor, accumulate=False
                )
            hidden_states = new_hidden_states

        if fake_image_forward and self.training:
            hidden_states = (
                hidden_states + (0 * img_set_tensor[0].to(hidden_states.dtype).to(hidden_states.device)).sum()
            )

        if self.drop is not None:
            hidden_states = self.drop(hidden_states)

        return hidden_states


########################################################## AUDIO #############################################


class Phi4MultimodalAudioMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layer_norm = nn.LayerNorm(config.hidden_size)
        self.act_fn = ACT2FN[config.bias_in_glu]
        # ALL AFTER THIS WAS INSIDE A nn.Sequntial CALLED `net` -> KEY CONVERSION
        # gate_up_proj was additionally inside a GLULinear module with `linear` name inside
        self.gate_up_proj = nn.Linear(config.hidden_size, config.intermediate_size * 2, config.bias_in_glu)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size)
        self.dropout = (nn.Dropout(config.dropout_rate),)

    def forward(self, hidden_states):
        up_states = self.gate_up_proj(hidden_states)
        up_states, gate = up_states.chunk(2, dim=-1)
        up_states = up_states * self.act_fn(gate)
        up_states = self.dropout(up_states)
        hidden_states = self.down_proj(up_states)
        out = self.dropout(out)

        return out


def audio_eager_attention_forward(
    module: nn.Module,
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
):
    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


class Phi4MultimodalAudioAttention(nn.Module):
    def __init__(self, config):
        self.config = config
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.dropout_rate
        self.is_causal = True

        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=True)

    def forward(
        self,
        hidden_states: torch.Tensor,
        mask: Optional[torch.Tensor],
        relative_attention_bias: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        attention_mask = None
        if mask is not None:
            mask = mask.unsqueeze(1)
            if relative_attention_bias is not None:
                attention_mask = mask + relative_attention_bias
            else:
                attention_mask = mask

        attention_interface: Callable = audio_eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, _ = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output


class Phi4MultimodalAudioDepthWiseSeperableConv1d(nn.Module):
    def __init__(self, config, padding=0):
        super().__init__()
        self.dw_conv = nn.Conv1d(
            config.hidden_size,
            config.hidden_size * config.depthwise_multiplier,
            config.kernel_size,
            1,
            padding=padding,
            groups=config.hidden_size,
        )
        self.pw_conv = nn.Conv1d(
            config.hidden_size * config.depthwise_multiplier, config.depthwise_seperable_out_channel, 1, 1, 0
        )

    def forward(self, x):
        x = self.dw_conv(x)
        x = self.pw_conv(x)
        return x


class Phi4MultimodalAudioGluPointWiseConv(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.output_dim = config.ext_pw_out_channel
        kernel_size = config.ext_pw_kernel_size

        self.ext_pw_conv_1d = nn.Conv1d(
            config.hidden_size,
            config.ext_pw_out_channel * 2,
            kernel_size,
            1,
            padding=(kernel_size - 1) if config.causal else (kernel_size - 1) // 2,
        )

        self.glu_act = ACT2FN[config.glu_type]

        if config.bias_in_glu:
            self.b1 = nn.Parameter(torch.zeros(1, config.ext_pw_out_channel, 1))
            self.b2 = nn.Parameter(torch.zeros(1, config.ext_pw_out_channel, 1))

    def forward(self, x):
        # to be consistent with GLULinear, we assume the input always has the #channel (#dim) in the last dimension of the
        # tensor, so need to switch the dimension first for 1D-Conv case
        x = x.permute([0, 2, 1])
        x = self.ext_pw_conv_1d(x)
        if self.bias_in_glu:
            x = (x[:, 0 : self.output_dim, :] + self.b1) * self.glu_act(
                x[:, self.output_dim : self.output_dim * 2, :] + self.b2
            )
        else:
            x = (x[:, 0 : self.output_dim, :]) * self.glu_act(x[:, self.output_dim : self.output_dim * 2, :])

        x = x.permute([0, 2, 1])
        return x


class Phi4MultimodalAudioConvModule(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.batch_norm = config.batch_norm
        self.kernel_size = config.kernel_size

        self.layer_norm = nn.LayerNorm(config.hidden_size)
        self.glu = Phi4MultimodalAudioGluPointWiseConv(config)
        self.ln1 = (
            nn.Linear(config.ext_pw_out_channel, config.hidden_size)
            if config.hidden_size != config.ext_pw_out_channel
            else nn.Identity()
        )

        if config.causal and config.export:
            padding = 0
        elif config.causal:
            padding = config.kernel_size - 1
        else:
            padding = (config.kernel_size - 1) // 2
        self.dw_sep_conv_1d = Phi4MultimodalAudioDepthWiseSeperableConv1d(config, padding=padding)

        if config.hidden_size != config.depthwise_seperable_out_channel:
            self.ln2 = nn.Linear(config.depthwise_seperable_out_channel, config.hidden_size)
        if config.batch_norm:
            self.bn_layer = nn.BatchNorm1d(config.hidden_size)

        self.act = ACT2FN[config.activation]

        self.ext_pw_conv_1d = nn.Conv1d(
            config.hidden_size,
            config.ext_pw_out_channel,
            config.ext_pw_kernel_size,
            1,
            padding=config.ext_pw_kernel_size - 1 if config.causal else (config.ext_pw_kernel_size - 1) // 2,
        )
        self.fix_len1 = True if config.causal and config.ext_pw_kernel_size > 1 else False
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(self, x):
        x = self.layer_norm(x)
        x = self.glu(x)
        if self.causal and self.ext_pw_kernel_size > 1:
            x = x[:, : -(self.ext_pw_kernel_size - 1), :]
        x = self.ln1(x)

        x = x.permute([0, 2, 1])

        x = self.dw_sep_conv_1d(x)
        if self.causal and self.kernel_size > 1:
            x = x[:, :, : -(self.kernel_size - 1)]
        if hasattr(self, "ln2"):
            x = x.permute([0, 2, 1])
            x = self.ln2(x)
            x = x.permute([0, 2, 1])
        if self.batch_norm:
            x = self.bn_layer(x)
        x = self.act(x)

        x = self.ext_pw_conv_1d(x)
        if self.fix_len1:
            x = x[:, :, : -(self.ext_pw_kernel_size - 1)]

        if self.apply_ln1:
            x = x.permute([0, 2, 1])
            x = self.ln1(x)
            x = x.permute([0, 2, 1])

        x = x.permute([0, 2, 1])
        x = self.dropout(x)
        return x


class Phi4MultimodalAudioConformerEncoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.feed_forward_in = Phi4MultimodalAudioMLP(config)
        self.self_attn = Phi4MultimodalAudioAttention(config)
        self.conv = Phi4MultimodalAudioConvModule(config)
        self.feed_forward_out = Phi4MultimodalAudioMLP(config)
        self.layer_norm_att = nn.LayerNorm(config.hidden_size)
        self.layer_norm = nn.LayerNorm(config.hidden_size)

    def forward(
        self,
        x,
        pos_k,
        pos_v,
        mask,
        relative_attention_bias: Optional[torch.Tensor] = None,
    ):
        """ConformerEncoder forward.

        Args:
            x: torch.Tensor
                input feature of shape (batch, max_time_in, size)
            pos_k: torch.Tensor
                positional key embedding.
            mask: torch.Tensor
                mask for x (batch, max_time_in)
            relative_attention_bias: Optional[torch.Tensor]
                bias added to attention logits w.r.t. relative positions (1, n_head, time1, time2)
        """
        x = x + 0.5 * self.feed_forward_in(x)
        norm_x = self.layer_norm_att(x)

        x = x + self.self_attn(
            norm_x,
            pos_k,
            pos_v,
            mask,
            relative_attention_bias=relative_attention_bias,
        )
        x = x + self.conv(x)
        x = x + 0.5 * self.feed_forward_out(x)

        out = self.layer_norm(x)

        return out


class Phi4MultimodalAudioNemoConvSubsampling(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.subsampling_factor = self.config.nemo_conv_settings["subsampling_factor"]

        if self.subsampling_factor % 2 != 0:
            raise ValueError("Sampling factor should be a multiply of 2!")
        self.sampling_num = int(math.log(self.subsampling_factor, 2))

        self.act_fn = ACT2CLS[self.config.nemo_conv_settings["activation"]]

        conv_channels = self.config.nemo_conv_settings["conv_channels"]
        layers = []

        layers.append(
            torch.nn.Conv2d(
                in_channels=1,
                out_channels=conv_channels,
                kernel_size=3,
                stride=2,
                padding=1,
            )
        )
        layers.append(self.act_fn)

        for _ in range(self.sampling_num - 1):
            layers.append(
                torch.nn.Conv2d(
                    in_channels=conv_channels,
                    out_channels=conv_channels,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    groups=conv_channels,
                )
            )
            layers.append(
                torch.nn.Conv2d(
                    in_channels=conv_channels,
                    out_channels=conv_channels,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    groups=1,
                )
            )
            layers.append(self.act_fn)

        self.conv = torch.nn.Sequential(*layers)

        in_length = torch.tensor(config.hidden_size, dtype=torch.float)
        out_length = calc_length(
            lengths=in_length,
            all_paddings=2,
            kernel_size=3,
            stride=2,
            ceil_mode=False,
            repeat_num=self.sampling_num,
        )
        self.out = torch.nn.Linear(conv_channels * int(out_length), config.hidden_size)

    def forward(self, x, mask):
        """
        Forward method for NeMo subsampling.

        Args:
            x[Batch, Time, Filters]: torch.Tensor
                input tensor
            x_mask: torch.Tensor
                input mask

        Returns:
            x: torch.Tensor
                Resulting tensor from subsampling (B, T // time_reduction_factor, feat_out)
            pad_mask: torch.Tensor
                tensor of padded hidden state sequences (B, 1, T // time_reduction_factor)
        """
        # Unsqueeze Channel Axis
        x = x.unsqueeze(1)

        x = self.conv(x)

        # Flatten Channel and Frequency Axes
        b, c, t, f = x.size()
        x = self.out(x.transpose(1, 2).reshape(b, t, -1))

        if mask is None:
            return x, None

        max_audio_length = x.shape[1]
        feature_lens = mask.sum(1)
        padding_length = torch.ceil(feature_lens / self.subsampling_factor)
        pad_mask = torch.arange(0, max_audio_length, device=x.device).expand(
            padding_length.size(0), -1
        ) < padding_length.unsqueeze(1)
        return x, pad_mask.unsqueeze(1)


def calc_length(lengths, all_paddings, kernel_size, stride, ceil_mode, repeat_num=1):
    """Calculates the output length of a Tensor passed through a convolution or max pooling layer"""
    add_pad: float = all_paddings - kernel_size
    one: float = 1.0
    for i in range(repeat_num):
        lengths = torch.div(lengths.to(dtype=torch.float) + add_pad, stride) + one
        if ceil_mode:
            lengths = torch.ceil(lengths)
        else:
            lengths = torch.floor(lengths)
    return lengths.to(dtype=torch.int)


class Phi4MultimodalAudioRelativeAttentionLogitBias(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.num_heads = config.num_attention_heads
        self.max_distance = config.relative_attention_bias_args.get("t5_bias_max_distance", 1000)
        self.symmetric = config.relative_attention_bias_args.get("t5_bias_symmetric", False)
        self.num_buckets = self.max_distance
        if not self.symmetric:
            self.num_buckets *= 2
        self.bias_values = nn.Embedding(self.num_buckets, self.num_heads)

    def forward(self, x):
        # instantiate bias compatible with shape of x
        max_pos = x.size(1)
        context_position = torch.arange(max_pos, device=x.device, dtype=torch.long)[:, None]
        memory_position = torch.arange(max_pos, device=x.device, dtype=torch.long)[None, :]
        relative_position = memory_position - context_position
        # clipping to a maximum distance using ops that play well with ONNX export
        relative_position = relative_position.masked_fill(relative_position < -self.max_distance, -self.max_distance)
        relative_position = relative_position.masked_fill(
            relative_position > self.max_distance - 1, self.max_distance - 1
        )

        # mapping from relative position to index in the bias parameter
        bias_idx = relative_position
        if self.symmetric:
            bias_idx = bias_idx.abs()
        else:
            bias_idx += self.num_buckets // 2

        t5_rel_att_bias = self.bias_values(bias_idx)  # [L, L, H]
        t5_rel_att_bias = t5_rel_att_bias.permute(2, 0, 1).unsqueeze(0)  # [1, H, L, L]

        return t5_rel_att_bias


class Phi4MultimodalAudioMeanVarianceNormLayer(nn.Module):
    """Mean/variance normalization layer.

    Will substract mean and multiply input by inverted standard deviation.
    Typically used as a very first layer in a model.

    Args:
        input_size: int
            layer input size.
    """

    # TODO: IT APPEARS THEY NEVER UPDATE THIS SO PROBABLY REMOVE
    def __init__(self, config):
        super().__init__()
        self.register_buffer(
            "global_mean", torch.zeros(config.encoder_embedding_config["input_size"]), persistent=False
        )
        self.register_buffer(
            "global_invstd", torch.ones(config.encoder_embedding_config["input_size"]), persistent=False
        )

    def forward(self, x):
        return (x - self.global_mean) * self.global_invstd


class Phi4MultimodalAudioConformerEncoder(nn.Module):
    extra_multi_layer_output_idxs: List[int]

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.encoder_embedding = Phi4MultimodalAudioMeanVarianceNormLayer(config)
        self.embed = Phi4MultimodalAudioNemoConvSubsampling(config)
        self.relative_attention_bias_layer = Phi4MultimodalAudioRelativeAttentionLogitBias(config)
        self.encoders = nn.ModuleList(
            [Phi4MultimodalAudioConformerEncoderLayer(config) for _ in range(config.num_blocks)]
        )

    def _chunk_size_selection(self, chunk_size=None, left_chunk=None):
        """If chunk size is a list, we will randomly select a chunk size."""

        if chunk_size is None:
            chunk_size = self.chunk_size
        if left_chunk is None:
            left_chunk = self.left_chunk
        if isinstance(chunk_size, list):
            # Variable chunk size during training
            chunk_size_index = int(torch.randint(low=0, high=len(chunk_size), size=(1,)))
            chunk_size_train_eff = chunk_size[chunk_size_index]
            if not isinstance(left_chunk, list):
                raise ValueError("Since chunk_size is a list, left_chunk must be a list")
            if len(left_chunk) != len(chunk_size):
                raise ValueError("The length of left_chunk must be the same as length of chunk_size.")
            left_chunk_train_eff = left_chunk[chunk_size_index]
        else:
            chunk_size_train_eff = chunk_size
            left_chunk_train_eff = left_chunk

        return chunk_size_train_eff, left_chunk_train_eff

    def _streaming_mask(self, seq_len, batch_size, chunk_size, left_chunk):
        chunk_size_train_eff, left_chunk_train_eff = self._chunk_size_selection(chunk_size, left_chunk)

        # Create mask matrix for streaming
        # S stores start index. if chunksize is 18, s is [0,18,36,....]
        chunk_start_idx = np.arange(0, seq_len, chunk_size_train_eff)
        # avoid randomness when run evaluation or decoding
        if self.training and np.random.rand() > 0.5:
            # Either first or last chunk is not complete.
            # If only the last one is not complete, EOS is not effective
            chunk_start_idx = seq_len - chunk_start_idx
            chunk_start_idx = chunk_start_idx[::-1]
            chunk_start_idx = chunk_start_idx[:-1]
            chunk_start_idx = np.insert(chunk_start_idx, 0, 0)

        enc_streaming_mask = (
            adaptive_enc_mask(seq_len, chunk_start_idx, left_window=left_chunk_train_eff)
            .unsqueeze(0)
            .expand([batch_size, -1, -1])
        )
        return enc_streaming_mask

    def forward_embeddings(self, xs_pad, masks):
        """Forwarding the inputs through the top embedding layers

        Args:
            xs_pad: torch.Tensor
                input tensor
            masks: torch.Tensor
                input mask
        """
        # pylint: disable=R0915
        # get new lens.
        seq_len = math.ceil(xs_pad.shape[1] / self.config.time_reduction)
        if seq_len <= 0:
            raise ValueError(
                f"""The squence length after time reduction is invalid: {seq_len}.
                Your input feature is too short. Consider filtering out the very
                short sentence from data loader""",
            )

        batch_size = xs_pad.shape[0]

        enc_streaming_mask = self._streaming_mask(seq_len, batch_size, self.config.chunk_size, self.config.left_chunk)

        if xs_pad.is_cuda:
            enc_streaming_mask = enc_streaming_mask.cuda()
            xs_pad = xs_pad.cuda()

        input_tensor = xs_pad
        input_tensor, masks = self.embed(input_tensor, masks)

        streaming_mask = enc_streaming_mask
        if streaming_mask is not None and masks is not None:
            hs_mask = masks & streaming_mask
        elif masks is not None:
            hs_mask = masks
        else:
            hs_mask = streaming_mask

        return input_tensor, hs_mask, masks

    def calculate_hs_mask(self, xs_pad, device, mask):
        max_audio_length = xs_pad.shape[1]
        batch_size = xs_pad.shape[0]
        enc_streaming_mask = self._streaming_mask(max_audio_length, batch_size, self.chunk_size, self.left_chunk)
        enc_streaming_mask = enc_streaming_mask.to(device)
        if mask is None:
            return enc_streaming_mask

        feature_lens = mask.sum(1)
        padding_length = feature_lens
        pad_mask = torch.arange(0, max_audio_length, device=device).expand(
            padding_length.size(0), -1
        ) < padding_length.unsqueeze(1)
        pad_mask = pad_mask.unsqueeze(1)
        pad_mask = pad_mask & enc_streaming_mask
        return pad_mask

    def forward(self, xs_pad, masks):
        """Conformer Forward function

        Args:
            xs_pad: torch.Tensor
                input tensor
            masks: torch.Tensor
                post-embedding input lengths
        """
        xs_pad = self.encoder_embedding(xs_pad)
        input_tensor, hs_mask, masks = self.forward_embeddings(xs_pad, masks)

        unfolded = False
        ori_bz, seq_len, D = input_tensor.shape
        max_seq_len = 500  # maxium position for absolute positional encoding
        if seq_len > max_seq_len:
            # audio sequence is longer than max_seq_len, unfold it into chunks of max_seq_len
            unfolded = True
            # the unfold op will drop residual frames, pad it to the multiple of max_seq_len
            if seq_len % max_seq_len > 0:
                chunk_pad_size = max_seq_len - (seq_len % max_seq_len)
            else:
                chunk_pad_size = 0
            if chunk_pad_size > 0:
                input_tensor_pad = F.pad(input_tensor, (0, 0, 0, chunk_pad_size), "constant", 0)
                input_tensor = input_tensor_pad.to(input_tensor.device)

            input_tensor = unfold_tensor(input_tensor, max_seq_len)
            if masks is not None:
                # revise hs_mask here because the previous calculated hs_mask did not consider extra pad
                subsampled_pad_mask = masks.squeeze(1)  # [bz, subsampled_unmask_seq_len]
                extra_padded_subsamlped_pad_mask = F.pad(
                    subsampled_pad_mask, (0, chunk_pad_size), "constant", False
                )  # extra padding to the pad mask
                extra_padded_subsamlped_pad_mask = extra_padded_subsamlped_pad_mask.unsqueeze(-1).float()
                masks_unfold = unfold_tensor(
                    extra_padded_subsamlped_pad_mask, max_seq_len
                )  # unfold the pad mask like we did to the input tensor
                masks_unfold = masks_unfold.squeeze(-1).bool()  # unfold op does not support bool tensor
            else:
                masks_unfold = None
            hs_mask = self.calculate_hs_mask(
                input_tensor, input_tensor.device, masks_unfold
            )  # calculate hs_mask based on the unfolded pad mask

        relative_attention_bias = self.relative_attention_bias_layer(input_tensor)

        for layer in self.encoders:
            input_tensor = layer(
                input_tensor,
                hs_mask,
                relative_attention_bias=relative_attention_bias,
            )

        if unfolded:
            embed_dim = input_tensor.shape[-1]
            input_tensor = input_tensor.reshape(ori_bz, -1, embed_dim)
            # if we ever padded before unfolding, we need to remove the padding
            if chunk_pad_size > 0:
                input_tensor = input_tensor[:, :-chunk_pad_size, :]

        return input_tensor, masks


def unfold_tensor(xs_pad, max_seq_len):
    """
    For a given tensor with shape of (N, T, D), if sequence length T is longer than max_seq_len,
    this function unfold it to a (NT', max_seq_len, D) where T' is T // max_seq_len.
    Args:
        xs_pad: N, T, D
    """
    _, _, D = xs_pad.shape
    xs_pad = xs_pad.transpose(-1, -2)  # convert to N, D, T
    # N x D x 1 x T => N x (D x max_seq_len) x T'
    xs_pad = F.unfold(
        xs_pad[..., None, :],
        kernel_size=(1, max_seq_len),
        stride=(1, max_seq_len),
    )

    new_bsz, _, slen = xs_pad.shape
    # N x D x max_seq_len x T'
    xs_pad = xs_pad.view(new_bsz, -1, max_seq_len, slen)
    # N x T' x max_seq_len x D
    xs_pad = xs_pad.permute(0, 3, 2, 1).contiguous()
    # NT' x max_seq_len x D
    xs_pad = xs_pad.view(-1, max_seq_len, D)
    return xs_pad


def adaptive_enc_mask(x_len, chunk_start_idx, left_window=0, right_window=0):
    """
    The function is very important for Transformer Transducer Streaming mode
    Args:
        xs_len (int): sequence length
        chunk_start_idx (list): first idx of each chunk, such as [0,18,36,48]. It also supports adaptive chunk size [0,10,15,45]
        left_window (int): how many left chunks can be seen
        right_window (int): how many right chunks can be seen. It is used for chunk overlap model.
        Returns:
            mask (torch.Tensor): a mask tensor for streaming model
            Torch 1.0.1
            tensor([[1., 1., 0., 0.],
                    [0., 1., 1., 0.],
                    [0., 0., 1., 1.]])
            Torch 1.4.1
            tensor([[True., True., False., False.],
                    [False., True., True., False.],
                    [False., False., True., True.]])
    """
    chunk_start_idx = torch.Tensor(chunk_start_idx).long()  # first idx of each chunk, such as [0,18,36,48].
    start_pad = torch.nn.functional.pad(
        chunk_start_idx, (1, 0)
    )  # append 0 to the beginning, so it becomes [0, 0, 18, 36, 48]
    end_pad = torch.nn.functional.pad(
        chunk_start_idx, (0, 1), value=x_len
    )  # append x_len to the end, so it becomes [0,18,36,48, x_len]
    seq_range = torch.arange(0, x_len).unsqueeze(-1)  # seq_range size: [x_len, 1]
    idx = ((seq_range < end_pad) & (seq_range >= start_pad)).nonzero()[:, 1]  # idx size: [x_len]
    boundary = end_pad[idx]  # boundary size: [x_len]
    seq_range_expand = torch.arange(0, x_len).unsqueeze(0).expand(x_len, -1)  # seq_range_expand size [x_len, x_len]
    idx_left = idx - left_window
    idx_left[idx_left < 0] = 0
    boundary_left = start_pad[idx_left]
    mask_left = seq_range_expand >= boundary_left.unsqueeze(-1)
    idx_right = idx + right_window
    idx_right[idx_right > len(chunk_start_idx)] = len(chunk_start_idx)
    boundary_right = end_pad[idx_right]
    mask_right = seq_range_expand < boundary_right.unsqueeze(-1)
    return mask_left & mask_right


class Phi4MultimodalAudioEmbedding(nn.Module):
    """Audio embedding."""

    def __init__(self, config: PretrainedConfig):
        super().__init__()
        self.config = config
        # n_embed or hidden_size for text LM
        hidden_size = config.hidden_size

        embd_drop = config.embd_pdrop
        self.drop = nn.Dropout(embd_drop)

        audio_dim_out = None  # Set this variable according to the actual audio processor
        self.layer_idx = -2

        self.encoder = Phi4MultimodalAudioConformerEncoder(config.audio_config)
        audio_dim_out = config.audio_config.hidden_size
        n_mels = config.audio_config.input_size

        self.audio_dim_out = audio_dim_out
        self.audio_dim_in = n_mels

        self.freeze_audio_processor = False

        self.downsample_rate = config.audio_config.audio_embd_layer["downsample_rate"]

        projection_cls = config.audio_config.audio_embd_layer["projection_cls"]
        if projection_cls == "linear":
            self.audio_projection = nn.Linear(audio_dim_out, hidden_size)
        elif projection_cls == "mlp":
            # follow llava-v1.5's implementation
            # (do not use image_projection and image_proj_norm)
            dim_projection = hidden_size
            self.linear_downsample_rate = self.downsample_rate

            layers_for_speech = [nn.Linear(audio_dim_out * self.linear_downsample_rate, dim_projection)]
            layers_for_speech.extend([nn.GELU(), nn.Linear(dim_projection, dim_projection)])
            audio_projection_for_speech = nn.Sequential(*layers_for_speech)

            layers_for_vision = [nn.Linear(audio_dim_out * self.linear_downsample_rate, dim_projection)]
            layers_for_vision.extend([nn.GELU(), nn.Linear(dim_projection, dim_projection)])
            audio_projection_for_vision = nn.Sequential(*layers_for_vision)

            self.audio_projection = nn.ModuleDict(
                {"speech": audio_projection_for_speech, "vision": audio_projection_for_vision}
            )

        self.vocab_size = config.vocab_size
        self.input_embeds = None
        self.audio_embed_sizes = None

    def set_audio_embeds(self, input_embeds: torch.FloatTensor) -> None:
        self.input_embeds = input_embeds

    def set_audio_embed_sizes(self, audio_embed_sizes: torch.LongTensor) -> None:
        self.audio_embed_sizes = audio_embed_sizes

    def get_audio_features(
        self,
        input_embeds: torch.FloatTensor,
        audio_attention_mask: torch.Tensor,
        audio_projection_mode: str = "speech",
    ):
        audio_features, _ = self.encoder(input_embeds, audio_attention_mask)

        if isinstance(self.audio_projection, nn.Linear):
            audio_set_tensor = self.audio_projection(audio_features)
        elif isinstance(self.audio_projection, nn.ModuleDict):
            audio_set_tensor = self.audio_projection[audio_projection_mode](audio_features)

        return audio_set_tensor

    def forward(
        self,
        input_ids: torch.LongTensor,
        input_embeds: torch.FloatTensor,
        audio_embed_sizes=None,
        audio_attention_mask=None,
        audio_projection_mode="speech",
        **kwargs,
    ) -> torch.FloatTensor:
        """
        arguments:
            input_ids: input text ids (B, U)
            input_embeds: audio features (B, T, D)  B: num audios in a sequence
        """
        if self.input_embeds is not None:
            input_embeds = self.input_embeds.clone()
        if self.audio_embed_sizes is not None:
            audio_embed_sizes = self.audio_embed_sizes.clone()

        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_shape[-1])

        with torch.no_grad():
            positions = torch.nonzero(input_ids == _AUDIO_SPECIAL_TOKEN_ID, as_tuple=False)
            positions_tuple = torch.nonzero(input_ids == _AUDIO_SPECIAL_TOKEN_ID, as_tuple=True)

        if isinstance(self.audio_projection, nn.Linear):
            target_device = self.audio_projection[0].bias.device
            target_dtype = self.audio_projection[0].bias.dtype
        elif isinstance(self.audio_projection, nn.ModuleDict):
            target_device = self.audio_projection[audio_projection_mode][0].bias.device
            target_dtype = self.audio_projection[audio_projection_mode][0].bias.dtype

        if input_embeds is not None:
            input_embeds = input_embeds.to(target_device).to(target_dtype)

        if len(positions.tolist()) > 0:
            audio_set_tensor = self.get_audio_features(input_embeds, audio_attention_mask, audio_projection_mode)
        else:
            # # create an audio tensor
            # To do: not sure if this is required for text only input
            if self.training:
                audio_embeds = torch.zeros(1, 500, self.audio_dim_in).to(target_device).to(target_dtype)
                audio_attention_mask = audio_embeds.new_ones(audio_embeds.size()[:2]).long()
                audio_set_tensor = self.get_audio_features(audio_embeds, audio_attention_mask, audio_projection_mode)

        hidden_states = kwargs["wte"](input_ids)

        if len(positions.tolist()) > 0:
            assert (
                audio_embed_sizes.sum().item() == len(positions)
            ), f"please ensure the encoder outputs have the same length as defined in input_ids! \n audio_embed_sizes.sum().item(): {audio_embed_sizes.sum().item()} \n len(positions): {len(positions)} \n audio_embed_sizes: {audio_embed_sizes} \n positions: {positions} \n input_ids.shape \n {input_ids.shape}"

            # audio_set_tensor: shape (N_audios, N_padded_tokens, C)
            # Shape: (merged_N_tokens, C)
            merged_audio_set_tensor = torch.cat(
                [audio_set_tensor[i, : audio_embed_sizes[i], :] for i in range(len(audio_embed_sizes))], dim=0
            )
            merged_audio_set_tensor = merged_audio_set_tensor.to(hidden_states.dtype).to(hidden_states.device)
            # Temporarily disable autocast to avoid issue on bf16 tensors
            # Ref: https://github.com/pytorch/pytorch/issues/132715
            with torch.autocast(device_type=hidden_states.device.type, enabled=False):
                new_hidden_states = hidden_states.index_put(
                    indices=positions_tuple, values=merged_audio_set_tensor, accumulate=False
                )
            hidden_states = new_hidden_states
        else:
            if self.training:
                hidden_states = (
                    hidden_states + (0 * audio_set_tensor[:, 0].to(hidden_states.dtype).to(hidden_states.device)).sum()
                )

        if self.drop is not None:
            hidden_states = self.drop(hidden_states)

        return hidden_states


#################################################### TEXT ####################################################


class Phi4MultimodalImageAudioEmbedding(nn.Module):
    """Image-audio embedding."""

    def __init__(self, config: PretrainedConfig) -> None:
        super().__init__()

        self.vocab_size = config.vocab_size

        self.image_input_id = -1
        self.audio_input_id = -10000
        assert self.image_input_id != self.audio_input_id, "image_input_id and audio_input_id should be different"

        self.image_embed = Phi4MultimodalImageEmbedding(config)
        self.audio_embed = Phi4MultimodalAudioEmbedding(config)

        self.input_image_embeds = None
        self.image_sizes = None
        self.image_attention_mask = None
        self.input_audio_embeds = None
        self.audio_embed_sizes = None

    def post_init(self, audio_config):
        # post init for audio embedding
        # ref: model.model.embed_tokens_extend.post_init(audio_config) in phyagi/getters/model.py
        self.audio_embed.post_init(audio_config)

    def set_input_image_embeds(self, input_image_embeds: torch.FloatTensor) -> None:
        self.input_image_embeds = input_image_embeds

    def set_image_sizes(self, image_sizes: torch.LongTensor) -> None:
        self.image_sizes = image_sizes

    def set_img_attn_mask(self, image_attention_mask: torch.FloatTensor) -> None:
        self.image_attention_mask = image_attention_mask

    def set_input_audio_embeds(self, input_audio_embeds: torch.FloatTensor) -> None:
        self.input_audio_embeds = input_audio_embeds

    def set_audio_embed_sizes(self, audio_embed_sizes: torch.LongTensor) -> None:
        self.audio_embed_sizes = audio_embed_sizes

    def forward(
        self,
        input_ids: torch.LongTensor,
        input_embeds,
        input_image_embeds: Optional[torch.FloatTensor] = None,
        input_audio_embeds: Optional[torch.FloatTensor] = None,
        image_sizes=None,
        image_attention_mask=None,
        audio_embed_sizes=None,
        audio_attention_mask=None,
        audio_projection_mode="speech",
        wte=None,
    ) -> torch.FloatTensor:
        # override image and audio embeddings and sizes from object itself
        # this is for inference
        # ref: phyagi/eval/utils/text_generation_vision_audio_pipeline.py
        if self.input_image_embeds is not None:
            assert input_image_embeds is None
            input_image_embeds = self.input_image_embeds.clone()
            # NOTE weijian: set input_image_embeds to None after first call in for eval stage
            # during evaluation, it will call model's forward() multiple times
            # the first time input_ids contains the prompt (including <|image_{}|>) and input_embeds exists
            # from the second time, the input_ids will only contain the generated text
            # thus, the input_image_embeds is no longer needed
            self.input_image_embeds = None

        if self.image_sizes is not None:
            assert image_sizes is None
            image_sizes = self.image_sizes

        if self.input_audio_embeds is not None:
            assert input_audio_embeds is None
            input_audio_embeds = self.input_audio_embeds.clone()
            self.input_audio_embeds = None

        if self.audio_embed_sizes is not None:
            assert audio_embed_sizes is None
            audio_embed_sizes = self.audio_embed_sizes.clone()

        if self.image_attention_mask is not None:
            assert image_attention_mask is None
            image_attention_mask = self.image_attention_mask.clone()
            self.image_attention_mask = None

        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_shape[-1])

        # backward compatibility
        with torch.no_grad():
            new_input_ids = input_ids.clone()
            new_input_ids[
                (input_ids >= _COMPATIBLE_IMAGE_SPECIAL_TOKEN_ID_RANGE[0])
                & (input_ids <= _COMPATIBLE_IMAGE_SPECIAL_TOKEN_ID_RANGE[1])
            ] = _IMAGE_SPECIAL_TOKEN_ID
            new_input_ids[
                (input_ids >= _COMPATIBLE_AUDIO_SPECIAL_TOKEN_ID_RANGE[0])
                & (input_ids <= _COMPATIBLE_AUDIO_SPECIAL_TOKEN_ID_RANGE[1])
            ] = _AUDIO_SPECIAL_TOKEN_ID
            input_ids = new_input_ids

        with torch.no_grad():
            image_position_mask = input_ids == _IMAGE_SPECIAL_TOKEN_ID
            non_image_position_mask = ~image_position_mask

        assert input_embeds is None
        if self.training:
            assert input_image_embeds is not None or input_audio_embeds is not None

        if input_image_embeds is not None:
            image_hidden_states = self.image_embed(
                input_ids=input_ids,
                input_embeds=input_image_embeds,
                image_sizes=image_sizes,
                wte=wte,
                image_attention_mask=image_attention_mask,
            )
        if input_audio_embeds is not None:
            audio_hidden_states = self.audio_embed(
                input_ids=input_ids,
                input_embeds=input_audio_embeds,
                audio_embed_sizes=audio_embed_sizes,
                audio_attention_mask=audio_attention_mask,
                wte=wte,
                audio_projection_mode=audio_projection_mode,
            )

        # merge image and audio hidden states
        # NOTE weijian: for non-image-audio tokens, here we use audio hidden states
        # actually, in the debug code above, the non-image-audio tokens from image_hidden_states and audio_hidden_states should be the same
        if input_image_embeds is not None and input_audio_embeds is not None:
            dtype = image_hidden_states.dtype
            hidden_states = image_hidden_states * image_position_mask.to(dtype).unsqueeze(
                -1
            ) + audio_hidden_states * non_image_position_mask.to(dtype).unsqueeze(-1)
        elif input_image_embeds is not None:
            hidden_states = image_hidden_states
        elif input_audio_embeds is not None:
            hidden_states = audio_hidden_states
        else:
            assert wte is not None
            hidden_states = wte(input_ids)

        return hidden_states


class Phi4MultimodalRMSNorm(Phi3RMSNorm):
    pass


class Phi4MultimodalDecoderLayer(Phi3DecoderLayer):
    pass


class Phi4MultimodalModel(Phi3Model, nn.Module):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`Phi4MultimodalMMDecoderLayer`]
    Args:
        config: Phi4MultimodalMMConfig
    """

    def __init__(self, config: Phi4MultimodalConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.embed_dropout = nn.Dropout(config.embd_pdrop)

        self.embed_tokens_extend = Phi4MultimodalImageAudioEmbedding(config)

        self.layers = nn.ModuleList(
            [Phi4MultimodalDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = Phi4MultimodalRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        input_image_embeds: Optional[torch.FloatTensor] = None,
        image_sizes: Optional[torch.LongTensor] = None,
        image_attention_mask=None,
        input_audio_embeds: Optional[torch.FloatTensor] = None,
        audio_embed_sizes=None,
        audio_attention_mask=None,
        audio_projection_mode=None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens_extend(
                input_ids=input_ids,
                input_embeds=inputs_embeds,
                input_image_embeds=input_image_embeds,
                input_audio_embeds=input_audio_embeds,
                image_sizes=image_sizes,
                image_attention_mask=image_attention_mask,
                audio_embed_sizes=audio_embed_sizes,
                audio_attention_mask=audio_attention_mask,
                audio_projection_mode=audio_projection_mode,
                wte=self.embed_tokens,
            )

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = self._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
        )

        hidden_states = inputs_embeds

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    causal_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                    cache_position,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        output = BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )
        return output if return_dict else output.to_tuple()


from enum import Enum


class InputMode(Enum):
    LANGUAGE = 0
    VISION = 1
    SPEECH = 2
    VISION_SPEECH = 3


class Phi4MultimodalForCausalLM(Phi3ForCausalLM, nn.Module):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = Phi4MultimodalModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        input_image_embeds: Optional[torch.FloatTensor] = None,
        image_sizes: Optional[torch.LongTensor] = None,
        image_attention_mask=None,
        input_audio_embeds: Optional[torch.FloatTensor] = None,
        audio_embed_sizes=None,
        audio_attention_mask=None,
        input_mode=None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
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
        >>> from transformers import AutoTokenizer, Phi4MultimodalForCausalLM
        >>> model = Phi4MultimodalForCausalLM.from_pretrained("TBA")
        >>> tokenizer = AutoTokenizer.from_pretrained("TBA")
        >>> prompt = "This is an example script ."
        >>> inputs = tokenizer(prompt, return_tensors="pt")
        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        'This is an example script .\n Certainly! Below is a sample script that demonstrates a simple task, such as calculating the sum'
        ```"""
        if (
            use_cache
            and self.config.rope_scaling
            and cache_position is not None
            and cache_position[0] == self.config.original_max_position_embeddings
        ):
            logger.warning(
                f"If you are not using the generate method, you may encounter nonsensical outputs after the {self.config.original_max_position_embeddings}th token, as the KV cache needs to be recomputed."
            )

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if isinstance(input_mode, torch.Tensor):
            assert len(input_mode) == 1
            input_mode = input_mode[0].item()
        input_mode = InputMode(input_mode)

        if input_mode in [InputMode.VISION_SPEECH, InputMode.VISION]:
            audio_projection_mode = "vision"
        elif input_mode == InputMode.SPEECH:
            audio_projection_mode = "speech"
        elif input_mode == InputMode.LANGUAGE:
            audio_projection_mode = "speech"
        else:
            raise ValueError(f"Invalid input_mode: {input_mode}")

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            input_image_embeds=input_image_embeds,
            image_sizes=image_sizes,
            image_attention_mask=image_attention_mask,
            input_audio_embeds=input_audio_embeds,
            audio_embed_sizes=audio_embed_sizes,
            audio_attention_mask=audio_attention_mask,
            audio_projection_mode=audio_projection_mode,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(logits, labels, self.vocab_size)

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
        input_image_embeds=None,
        image_sizes=None,
        image_attention_mask=None,
        input_audio_embeds=None,
        audio_embed_sizes=None,
        audio_attention_mask=None,
        input_mode=None,
        cache_position=None,
        position_ids=None,
        use_cache=True,
        num_logits_to_keep=None,
        **kwargs,
    ):
        # Overwritten -- this model may need to switch between short and long rope, invalidating the cache in the
        # process

        # When the first time input length reached long and short factor switching point, enforce re-compute cache
        # It will cause downside of slower at this single token position, however, better than current failure.
        if (
            past_key_values
            and self.config.rope_scaling
            and input_ids.shape[1] >= self.config.original_max_position_embeddings + 1
        ):
            past_length = cache_position[0]
            if past_length <= self.config.original_max_position_embeddings:
                past_key_values = None

        model_inputs = super().prepare_inputs_for_generation(
            input_ids=input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            input_image_embeds=input_image_embeds,
            image_sizes=image_sizes,
            image_attention_mask=image_attention_mask,
            input_audio_embeds=input_audio_embeds,
            audio_embed_sizes=audio_embed_sizes,
            audio_attention_mask=audio_attention_mask,
            input_mode=input_mode,
            cache_position=cache_position,
            position_ids=position_ids,
            use_cache=use_cache,
            num_logits_to_keep=num_logits_to_keep,
            **kwargs,
        )
        return model_inputs


__all__ = ["Phi4MultimodalPreTrainedModel", "Phi4MultimodalModel", "Phi4MultimodalForCausalLM"]  # noqa
