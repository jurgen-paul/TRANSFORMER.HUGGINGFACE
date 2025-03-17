# coding=utf-8
# Copyright 2025 Baidu Inc and The HuggingFace Inc. team.
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
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ...configuration_utils import PretrainedConfig
from ...utils import is_torchdynamo_compiling, logging
from ...utils.backbone_utils import (
    verify_backbone_config_arguments,
)
from ..auto import CONFIG_MAPPING
from ..rt_detr.modeling_rt_detr import (
    MultiScaleDeformableAttention,
    MultiScaleDeformableAttentionFunction,
    RTDetrForObjectDetection,
    RTDetrModel,
    RTDetrMultiscaleDeformableAttention,
    RTDetrPreTrainedModel,
)


logger = logging.get_logger(__name__)

_CHECKPOINT_FOR_DOC = "PekingU/rtdetr_v2_r50vd"
_DETECTION_OUTPUT_FOR_DOC = """
    Detected 'cat' (0.96) at [13.71, 54.12, 317.53, 472.65]
    Detected 'cat' (0.95) at [343.73, 23.68, 640.28, 373.05]
    Detected 'sofa' (0.94) at [0.2, 1.32, 640.17, 474.38]
    Detected 'remote' (0.93) at [40.6, 73.21, 175.74, 118.33]
    Detected 'remote' (0.89) at [333.51, 76.79, 370.17, 188.13]
"""


class RTDetrV2Config(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`RTDetrV2Model`]. It is used to instantiate a
    RT-DETR model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the RT-DETR architecture.

    e.g. [PekingU/rtdetr_r18vd](https://huggingface.co/PekingU/rtdetr_r18vd)

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        initializer_range (`float`, *optional*, defaults to 0.01):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        initializer_bias_prior_prob (`float`, *optional*):
            The prior probability used by the bias initializer to initialize biases for `enc_score_head` and `class_embed`.
            If `None`, `prior_prob` computed as `prior_prob = 1 / (num_labels + 1)` while initializing model weights.
        layer_norm_eps (`float`, *optional*, defaults to 1e-05):
            The epsilon used by the layer normalization layers.
        batch_norm_eps (`float`, *optional*, defaults to 1e-05):
            The epsilon used by the batch normalization layers.
        backbone_config (`Dict`, *optional*, defaults to `RTDetrV2ResNetConfig()`):
            The configuration of the backbone model.
        backbone (`str`, *optional*):
            Name of backbone to use when `backbone_config` is `None`. If `use_pretrained_backbone` is `True`, this
            will load the corresponding pretrained weights from the timm or transformers library. If `use_pretrained_backbone`
            is `False`, this loads the backbone's config and uses that to initialize the backbone with random weights.
        use_pretrained_backbone (`bool`, *optional*, defaults to `False`):
            Whether to use pretrained weights for the backbone.
        use_timm_backbone (`bool`, *optional*, defaults to `False`):
            Whether to load `backbone` from the timm library. If `False`, the backbone is loaded from the transformers
            library.
        freeze_backbone_batch_norms (`bool`, *optional*, defaults to `True`):
            Whether to freeze the batch normalization layers in the backbone.
        backbone_kwargs (`dict`, *optional*):
            Keyword arguments to be passed to AutoBackbone when loading from a checkpoint
            e.g. `{'out_indices': (0, 1, 2, 3)}`. Cannot be specified if `backbone_config` is set.
        encoder_hidden_dim (`int`, *optional*, defaults to 256):
            Dimension of the layers in hybrid encoder.
        encoder_in_channels (`list`, *optional*, defaults to `[512, 1024, 2048]`):
            Multi level features input for encoder.
        feat_strides (`List[int]`, *optional*, defaults to `[8, 16, 32]`):
            Strides used in each feature map.
        encoder_layers (`int`, *optional*, defaults to 1):
            Total of layers to be used by the encoder.
        encoder_ffn_dim (`int`, *optional*, defaults to 1024):
            Dimension of the "intermediate" (often named feed-forward) layer in decoder.
        encoder_attention_heads (`int`, *optional*, defaults to 8):
            Number of attention heads for each attention layer in the Transformer encoder.
        dropout (`float`, *optional*, defaults to 0.0):
            The ratio for all dropout layers.
        activation_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for activations inside the fully connected layer.
        encode_proj_layers (`List[int]`, *optional*, defaults to `[2]`):
            Indexes of the projected layers to be used in the encoder.
        positional_encoding_temperature (`int`, *optional*, defaults to 10000):
            The temperature parameter used to create the positional encodings.
        encoder_activation_function (`str`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"silu"` and `"gelu_new"` are supported.
        activation_function (`str`, *optional*, defaults to `"silu"`):
            The non-linear activation function (function or string) in the general layer. If string, `"gelu"`,
            `"relu"`, `"silu"` and `"gelu_new"` are supported.
        eval_size (`Tuple[int, int]`, *optional*):
            Height and width used to compute the effective height and width of the position embeddings after taking
            into account the stride.
        normalize_before (`bool`, *optional*, defaults to `False`):
            Determine whether to apply layer normalization in the transformer encoder layer before self-attention and
            feed-forward modules.
        hidden_expansion (`float`, *optional*, defaults to 1.0):
            Expansion ratio to enlarge the dimension size of RepVGGBlock and CSPRepLayer.
        d_model (`int`, *optional*, defaults to 256):
            Dimension of the layers exclude hybrid encoder.
        num_queries (`int`, *optional*, defaults to 300):
            Number of object queries.
        decoder_in_channels (`list`, *optional*, defaults to `[256, 256, 256]`):
            Multi level features dimension for decoder
        decoder_ffn_dim (`int`, *optional*, defaults to 1024):
            Dimension of the "intermediate" (often named feed-forward) layer in decoder.
        num_feature_levels (`int`, *optional*, defaults to 3):
            The number of input feature levels.
        decoder_n_points (`int`, *optional*, defaults to 4):
            The number of sampled keys in each feature level for each attention head in the decoder.
        decoder_layers (`int`, *optional*, defaults to 6):
            Number of decoder layers.
        decoder_attention_heads (`int`, *optional*, defaults to 8):
            Number of attention heads for each attention layer in the Transformer decoder.
        decoder_activation_function (`str`, *optional*, defaults to `"relu"`):
            The non-linear activation function (function or string) in the decoder. If string, `"gelu"`,
            `"relu"`, `"silu"` and `"gelu_new"` are supported.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        num_denoising (`int`, *optional*, defaults to 100):
            The total number of denoising tasks or queries to be used for contrastive denoising.
        label_noise_ratio (`float`, *optional*, defaults to 0.5):
            The fraction of denoising labels to which random noise should be added.
        box_noise_scale (`float`, *optional*, defaults to 1.0):
            Scale or magnitude of noise to be added to the bounding boxes.
        learn_initial_query (`bool`, *optional*, defaults to `False`):
            Indicates whether the initial query embeddings for the decoder should be learned during training
        anchor_image_size (`Tuple[int, int]`, *optional*):
            Height and width of the input image used during evaluation to generate the bounding box anchors. If None, automatic generate anchor is applied.
        with_box_refine (`bool`, *optional*, defaults to `True`):
            Whether to apply iterative bounding box refinement, where each decoder layer refines the bounding boxes
            based on the predictions from the previous layer.
        is_encoder_decoder (`bool`, *optional*, defaults to `True`):
            Whether the architecture has an encoder decoder structure.
        matcher_alpha (`float`, *optional*, defaults to 0.25):
            Parameter alpha used by the Hungarian Matcher.
        matcher_gamma (`float`, *optional*, defaults to 2.0):
            Parameter gamma used by the Hungarian Matcher.
        matcher_class_cost (`float`, *optional*, defaults to 2.0):
            The relative weight of the class loss used by the Hungarian Matcher.
        matcher_bbox_cost (`float`, *optional*, defaults to 5.0):
            The relative weight of the bounding box loss used by the Hungarian Matcher.
        matcher_giou_cost (`float`, *optional*, defaults to 2.0):
            The relative weight of the giou loss of used by the Hungarian Matcher.
        use_focal_loss (`bool`, *optional*, defaults to `True`):
            Parameter informing if focal loss should be used.
        auxiliary_loss (`bool`, *optional*, defaults to `True`):
            Whether auxiliary decoding losses (loss at each decoder layer) are to be used.
        focal_loss_alpha (`float`, *optional*, defaults to 0.75):
            Parameter alpha used to compute the focal loss.
        focal_loss_gamma (`float`, *optional*, defaults to 2.0):
            Parameter gamma used to compute the focal loss.
        weight_loss_vfl (`float`, *optional*, defaults to 1.0):
            Relative weight of the varifocal loss in the object detection loss.
        weight_loss_bbox (`float`, *optional*, defaults to 5.0):
            Relative weight of the L1 bounding box loss in the object detection loss.
        weight_loss_giou (`float`, *optional*, defaults to 2.0):
            Relative weight of the generalized IoU loss in the object detection loss.
        eos_coefficient (`float`, *optional*, defaults to 0.0001):
            Relative classification weight of the 'no-object' class in the object detection loss.
        decoder_n_levels (`int`, *optional*, defaults to 3):
            The number of feature levels used by the decoder.
        decoder_offset_scale (`float`, *optional*, defaults to 0.5):
            Scaling factor applied to the attention offsets in the decoder.
        decoder_method (`str`, *optional*, defaults to `"default"`):
            The method to use for the decoder: `"default"` or `"discrete"`.

    Examples:

    ```python
    >>> from transformers import RTDetrV2Config, RTDetrV2Model

    >>> # Initializing a RT-DETR configuration
    >>> configuration = RTDetrV2Config()

    >>> # Initializing a model (with random weights) from the configuration
    >>> model = RTDetrV2Model(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```
    """

    model_type = "rt_detr_v2"
    layer_types = ["basic", "bottleneck"]
    attribute_map = {
        "hidden_size": "d_model",
        "num_attention_heads": "encoder_attention_heads",
    }

    def __init__(
        self,
        initializer_range=0.01,
        initializer_bias_prior_prob=None,
        layer_norm_eps=1e-5,
        batch_norm_eps=1e-5,
        # backbone
        backbone_config=None,
        backbone=None,
        use_pretrained_backbone=False,
        use_timm_backbone=False,
        freeze_backbone_batch_norms=True,
        backbone_kwargs=None,
        # encoder HybridEncoder
        encoder_hidden_dim=256,
        encoder_in_channels=[512, 1024, 2048],
        feat_strides=[8, 16, 32],
        encoder_layers=1,
        encoder_ffn_dim=1024,
        encoder_attention_heads=8,
        dropout=0.0,
        activation_dropout=0.0,
        encode_proj_layers=[2],
        positional_encoding_temperature=10000,
        encoder_activation_function="gelu",
        activation_function="silu",
        eval_size=None,
        normalize_before=False,
        hidden_expansion=1.0,
        # decoder RTDetrV2Transformer
        d_model=256,
        num_queries=300,
        decoder_in_channels=[256, 256, 256],
        decoder_ffn_dim=1024,
        num_feature_levels=3,
        decoder_n_points=4,
        decoder_layers=6,
        decoder_attention_heads=8,
        decoder_activation_function="relu",
        attention_dropout=0.0,
        num_denoising=100,
        label_noise_ratio=0.5,
        box_noise_scale=1.0,
        learn_initial_query=False,
        anchor_image_size=None,
        with_box_refine=True,
        is_encoder_decoder=True,
        # Loss
        matcher_alpha=0.25,
        matcher_gamma=2.0,
        matcher_class_cost=2.0,
        matcher_bbox_cost=5.0,
        matcher_giou_cost=2.0,
        use_focal_loss=True,
        auxiliary_loss=True,
        focal_loss_alpha=0.75,
        focal_loss_gamma=2.0,
        weight_loss_vfl=1.0,
        weight_loss_bbox=5.0,
        weight_loss_giou=2.0,
        eos_coefficient=1e-4,
        decoder_n_levels=3,  # default value
        decoder_offset_scale=0.5,  # default value
        decoder_method="default",
        **kwargs,
    ):
        super().__init__(is_encoder_decoder=is_encoder_decoder, **kwargs)
        self.initializer_range = initializer_range
        self.initializer_bias_prior_prob = initializer_bias_prior_prob
        self.layer_norm_eps = layer_norm_eps
        self.batch_norm_eps = batch_norm_eps
        # backbone
        if backbone_config is None and backbone is None:
            logger.info(
                "`backbone_config` and `backbone` are `None`. Initializing the config with the default `RTDetrV2-ResNet` backbone."
            )
            backbone_model_type = "rt_detr_resnet"
            config_class = CONFIG_MAPPING[backbone_model_type]
            # this will map it to RTDetrResNetConfig
            # note: we can instead create RTDetrV2ResNetConfig but it will be exactly the same as V1
            # and we would need to create RTDetrV2ResNetModel
            backbone_config = config_class(
                num_channels=3,
                embedding_size=64,
                hidden_sizes=[256, 512, 1024, 2048],
                depths=[3, 4, 6, 3],
                layer_type="bottleneck",
                hidden_act="relu",
                downsample_in_first_stage=False,
                downsample_in_bottleneck=False,
                out_features=None,
                out_indices=[2, 3, 4],
            )
        elif isinstance(backbone_config, dict):
            backbone_model_type = backbone_config.pop("model_type")
            config_class = CONFIG_MAPPING[backbone_model_type]
            backbone_config = config_class.from_dict(backbone_config)

        verify_backbone_config_arguments(
            use_timm_backbone=use_timm_backbone,
            use_pretrained_backbone=use_pretrained_backbone,
            backbone=backbone,
            backbone_config=backbone_config,
            backbone_kwargs=backbone_kwargs,
        )

        self.backbone_config = backbone_config
        self.backbone = backbone
        self.use_pretrained_backbone = use_pretrained_backbone
        self.use_timm_backbone = use_timm_backbone
        self.freeze_backbone_batch_norms = freeze_backbone_batch_norms
        self.backbone_kwargs = backbone_kwargs
        # encoder
        self.encoder_hidden_dim = encoder_hidden_dim
        self.encoder_in_channels = encoder_in_channels
        self.feat_strides = feat_strides
        self.encoder_ffn_dim = encoder_ffn_dim
        self.dropout = dropout
        self.activation_dropout = activation_dropout
        self.encode_proj_layers = encode_proj_layers
        self.encoder_layers = encoder_layers
        self.positional_encoding_temperature = positional_encoding_temperature
        self.eval_size = eval_size
        self.normalize_before = normalize_before
        self.encoder_activation_function = encoder_activation_function
        self.activation_function = activation_function
        self.hidden_expansion = hidden_expansion
        self.num_queries = num_queries
        self.decoder_ffn_dim = decoder_ffn_dim
        self.decoder_in_channels = decoder_in_channels
        self.num_feature_levels = num_feature_levels
        self.decoder_n_points = decoder_n_points
        self.decoder_layers = decoder_layers
        self.decoder_attention_heads = decoder_attention_heads
        self.decoder_activation_function = decoder_activation_function
        self.attention_dropout = attention_dropout
        self.num_denoising = num_denoising
        self.label_noise_ratio = label_noise_ratio
        self.box_noise_scale = box_noise_scale
        self.learn_initial_query = learn_initial_query
        self.anchor_image_size = anchor_image_size
        self.auxiliary_loss = auxiliary_loss
        self.with_box_refine = with_box_refine
        # Loss
        self.matcher_alpha = matcher_alpha
        self.matcher_gamma = matcher_gamma
        self.matcher_class_cost = matcher_class_cost
        self.matcher_bbox_cost = matcher_bbox_cost
        self.matcher_giou_cost = matcher_giou_cost
        self.use_focal_loss = use_focal_loss
        self.focal_loss_alpha = focal_loss_alpha
        self.focal_loss_gamma = focal_loss_gamma
        self.weight_loss_vfl = weight_loss_vfl
        self.weight_loss_bbox = weight_loss_bbox
        self.weight_loss_giou = weight_loss_giou
        self.eos_coefficient = eos_coefficient

        if not hasattr(self, "d_model"):
            self.d_model = d_model

        if not hasattr(self, "encoder_attention_heads"):
            self.encoder_attention_heads = encoder_attention_heads
        # add the new attributes with the given values or defaults
        self.decoder_n_levels = decoder_n_levels
        self.decoder_offset_scale = decoder_offset_scale
        self.decoder_method = decoder_method

    @classmethod
    def from_backbone_configs(cls, backbone_config: PretrainedConfig, **kwargs):
        """Instantiate a [`RTDetrV2Config`] (or a derived class) from a pre-trained backbone model configuration and DETR model
        configuration.

            Args:
                backbone_config ([`PretrainedConfig`]):
                    The backbone configuration.

            Returns:
                [`RTDetrV2Config`]: An instance of a configuration object
        """
        return cls(
            backbone_config=backbone_config,
            **kwargs,
        )


def multi_scale_deformable_attention_v2(
    value: Tensor,
    value_spatial_shapes: Union[Tensor, List[Tuple]],
    sampling_locations: Tensor,
    attention_weights: Tensor,
    method="default",
) -> Tensor:
    """
    In addition to the `multi_scale_deformable_attention` (v1) function,
    this function supports the `discrete` method of sampling, for `default` sampling method
    the behavior is the same as the `multi_scale_deformable_attention` (v1) function.
    """
    batch_size, _, num_heads, hidden_dim = value.shape
    batched_num_queries, num_heads, num_levels, num_points, _ = sampling_locations.shape
    num_queries = batched_num_queries // batch_size

    if method == "default":
        sampling_grids = 2 * sampling_locations - 1
    elif method == "discrete":
        sampling_grids = sampling_locations

    value_levels = value.split([height * width for height, width in value_spatial_shapes], dim=1)

    sampled_values = []
    for idx, (height, width) in enumerate(value_spatial_shapes):
        # batch_size, height * width, num_heads, hidden_dim
        # -> batch_size, num_heads * hidden_dim, height * width
        # -> batch_size * num_heads, hidden_dim, height, width
        value_i = value_levels[idx]
        value_i = value_i.flatten(2).transpose(1, 2)
        value_i = value_i.reshape(batch_size * num_heads, hidden_dim, height, width)

        # batch_size * num_queries, num_heads, num_points, 2
        # -> batch_size * num_heads, num_queries, num_points, 2
        sampling_grid_i = sampling_grids[:, :, idx]
        sampling_grid_i = sampling_grid_i.view(batch_size, num_queries, num_heads, num_points, 2)
        sampling_grid_i = sampling_grid_i.transpose(1, 2).flatten(0, 1)

        # batch_size * num_heads, hidden_dim, num_queries, num_points
        if method == "default":
            sampled_value_i = nn.functional.grid_sample(
                value_i, sampling_grid_i, mode="bilinear", padding_mode="zeros", align_corners=False
            )
        elif method == "discrete":
            sampling_coord = sampling_grid_i * torch.tensor([[width, height]], device=value_i.device) + 0.5
            sampling_coord = sampling_coord.to(torch.int64)

            # Separate clamping for x and y coordinates
            sampling_coord_x = sampling_coord[..., 0].clamp(0, width - 1)
            sampling_coord_y = sampling_coord[..., 1].clamp(0, height - 1)

            sampling_coord_x = sampling_coord_x.reshape(batch_size * num_heads, num_queries * num_points)
            sampling_coord_y = sampling_coord_y.reshape(batch_size * num_heads, num_queries * num_points)

            sampling_idx = torch.arange(batch_size * num_heads, device=value_i.device)
            sampling_idx = sampling_idx.unsqueeze(-1).repeat(1, num_queries * num_points)

            sampled_value_i = value_i[sampling_idx, :, sampling_coord_y, sampling_coord_x]
            sampled_value_i = sampled_value_i.transpose(1, 2)
            sampled_value_i = sampled_value_i.reshape(batch_size * num_heads, hidden_dim, num_queries, num_points)

        sampled_values.append(sampled_value_i)
    sampled_values = torch.stack(sampled_values, dim=-2)

    # (batch_size, num_queries, num_heads, ...) -> (batch_size, num_heads, num_queries, ...)
    attention_weights = attention_weights.transpose(1, 2)
    attention_weights = attention_weights.reshape(batch_size * num_heads, 1, num_queries, num_levels * num_points)

    output = attention_weights * sampled_values.flatten(-2)
    output = output.sum(-1)

    output = output.view(batch_size, num_heads * hidden_dim, num_queries)
    output = output.transpose(1, 2).contiguous()

    return output


class RTDetrV2MultiscaleDeformableAttention(RTDetrMultiscaleDeformableAttention):
    """
    RTDetrV2 version of multiscale deformable attention, extending the base implementation
    with improved offset handling and initialization.
    """

    def __init__(self, config: RTDetrV2Config, num_heads: int, num_points: int):
        super().__init__(config, num_heads, num_points)
        self.offset_scale = config.decoder_offset_scale
        self.method = config.decoder_method

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        reference_points: torch.Tensor,
        spatial_shapes: torch.Tensor,
        spatial_shapes_list: List[Tuple[int, int]],
        attention_mask: Optional[torch.Tensor] = None,
        position_embeddings: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ):
        # Process inputs up to sampling locations calculation using parent class logic
        if position_embeddings is not None:
            hidden_states = hidden_states + position_embeddings

        batch_size, num_queries, _ = hidden_states.shape
        batch_size, sequence_length, _ = encoder_hidden_states.shape

        total_elements = sum(height * width for height, width in spatial_shapes_list)
        if total_elements != sequence_length:
            raise ValueError(
                "Make sure to align the spatial shapes with the sequence length of the encoder hidden states"
            )

        value = self.value_proj(encoder_hidden_states)
        if attention_mask is not None:
            # we invert the attention_mask
            value = value.masked_fill(~attention_mask[..., None], 0.0)

        value = value.view(batch_size, sequence_length, self.num_heads, self.head_dim)

        attention_weights = self.attention_weights(hidden_states).view(
            batch_size, num_queries, self.num_heads, self.num_levels * self.num_points
        )
        attention_weights = F.softmax(attention_weights, -1)
        attention_weights = attention_weights.view(
            batch_size, num_queries, self.num_heads, self.num_levels, self.num_points
        )

        # Sampling offsets shape
        sampling_offsets = self.sampling_offsets(hidden_states)
        sampling_offsets = sampling_offsets.view(
            batch_size * num_queries, self.num_heads, self.num_levels, self.num_points, 2
        )

        # Sampling locations calculation
        batch_size, num_reference_points, _, num_coordinates = reference_points.shape
        reference_points = reference_points.view(batch_size * num_reference_points, 1, -1, 1, num_coordinates)

        if num_coordinates == 2:
            height, width = spatial_shapes[..., 0], spatial_shapes[..., 1]
            offset_normalizer = torch.stack([width, height], -1)
            normalized_sampling_offsets = sampling_offsets / offset_normalizer[None, None, :, None, :]
            sampling_locations = reference_points + normalized_sampling_offsets

        elif num_coordinates == 4:
            reference_points_xy = reference_points[..., :2]
            offset = sampling_offsets / self.num_points * reference_points[..., 2:] * self.offset_scale
            sampling_locations = reference_points_xy + offset

        else:
            raise ValueError(f"Last dim of reference_points must be 2 or 4, but got {reference_points.shape[-1]}")

        if (
            self.disable_custom_kernels  # manually disabled in config
            or MultiScaleDeformableAttention is None  # error while loading the kernel
            or is_torchdynamo_compiling()  # torch.compile / torch.export mode
        ):
            # PyTorch implementation
            output = multi_scale_deformable_attention_v2(
                value, spatial_shapes_list, sampling_locations, attention_weights, method=self.method
            )
        else:
            try:
                # Calling custom kernel
                # Note: for custom kernel we pass sampling locations as 6D tensor,
                #       but for torch implementation we keep it as 5D tensor (for CoreML compat)
                kernel_sampling_locations = sampling_locations.view(
                    batch_size, num_queries, self.num_heads, self.num_levels, self.num_points, 2
                )
                level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
                output = MultiScaleDeformableAttentionFunction.apply(
                    value,
                    spatial_shapes,
                    level_start_index,
                    kernel_sampling_locations,
                    attention_weights,
                    self.im2col_step,
                )
            except Exception:
                # PyTorch implementation
                output = multi_scale_deformable_attention_v2(
                    value, spatial_shapes_list, sampling_locations, attention_weights, method=self.method
                )

        output = self.output_proj(output)

        if not output_attentions:
            attention_weights = None

        return output, attention_weights


class RTDetrV2PreTrainedModel(RTDetrPreTrainedModel):
    pass


class RTDetrV2Model(RTDetrModel):
    pass


class RTDetrV2ForObjectDetection(RTDetrForObjectDetection, RTDetrV2PreTrainedModel):
    pass


__all__ = [
    "RTDetrV2Config",
    "RTDetrV2Model",
    "RTDetrV2PreTrainedModel",
    "RTDetrV2ForObjectDetection",
]
