# coding=utf-8
# Copyright 2021 Google AI, Ross Wightman, The HuggingFace Inc. team. All rights reserved.
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
""" PyTorch Cvt model."""


import collections.abc
import math
import warnings
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss

from ...activations import ACT2FN
from ...file_utils import add_code_sample_docstrings, add_start_docstrings, add_start_docstrings_to_model_forward
from ...modeling_outputs import ModelOutput, SequenceClassifierOutput
from ...modeling_utils import PreTrainedModel, find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import logging
from .configuration_cvt import CvtConfig


logger = logging.get_logger(__name__)

# General docstring
_CONFIG_FOR_DOC = "CvtConfig"
_FEAT_EXTRACTOR_FOR_DOC = "CvtFeatureExtractor"

# Base docstring
_CHECKPOINT_FOR_DOC = "microsoft/cvt-13"
_EXPECTED_OUTPUT_SHAPE = [1, 384, 14, 14]

# Image classification docstring
_IMAGE_CLASS_CHECKPOINT = "microsoft/cvt-13"
_IMAGE_CLASS_EXPECTED_OUTPUT = "'tabby cat'"


CVT_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "anugunj/cvt-13",
    # See all Cvt models at https://huggingface.co/models?filter=cvt
]


@dataclass
class BaseModelOutputWithCLSToken(ModelOutput):
    """
    Base class for model's outputs, with potential hidden states and attentions.

    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    last_hidden_state: torch.FloatTensor = None
    cls_token_value: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


# Inspired by
# https://github.com/rwightman/pytorch-image-models/blob/b9bd960a032c75ca6b808ddeed76bee5f3ed4972/timm/models/layers/helpers.py
# From PyTorch internals
def to_2tuple(x):
    if isinstance(x, collections.abc.Iterable):
        return x
    return (x, x)


def drop_path(x, drop_prob: float = 0.0, training: bool = False, scale_by_keep: bool = True):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however, the original name is
    misleading as 'Drop Connect' is a different form of dropout in a separate paper... See discussion:
    https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for changing the layer and
    argument names to 'drop path' rather than mix DropConnect as a layer name and use 'survival rate' as the argument.
    """
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn(
            "mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
            "The distribution of values may be incorrect.",
            stacklevel=2,
        )

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.0))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0):
    ## type: (Tensor, float, float, float, float) -> Tensor
    r"""Fills the input Tensor with values drawn from a truncated
    Args:
    normal distribution. The values are effectively drawn from the normal distribution :math:`\mathcal{N}(\text{mean},
    \text{std}^2)` with values outside :math:`[a, b]` redrawn until they are within the bounds. The method used for
    generating the random values works best when :math:`a \leq \text{mean} \leq b`.
        tensor: an n-dimensional `torch.Tensor` mean: the mean of the normal distribution std: the standard deviation
        of the normal distribution a: the minimum cutoff value b: the maximum cutoff value
    Examples:
        >>> w = torch.empty(3, 5) >>> nn.init.trunc_normal_(w)
    """
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob=None, scale_by_keep=True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)


class CvtEmbeddings(nn.Module):
    """
    Construct the CLS token, position and patch embeddings.

    """

    def __init__(self, patch_size, num_channels, embed_dim, stride, padding, dropout_rate):
        super().__init__()
        self.conv_embeddings = ConvEmbeddings(
            patch_size=patch_size, num_channels=num_channels, embed_dim=embed_dim, stride=stride, padding=padding
        )
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, pixel_values):
        embeddings = self.conv_embeddings(pixel_values)
        embeddings = self.dropout(embeddings)
        return embeddings


class ConvEmbeddings(nn.Module):
    """
    Image to Conv Embedding.

    """
    def __init__(self, patch_size, num_channels, embed_dim, stride, padding):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size
        self.projection = nn.Conv2d(num_channels, embed_dim, kernel_size=patch_size, stride=stride, padding=padding)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, pixel_values):
        pixel_values = self.projection(pixel_values)
        batch_size, num_channels, height, width = pixel_values.shape
        hidden_size = height * width
        # rearrange "b c h w -> b (h w) c"
        pixel_values = pixel_values.view(batch_size, num_channels, hidden_size).permute(0, 2, 1)
        if self.norm:
            pixel_values = self.norm(pixel_values)
        # rearrange "b (h w) c" -> b c h w"
        pixel_values = pixel_values.permute(0, 2, 1).view(batch_size, num_channels, height, width)
        return pixel_values


class CvtSelfAttentionConvProjection(nn.Sequential):
    def __init__(self, embed_dim, kernel_size, padding, stride):
        super().__init__()
        self.conv = nn.Conv2d(
            embed_dim,
            embed_dim,
            kernel_size=kernel_size,
            padding=padding,
            stride=stride,
            bias=False,
            groups=embed_dim,
        )
        self.norm = nn.BatchNorm2d(embed_dim)


class CvtSelfAttentionLinearProjection(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, hidden_states):
        batch_size, num_channels, height, width = hidden_states.shape
        hidden_size = height * width
        # rearrange " b c h w -> b (h w) c"
        hidden_states = hidden_states.view(batch_size, num_channels, hidden_size).permute(0, 2, 1)
        return hidden_states


class CvtSelfAttentionProjection(nn.Sequential):
    def __init__(self, embed_dim, kernel_size, padding, stride, projection_method="dw_bn"):
        super().__init__()
        if projection_method == "dw_bn":
            self.conv_projection = CvtSelfAttentionConvProjection(embed_dim, kernel_size, padding, stride)
        self.linear_projection = CvtSelfAttentionLinearProjection()


class CvtSelfAttention(nn.Module):
    def __init__(
        self,
        num_heads,
        embed_dim,
        kernel_size,
        padding_q,
        padding_kv,
        stride_q,
        stride_kv,
        qkv_projection_method,
        qkv_bias,
        attention_drop_rate,
        with_cls_token=True,
        **kwargs
    ):
        super().__init__()
        self.scale = embed_dim**-0.5
        self.with_cls_token = with_cls_token
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        self.conv_projection_query = CvtSelfAttentionProjection(
            embed_dim,
            kernel_size,
            padding_q,
            stride_q,
            projection_method="linear" if qkv_projection_method == "avg" else qkv_projection_method,
        )
        self.conv_projection_key = CvtSelfAttentionProjection(
            embed_dim, kernel_size, padding_kv, stride_kv, projection_method=qkv_projection_method
        )
        self.conv_projection_value = CvtSelfAttentionProjection(
            embed_dim, kernel_size, padding_kv, stride_kv, projection_method=qkv_projection_method
        )

        self.projection_query = nn.Linear(embed_dim, embed_dim, bias=qkv_bias)
        self.projection_key = nn.Linear(embed_dim, embed_dim, bias=qkv_bias)
        self.projection_value = nn.Linear(embed_dim, embed_dim, bias=qkv_bias)

        self.dropout = nn.Dropout(attention_drop_rate)

    def rearrange_for_multi_head_attention(self, inputs):
        (
            batch_size,
            hidden_size,
            _,
        ) = inputs.shape
        head_dim = self.embed_dim // self.num_heads
        # rearrange 'b t (h d) -> b h t d'
        return inputs.view(batch_size, hidden_size, self.num_heads, head_dim).permute(0, 2, 1, 3)

    def forward(self, hidden_states, height, width, head_mask=None, output_attentions=False):
        if self.with_cls_token:
            cls_token, hidden_states = torch.split(hidden_states, [1, height * width], 1)
        batch_size, hidden_size, num_channels = hidden_states.shape
        # rearrange "b (h w) c -> b c h w"
        hidden_states = hidden_states.permute(0, 2, 1).view(batch_size, num_channels, height, width)

        key = self.conv_projection_key(hidden_states)
        query = self.conv_projection_query(hidden_states)
        value = self.conv_projection_value(hidden_states)

        if self.with_cls_token:
            query = torch.cat((cls_token, query), dim=1)
            key = torch.cat((cls_token, key), dim=1)
            value = torch.cat((cls_token, value), dim=1)

        head_dim = self.embed_dim // self.num_heads

        query = self.rearrange_for_multi_head_attention(self.projection_query(query))
        key = self.rearrange_for_multi_head_attention(self.projection_key(key))
        value = self.rearrange_for_multi_head_attention(self.projection_value(value))

        attention_score = torch.einsum("bhlk,bhtk->bhlt", [query, key]) * self.scale
        attention_probs = torch.nn.functional.softmax(attention_score, dim=-1)
        attention_probs = self.dropout(attention_probs)

        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context = torch.einsum("bhlt,bhtv->bhlv", [attention_probs, value])
        # rearrange"b h t d -> b t (h d)"
        _, _, hidden_size, _ = context.shape
        context = context.permute(0, 2, 1, 3).contiguous().view(batch_size, hidden_size, self.num_heads * head_dim)

        outputs = (context, attention_probs) if output_attentions else (context,)
        return outputs


class CvtSelfOutput(nn.Module):
    """
    The residual connection is defined in CvtLayer instead of here (as is the case with other models), due to the
    layernorm applied before each block.
    """

    def __init__(self, embed_dim, drop_rate):
        super().__init__()
        self.dense = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(drop_rate)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states


class CvtAttention(nn.Module):
    def __init__(
        self,
        num_heads,
        embed_dim,
        kernel_size,
        padding_q,
        padding_kv,
        stride_q,
        stride_kv,
        qkv_projection_method,
        qkv_bias,
        attention_drop_rate,
        drop_rate,
        with_cls_token=True,
    ):
        super().__init__()
        self.attention = CvtSelfAttention(
            num_heads,
            embed_dim,
            kernel_size,
            padding_q,
            padding_kv,
            stride_q,
            stride_kv,
            qkv_projection_method,
            qkv_bias,
            attention_drop_rate,
            with_cls_token,
        )
        self.output = CvtSelfOutput(embed_dim, drop_rate)
        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads, self.attention.num_attention_heads, self.attention.attention_head_size, self.pruned_heads
        )

        # Prune linear layers
        self.attention.query = prune_linear_layer(self.attention.query, index)
        self.attention.key = prune_linear_layer(self.attention.key, index)
        self.attention.value = prune_linear_layer(self.attention.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # Update hyper params and store pruned heads
        self.attention.num_attention_heads = self.attention.num_attention_heads - len(heads)
        self.attention.all_head_size = self.attention.attention_head_size * self.attention.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(self, hidden_states, height, width, head_mask=None, output_attentions=False):
        self_outputs = self.attention(hidden_states, height, width, head_mask, output_attentions)

        attention_output = self.output(self_outputs[0], hidden_states)
        # add attentions if we output them
        outputs = (attention_output,) + self_outputs[1:]
        return outputs


class CvtIntermediate(nn.Module):
    def __init__(self, embed_dim, mlp_ratio):
        super().__init__()
        self.hidden_dim = int(embed_dim * mlp_ratio)
        self.dense = nn.Linear(embed_dim, self.hidden_dim)
        self.intermediate_act_fn = nn.GELU()

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class CvtOutput(nn.Module):
    def __init__(self, embed_dim, mlp_ratio, drop_rate):
        super().__init__()
        self.hidden_dim = int(embed_dim * mlp_ratio)
        self.dense = nn.Linear(self.hidden_dim, embed_dim)
        self.dropout = nn.Dropout(drop_rate)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = hidden_states + input_tensor
        return hidden_states


class CvtLayer(nn.Module):
    """This corresponds to the Block class in the timm implementation."""

    def __init__(
        self,
        num_heads,
        embed_dim,
        kernel_size,
        padding_q,
        padding_kv,
        stride_q,
        stride_kv,
        qkv_projection_method,
        qkv_bias,
        attention_drop_rate,
        drop_rate,
        mlp_ratio,
        drop_path_rate,
        with_cls_token=True,
    ):
        super().__init__()
        self.attention = CvtAttention(
            num_heads,
            embed_dim,
            kernel_size,
            padding_q,
            padding_kv,
            stride_q,
            stride_kv,
            qkv_projection_method,
            qkv_bias,
            attention_drop_rate,
            drop_rate,
            with_cls_token,
        )

        self.intermediate = CvtIntermediate(embed_dim, mlp_ratio)
        self.output = CvtOutput(embed_dim, mlp_ratio, drop_rate)
        self.drop_path = DropPath(drop_prob=drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()
        self.layernorm_before = nn.LayerNorm(embed_dim)
        self.layernorm_after = nn.LayerNorm(embed_dim)

    def forward(self, hidden_states, height, width, head_mask=None, output_attentions=False):
        self_attention_outputs = self.attention(
            self.layernorm_before(hidden_states),  # in Cvt, layernorm is applied before self-attention
            height,
            width,
            head_mask,
            output_attentions=output_attentions,
        )
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights
        attention_output = self.drop_path(attention_output)

        # first residual connection
        hidden_states = attention_output + hidden_states

        # in Cvt, layernorm is also applied after self-attention
        layer_output = self.layernorm_after(hidden_states)
        layer_output = self.intermediate(layer_output)

        # second residual connection is done here
        layer_output = self.output(layer_output, hidden_states)
        layer_output = self.drop_path(layer_output)
        outputs = (layer_output,) + outputs
        return outputs


class CvtEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        embeddings = []
        for i in range(config.num_stages):
            embeddings.append(
                CvtEmbeddings(
                    patch_size=config.patch_sizes[i],
                    stride=config.patch_stride[i],
                    num_channels=config.num_channels if i == 0 else config.embed_dim[i - 1],
                    embed_dim=config.embed_dim[i],
                    padding=config.patch_padding[i],
                    dropout_rate=config.drop_rate[i],
                )
            )

        self.patch_embeddings = nn.ModuleList(embeddings)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.config.embed_dim[-1]))
        # TODO, to remove below?
        # trunc_normal_(self.cls_token, std=.02)

        stages = []
        for i in range(config.num_stages):
            dpr = [
                x.item() for x in torch.linspace(0, config.drop_path_rate[i], config.depth[i])
            ]  # stochastic depth decay rule
            layers = []
            for _ in range(config.depth[i]):
                layers.append(
                    CvtLayer(
                        num_heads=config.num_heads[i],
                        embed_dim=config.embed_dim[i],
                        kernel_size=config.kernel_qkv[i],
                        padding_q=config.padding_q[i],
                        padding_kv=config.padding_kv[i],
                        stride_kv=config.stride_kv[i],
                        stride_q=config.stride_q[i],
                        qkv_projection_method=config.qkv_projection_method[i],
                        qkv_bias=config.qkv_bias[i],
                        attention_drop_rate=config.attention_drop_rate[i],
                        drop_rate=config.drop_rate[i],
                        drop_path_rate=dpr[i],
                        mlp_ratio=config.mlp_ratio[i],
                        with_cls_token=config.cls_token[i],
                    )
                )
            stages.append(nn.ModuleList(layers))

        self.stages = nn.ModuleList(stages)

    def forward(
        self,
        pixel_values,
        head_mask=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    ):
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        hidden_states = pixel_values

        cls_token = None
        for i, (embedding_layer, stage) in enumerate(zip(self.patch_embeddings, self.stages)):
            if self.config.cls_token[i]:
                cls_token = self.cls_token
            else:
                cls_token = None

            layer_head_mask = head_mask[i] if head_mask is not None else None
            hidden_states = embedding_layer(hidden_states)

            batch_size, num_channels, height, width = hidden_states.shape
            # rearrange b c h w -> b (h w) c"
            hidden_states = hidden_states.view(batch_size, num_channels, height * width).permute(0, 2, 1)

            if cls_token is not None:
                cls_token = cls_token.expand(batch_size, -1, -1)
                hidden_states = torch.cat((cls_token, hidden_states), dim=1)

            for layer in stage:
                layer_outputs = layer(hidden_states, height, width, layer_head_mask, output_attentions)
                hidden_states = layer_outputs[0]
                if output_attentions:
                    all_self_attentions = all_self_attentions + (layer_outputs[1],)

            if cls_token is not None:
                cls_token, hidden_states = torch.split(hidden_states, [1, height * width], 1)

            # rearrange b (h w) c -> b c h w"
            hidden_states = hidden_states.permute(0, 2, 1).view(batch_size, num_channels, height, width)

            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v for v in [hidden_states, cls_token, all_hidden_states, all_self_attentions] if v is not None
            )

        return BaseModelOutputWithCLSToken(
            last_hidden_state=hidden_states,
            cls_token_value=cls_token,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )


class CvtPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = CvtConfig
    base_model_prefix = "cvt"
    main_input_name = "pixel_values"

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


CVT_START_DOCSTRING = r"""
    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it
    as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`CvtConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

CVT_INPUTS_DOCSTRING = r"""
    Args:
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Pixel values. Pixel values can be obtained using [`CvtFeatureExtractor`]. See
            [`CvtFeatureExtractor.__call__`] for details.

        head_mask (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):
            Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.

        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        interpolate_pos_encoding (`bool`, *optional*):
            Whether to interpolate the pre-trained position encodings.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~file_utils.ModelOutput`] instead of a plain tuple.
"""


@add_start_docstrings(
    "The bare Cvt Model transformer outputting raw hidden-states without any specific head on top.",
    CVT_START_DOCSTRING,
)
class CvtModel(CvtPreTrainedModel):
    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config)
        self.config = config
        self.encoder = CvtEncoder(config)
        self.pooler = CvtPooler(config) if add_pooling_layer else None

        # Initialize weights and apply final processing
        self.post_init()

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    @add_start_docstrings_to_model_forward(CVT_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        processor_class=_FEAT_EXTRACTOR_FOR_DOC,
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=BaseModelOutputWithCLSToken,
        config_class=_CONFIG_FOR_DOC,
        modality="vision",
        expected_output=_EXPECTED_OUTPUT_SHAPE,
    )
    def forward(
        self,
        pixel_values=None,
        head_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, sum(self.config.depth))

        encoder_outputs = self.encoder(
            pixel_values,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]

        if not return_dict:
            return (sequence_output,) + encoder_outputs[1:]

        return BaseModelOutputWithCLSToken(
            last_hidden_state=sequence_output,
            cls_token_value=encoder_outputs.cls_token_value,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


class CvtPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.embed_dim[-1], config.embed_dim[-1])

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        return pooled_output


@add_start_docstrings(
    """
    Cvt Model transformer with an image classification head on top (a linear layer on top of the final hidden state of
    the [CLS] token) e.g. for ImageNet.
    """,
    CVT_START_DOCSTRING,
)
class CvtForImageClassification(CvtPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.num_labels = config.num_labels
        self.cvt = CvtModel(config, add_pooling_layer=False)
        self.layernorm = nn.LayerNorm(config.embed_dim[-1])
        # Classifier head
        self.classifier = (
            nn.Linear(config.embed_dim[-1], config.num_labels) if config.num_labels > 0 else nn.Identity()
        )

        # Initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings_to_model_forward(CVT_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        processor_class=_FEAT_EXTRACTOR_FOR_DOC,
        checkpoint=_IMAGE_CLASS_CHECKPOINT,
        output_type=SequenceClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
        expected_output=_IMAGE_CLASS_EXPECTED_OUTPUT,
    )
    def forward(
        self,
        pixel_values=None,
        head_mask=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the image classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.cvt(
            pixel_values,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        cls_token = outputs[1]
        if self.config.cls_token[-1]:
            sequence_output = self.layernorm(cls_token)
        else:
            batch_size, num_channels, height, width = sequence_output.shape
            # rearrange "b c h w -> b (h w) c"
            sequence_output = sequence_output.view(batch_size, num_channels, height * width).permute(0, 2, 1)
            sequence_output = self.layernorm(sequence_output)

        sequence_output_mean = sequence_output.mean(dim=1)
        logits = self.classifier(sequence_output_mean)

        loss = None
        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
