# coding=utf-8
# Copyright 2023 TSFM team. All rights reserved.
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
""" PyTorch PatchTST model."""

import math
import random
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch
from torch import nn

from ...modeling_outputs import BaseModelOutputWithNoAttention
from ...modeling_utils import PreTrainedModel
from ...time_series_utils import NegativeBinomialOutput, NormalOutput, StudentTOutput
from ...utils import ModelOutput, add_start_docstrings, logging
from .configuration_patchtst import PatchTSTConfig


logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "PatchTSTConfig"

PATCHTST_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "ibm/patchtst-etth1-pretrain",
    # See all PatchTST models at https://huggingface.co/models?filter=patchtst
]


# Copied from transformers.models.bart.modeling_bart.BartAttention with Bart->PatchTST
class PatchTSTAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        is_decoder: bool = False,
        bias: bool = True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads

        if (self.head_dim * num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {num_heads})."
            )
        self.scaling = self.head_dim**-0.5
        self.is_decoder = is_decoder

        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""

        # if key_value_states are provided this layer is used as a cross-attention layer
        # for the decoder
        is_cross_attention = key_value_states is not None

        bsz, tgt_len, _ = hidden_states.size()

        # get query proj
        query_states = self.q_proj(hidden_states) * self.scaling
        # get key, value proj
        # `past_key_value[0].shape[2] == key_value_states.shape[1]`
        # is checking that the `sequence_length` of the `past_key_value` is the same as
        # the provided `key_value_states` to support prefix tuning
        if (
            is_cross_attention
            and past_key_value is not None
            and past_key_value[0].shape[2] == key_value_states.shape[1]
        ):
            # reuse k,v, cross_attentions
            key_states = past_key_value[0]
            value_states = past_key_value[1]
        elif is_cross_attention:
            # cross_attentions
            key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
            value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
        elif past_key_value is not None:
            # reuse k, v, self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
        else:
            # self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

        if self.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_states, value_states)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.reshape(*proj_shape)
        value_states = value_states.reshape(*proj_shape)

        src_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

        if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        if layer_head_mask is not None:
            if layer_head_mask.size() != (self.num_heads,):
                raise ValueError(
                    f"Head mask for a single layer should be of size {(self.num_heads,)}, but is"
                    f" {layer_head_mask.size()}"
                )
            attn_weights = layer_head_mask.view(1, -1, 1, 1) * attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        if output_attentions:
            # this operation is a bit awkward, but it's required to
            # make sure that attn_weights keeps its gradient.
            # In order to do so, attn_weights have to be reshaped
            # twice and have to be reused in the following
            attn_weights_reshaped = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights_reshaped.view(bsz * self.num_heads, tgt_len, src_len)
        else:
            attn_weights_reshaped = None

        attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)

        attn_output = torch.bmm(attn_probs, value_states)

        if attn_output.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz * self.num_heads, tgt_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2)

        # Use the `embed_dim` from the config (stored in the class) rather than `hidden_state` because `attn_output` can be
        # partitioned across GPUs when using tensor-parallelism.
        attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)

        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights_reshaped, past_key_value


def get_activation_fn(activation):
    if callable(activation):
        return activation()
    elif activation.lower() == "relu":
        return nn.ReLU()
    elif activation.lower() == "gelu":
        return nn.GELU()
    raise ValueError(f'{activation} is not available. You can use "relu", "gelu", or a callable')


class Transpose(nn.Module):
    def __init__(self, *dims, contiguous=False):
        super().__init__()
        self.dims, self.contiguous = dims, contiguous

    def forward(self, x):
        if self.contiguous:
            return x.transpose(*self.dims).contiguous()
        else:
            return x.transpose(*self.dims)


def positional_encoding(pe, learn_pe, q_len, d_model):
    # Positional encoding
    if pe is None:
        w_pos = torch.empty((q_len, d_model))  # pe = None and learn_pe = False can be used to measure impact of pe
        nn.init.uniform_(w_pos, -0.02, 0.02)
        learn_pe = False
    elif pe == "zeros":
        w_pos = torch.empty((q_len, d_model))
        nn.init.uniform_(w_pos, -0.02, 0.02)
    elif pe == "normal":
        w_pos = torch.zeros((q_len, 1))
        torch.nn.init.normal_(w_pos, mean=0.0, std=0.1)
    elif pe == "uniform":
        w_pos = torch.zeros((q_len, 1))
        nn.init.uniform_(w_pos, a=0.0, b=0.1)
    elif pe == "sincos":
        pos_enc = torch.zeros(q_len, d_model)
        position = torch.arange(0, q_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pos_enc[:, 0::2] = torch.sin(position * div_term)
        pos_enc[:, 1::2] = torch.cos(position * div_term)
        pos_enc = pos_enc - pos_enc.mean()
        pos_enc = pos_enc / (pos_enc.std() * 10)
        w_pos = pos_enc
    else:
        raise ValueError(
            f"{pe} is not a valid positional encoder. Available types are 'normal', 'zeros', 'zero', uniform', 'sincos', None."
        )
    return nn.Parameter(w_pos, requires_grad=learn_pe)


def set_seed(x=42):
    random.seed(x)
    np.random.seed(x)
    torch.manual_seed(x)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(x)


def random_masking(
    xb: torch.Tensor,
    mask_ratio: float,
    unmasked_channel_indices: list = None,
    channel_consistent_masking: bool = False,
    mask_value=0,
    seed_number: Optional[int] = None,
):
    """random_masking: Mask the input considering the control variables.

    Args:
        xb (Tensor): Input to mask [ bs x nvars x num_patches x patch_length]
        mask_ratio (float): Mask ratio.
        unmasked_channel_indices (list, optional):
            indices of unmasked channels. These channels will not be masked. Defaults to None.
        channel_consistent_masking (bool, optional):
            When true, masking will be same across all channels of a timeseries. Otherwise, masking positions will vary
            across channels. Defaults to True.
        mask_value (int, optional): Value to use for masking. Defaults to 0.
        seed_number (int, optional): Value to set for the random seed.

    Returns:
        Tensor: xb_mask, masked input, same shape as input Tensor: Mask tensor of shape [bs x c x n]
    """
    if seed_number:
        set_seed(seed_number)

    bs, nvars, L, D = xb.shape

    len_keep = int(L * (1 - mask_ratio))

    if channel_consistent_masking:
        noise = torch.rand(bs, 1, L, device=xb.device)  # noise in [0, 1], bs x 1 x  L
        noise = noise.repeat(1, nvars, 1)  # bs x nvars x L
    else:
        noise = torch.rand(bs, nvars, L, device=xb.device)  # noise in [0, 1], bs x nvars x L

    mask = torch.ones(bs, nvars, L, device=xb.device)  # mask: [bs x nvars x num_patch]
    mask[:, :, :len_keep] = 0

    # sort noise for each sample
    ids_shuffle = torch.argsort(noise, dim=-1)  # ascend: small is keep, large is remove
    ids_restore = torch.argsort(ids_shuffle, dim=-1)  # ids_restore: [bs x nvars x L]

    mask = torch.gather(mask, dim=-1, index=ids_restore)
    mask = mask.unsqueeze(-1).repeat(1, 1, 1, D)  # mask: [bs x nvars x num_patches x patch_length]
    if unmasked_channel_indices is not None:
        mask[:, unmasked_channel_indices, :, :] = 0

    xb_mask = xb.masked_fill(mask.bool(), mask_value)
    return xb_mask, mask[..., 0]


def compute_num_patches(sequence_length, patch_length, stride):
    return (max(sequence_length, patch_length) - patch_length) // stride + 1


class Patchify(nn.Module):
    """
    A class to patchify the time series sequence into different patches

    Args:
        sequence_length (int, required): input sequence length.
        patch_length (int, required): patch length.
        stride (int, required): stride between patches.

    Returns:
        z: output tensor data [bs x num_input_channels x num_patches x patch_length]
    """

    def __init__(
        self,
        sequence_length: int,
        patch_length: int,
        stride: int,
        padding: bool = False,  # TODO: use this to set whether we want to pad zeros to the sequence
    ):
        super().__init__()

        assert (
            sequence_length > patch_length
        ), f"Sequence length ({sequence_length}) has to be greater than the patch length ({patch_length})"

        self.sequence_length = sequence_length
        self.patch_length = patch_length
        self.stride = stride

        # get the number of patches
        self.num_patches = compute_num_patches(sequence_length, patch_length, stride)
        new_sequence_length = patch_length + stride * (self.num_patches - 1)
        self.s_begin = sequence_length - new_sequence_length

    def forward(self, past_values: torch.Tensor):
        """
        Args:
            past_values (torch.Tensor, required): Input of shape [bs x sequence_length x num_input_channels]

        Returns:
            x: output tensor data [bs x num_input_channels x num_patches x patch_length]
        """
        sequence_length = past_values.shape[-2]
        assert (
            sequence_length == self.sequence_length
        ), f"Input sequence length ({sequence_length}) doesn't match model configuration ({self.sequence_length})."

        x = past_values[:, self.s_begin :, :]  # x: [bs x new_sequence_length x nvars]
        x = x.unfold(
            dimension=-2, size=self.patch_length, step=self.stride
        )  # x: [bs x num_patches x num_input_channels x patch_length]
        x = x.transpose(-2, -3).contiguous()  # xb: [bs x num_input_channels x num_patches x patch_length]
        return x


class PatchEmbeddings(nn.Module):
    """
    Args:
    A class to patchify the time series sequence into different patches
        sequence_length (int, required): input sequence length. patch_length (int, required): patch length. stride
        (int, required): stride between patches.

    Returns:
        embeddings: output tensor data [bs x num_input_channels x num_patches x embed_dim]
    """

    def __init__(self, sequence_length: int, patch_length: int, stride: int, embed_dim: int):
        super().__init__()

        assert (
            sequence_length > patch_length
        ), f"Sequence length ({sequence_length}) has to be greater than the patch length ({patch_length})"

        # assert ((max(sequence_length, patch_length) - patch_length) % stride == 0), f"sequence length minus patch length has to be divisible to the stride"

        self.sequence_length = sequence_length
        self.patch_length = patch_length
        self.stride = stride
        self.embed_dim = embed_dim

        # get the number of patches
        self.num_patches = compute_num_patches(sequence_length, patch_length, stride)
        new_sequence_length = patch_length + stride * (self.num_patches - 1)
        self.s_begin = sequence_length - new_sequence_length

        # Embedding
        self.projection = nn.Conv1d(
            in_channels=1,
            out_channels=embed_dim,
            kernel_size=patch_length,
            stride=stride,
        )

    def forward(self, past_values: torch.Tensor):
        """
        Args:
            past_values (torch.Tensor, required): Input of shape [bs x sequence_length x num_input_channels]
        Returns:
            embeddings: output tensor data [bs x num_input_channels x num_patches x emb_dim]
        """
        bs, sequence_length, num_input_channels = past_values.shape
        assert (
            sequence_length == self.sequence_length
        ), f"Input sequence length ({sequence_length}) doesn't match the configuration sequence length ({self.sequence_length})."

        x = past_values[:, self.s_begin :, :]  # x: [bs x new_sequence_length x nvars]
        # convert past_values to shape [bs*num_input_channels x 1 x sequence_length ]
        x = x.transpose(1, 2).reshape(bs * num_input_channels, 1, -1).contiguous()
        # projection
        embeddings = self.projection(x)  # embeddings: [bs*num_input_channels x emb_dim x num_patches]
        # reshape
        embeddings = (
            embeddings.transpose(1, 2).view(bs, num_input_channels, -1, self.embed_dim).contiguous()
        )  # embeddings: [bs x num_input_channels x num_patches x emb_dim]
        # embeddings = embeddings.flatten(2).transpose(1, 2)
        return embeddings


class PatchMasking(nn.Module):
    """
    PatchMasking: Class to random or forcast masking.

    Args:
        mask_type (str, optional): Masking type. Allowed values are random, forecast. Defaults to random.
        mask_ratio (float, optional): Mask ratio.
        mask_patches (list, optional): List of patch lengths to mask in the end of the data.
        mask_patch_ratios (list, optional): List of weights to use for each patch length. For Ex.
        if patch_lengths is [5,4] and mix_ratio is [1,1], then equal weights to both patch lengths. Defaults to None.
        unmasked_channel_indices (list, optional):
            Control Variable channel indices. These channels will not be masked. Defaults to None.
        channel_consistent_masking (bool, optional):
            When true, masking will be same across all channels of a timeseries. Otherwise, masking positions will vary
            across channels. Defaults to True.
        mask_value (int, optional): Value to use for masking. Defaults to 0.
        seed_number (int, optional): Random seed, when None seed is not set. Defaults to None.
    """

    def __init__(
        self,
        mask_type: str = "random",
        mask_ratio=0.5,
        mask_patches: list = [2, 3],
        mask_patch_ratios: list = [1, 1],
        channel_consistent_masking: bool = False,
        unmasked_channel_indices: list = None,
        mask_value=0,
        seed_number: Optional[int] = None,
    ):
        # if seed_number:
        #     set_seed(seed_number)
        self.mask_ratio = mask_ratio
        self.channel_consistent_masking = channel_consistent_masking
        self.mask_type = mask_type
        self.mask_patches = mask_patches
        self.mask_patch_ratios = mask_patch_ratios
        self.unmasked_channel_indices = unmasked_channel_indices
        self.mask_value = mask_value
        if self.unmasked_channel_indices is not None:
            self.unmasked_channel_indices.sort()
        self.seed_number = seed_number

        super().__init__()

    def forward(self, x: torch.Tensor):
        """
        Input:
            x: patched input
                4D: [bs x num_input_channels x num_patches x patch_length]

        Output:
            x_mask: Masked patched input
                4D: [bs x num_input_channels x num_patches x patch_length]
            mask: bool tensor indicating True on masked points
                4D: [bs x num_input_channels x num_patch]
        """

        if self.mask_type == "random":
            x_mask, mask = random_masking(
                xb=x,
                mask_ratio=self.mask_ratio,
                unmasked_channel_indices=self.unmasked_channel_indices,
                channel_consistent_masking=self.channel_consistent_masking,
                mask_value=self.mask_value,
                seed_number=self.seed_number,
            )

        else:
            raise Exception("Invalid mask type")

        mask = mask.bool()  # mask: [bs x num_input_channels x num_patch]

        return x_mask, mask


class ChannelAttentionTSTEncoder(nn.Module):
    def __init__(self, config: PatchTSTConfig):
        super().__init__()

        self.layers = nn.ModuleList([ChannelAttentionTSTEncoderLayer(config) for i in range(config.encoder_layers)])

    def forward(self, src: torch.Tensor, output_hidden_states: Optional[bool] = None):
        """
        src: tensor [bs x nvars x sequence_length x d_model] Return:
            Tensor [bs x nvars x sequence_length x d_model]
        """
        all_hidden_states = []

        for mod in self.layers:
            src = mod(src)
            if output_hidden_states:
                all_hidden_states.append(src)
        if output_hidden_states is None:
            return src, None
        return src, all_hidden_states


class ChannelAttentionTSTEncoderLayer(nn.Module):
    def __init__(self, config: PatchTSTConfig):
        super().__init__()

        self.channel_attention = config.channel_attention
        # Multi-Head attention
        # self.self_attn = PatchTSTAttention(config)

        self.self_attn = PatchTSTAttention(
            embed_dim=config.d_model,
            num_heads=config.encoder_attention_heads,
            dropout=config.attention_dropout,
        )

        # Add & Norm of the sublayer 1
        self.dropout_path1 = nn.Dropout(config.dropout_path) if config.dropout_path > 0 else nn.Identity()
        if "batch" in config.norm.lower():
            self.norm_sublayer1 = nn.Sequential(Transpose(1, 2), nn.BatchNorm1d(config.d_model), Transpose(1, 2))
        else:
            self.norm_sublayer1 = nn.LayerNorm(config.d_model)

        # Add & Norm of the sublayer 2
        if self.channel_attention:
            self.dropout_path2 = nn.Dropout(config.dropout_path) if config.dropout_path > 0 else nn.Identity()
            if "batch" in config.norm.lower():
                self.norm_sublayer2 = nn.Sequential(Transpose(1, 2), nn.BatchNorm1d(config.d_model), Transpose(1, 2))
            else:
                self.norm_sublayer2 = nn.LayerNorm(config.d_model)

        # Position-wise Feed-Forward
        self.ff = nn.Sequential(
            nn.Linear(config.d_model, config.encoder_ffn_dim, bias=config.bias),
            get_activation_fn(config.activation_function),
            nn.Dropout(config.ff_dropout) if config.ff_dropout > 0 else nn.Identity(),
            nn.Linear(config.encoder_ffn_dim, config.d_model, bias=config.bias),
        )

        # Add & Norm of sublayer 3
        self.dropout_path3 = nn.Dropout(config.dropout_path) if config.dropout_path > 0 else nn.Identity()
        if "batch" in config.norm.lower():
            self.norm_sublayer3 = nn.Sequential(Transpose(1, 2), nn.BatchNorm1d(config.d_model), Transpose(1, 2))
        else:
            self.norm_sublayer3 = nn.LayerNorm(config.d_model)

        self.pre_norm = config.pre_norm

    def forward(self, src: torch.Tensor):
        """
        src: tensor [bs x nvars x sequence_length x d_model] Return:
            Tensor [bs x nvars x sequence_length x d_model]
        """
        bs, num_input_channels, sequence_length, d_model = src.shape

        # First sublayer: attention across time
        src = src.view(
            bs * num_input_channels, sequence_length, d_model
        )  # src: [(bs*nvars) x sequence_length x d_model]
        if self.pre_norm:
            ## Norm and Multi-Head attention and Add residual connection
            src = src + self.dropout_path1(
                self.self_attn(self.norm_sublayer1(src)[0])
            )  # Add: residual connection with residual dropout
        else:
            ## Multi-Head attention and Add residual connection and Norm - Standard Transformer from BERT
            src = self.norm_sublayer1(
                src + self.dropout_path1(self.self_attn(src)[0])
            )  # src: [(bs*nvars) x sequence_length x d_model]
        src = src.reshape(bs, num_input_channels, sequence_length, d_model)  # [bs x nvars x sequence_length x d_model]

        # second sublayer: attention across variable at any given time
        # [bs x nvars x sequence_length x d_model] -> [bs x sequence_length x nvars x d_model] -> [(bs*sequence_length) x nvars x d_model]
        if self.channel_attention:
            src = (
                src.transpose(2, 1).contiguous().view(bs * sequence_length, num_input_channels, d_model)
            )  # [(bs*sequence_length) x nvars x d_model]
            if self.pre_norm:
                ## Norm and Multi-Head attention and Add residual connection
                src = src + self.dropout_path2(
                    self.self_attn(self.norm_sublayer2(src)[0])
                )  # Add: residual connection with residual dropout
            else:
                ## Multi-Head attention and Add residual connection and Norm
                src = self.norm_sublayer2(
                    src + self.dropout_path2(self.self_attn(src)[0])
                )  # src: [(bs*sequence_length) x nvars x d_model]
            src = (
                src.reshape(bs, sequence_length, num_input_channels, d_model).transpose(1, 2).contiguous()
            )  # src: [bs x nvars x sequence_length x d_model]

        # Third sublayer: mixing across hidden
        src = src.view(
            bs * num_input_channels, sequence_length, d_model
        )  # src: [(bs*nvars) x sequence_length x d_model]
        if self.pre_norm:
            ## Norm and Position-wise Feed-Forward and Add residual connection
            src = src + self.dropout_path3(
                self.ff(self.norm_sublayer3(src))
            )  # Add: residual connection with residual dropout
        else:
            ## Position-wise Feed-Forward and Add residual connection and Norm
            src = self.norm_sublayer3(
                src + self.dropout_path3(self.ff(src))
            )  # Add: residual connection with residual dropout
        src = src.reshape(bs, num_input_channels, sequence_length, d_model)  # [bs x nvars x sequence_length x d_model]

        return src


class PatchTSTPreTrainedModel(PreTrainedModel):
    config_class = PatchTSTConfig
    base_model_prefix = "model"
    main_input_name = "past_values"
    supports_gradient_checkpointing = True

    def _init_weights(self, module):
        """Initialize weights"""
        if self.config.use_cls_token:
            torch.nn.init.normal_(self.config.cls_token, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, (nn.Linear, nn.Conv1d)):
            module.weight.data.normal_(mean=0.0, std=self.config.init_std)
            if module.bias is not None:
                module.bias.data.zero_()

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, (ChannelAttentionPatchTSTEncoder)):
            module.gradient_checkpointing = value


class ChannelAttentionPatchTSTEncoder(PatchTSTPreTrainedModel):
    def __init__(self, config: PatchTSTConfig):
        super().__init__(config)
        self.num_input_channels = config.num_input_channels
        self.num_patches = config.num_patches
        self.patch_length = config.patch_length
        self.d_model = config.d_model
        self.shared_embedding = config.shared_embedding
        self.use_cls_token = config.use_cls_token
        self.gradient_checkpointing = False

        # Input encoding: projection of feature vectors onto a d-dim vector space
        if not config.shared_embedding:
            self.w_p = nn.ModuleList()
            for _ in range(self.num_input_channels):
                self.w_p.append(nn.Linear(config.patch_length, config.d_model))
        else:
            self.w_p = nn.Linear(config.patch_length, config.d_model)

        # Positional encoding
        if config.use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, 1, config.d_model))
            self.w_pos = positional_encoding(
                config.positional_encoding, config.learn_pe, config.num_patches + 1, config.d_model
            )
        else:
            self.w_pos = positional_encoding(
                config.positional_encoding, config.learn_pe, config.num_patches, config.d_model
            )

        # Positional dropout
        self.positional_dropout = (
            nn.Dropout(config.positional_dropout) if config.positional_dropout > 0 else nn.Identity()
        )

        # Encoder
        self.encoder = ChannelAttentionTSTEncoder(config)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self, past_values: torch.Tensor, output_hidden_states: Optional[bool] = None
    ) -> BaseModelOutputWithNoAttention:
        """
        Args:
            past_values: tensor [bs x nvars x num_patches x patch_length].
            output_hidden_states (bool, optional): Indicates if hidden states should be output.

        return:
            tensor [bs x nvars x num_patches x d_model]
                or [bs x nvars x (num_patches+1) x d_model] if use cls_token
        """
        # bs, num_patches, num_input_channels, patch_length = x.shape
        bs, num_input_channels, num_patches, patch_length = past_values.shape

        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # Input encoding
        if not self.shared_embedding:
            x_out = []
            for i in range(num_input_channels):
                z = self.w_p[i](past_values[:, i, :, :])
                x_out.append(z)
            past_values = torch.stack(x_out, dim=1)
        else:
            past_values = self.w_p(past_values)  # x: [bs x nvars  x num_patches x d_model]

        if self.use_cls_token:
            # x: [bs x nvars x num_patches x d_model]
            past_values = self.positional_dropout(past_values + self.w_pos[1:, :])
            # append cls token
            cls_token = self.cls_token + self.w_pos[:1, :]  # cls_token: [1 x 1 x 1 x d_model]
            cls_tokens = cls_token.expand(past_values.shape[0], -1, -1)  # get the same copy for all the batch samples
            past_values = torch.cat((cls_tokens, past_values), dim=1)  # x: [bs x nvars x (num_patches+1) x d_model]
        else:
            past_values = self.positional_dropout(past_values + self.w_pos)  # x: [bs x nvars x num_patches x d_model]

        # Encoder
        past_values, hidden_states = self.encoder(
            past_values, output_hidden_states
        )  # x: [bs x nvars x num_patches x d_model]
        # or [bs x nvars x (num_patches+1) x d_model] if use cls_token

        # return past_values, hidden_states
        return BaseModelOutputWithNoAttention(last_hidden_state=past_values, hidden_states=hidden_states)


PATCHTST_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`PatchTSTConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

PATCHTST_INPUTS_DOCSTRING = r"""
    Args:
        past_values (`torch.FloatTensor` of shape `(batch_size, sequence_length)` or `(batch_size, sequence_length, num_input_channels)`):
            Past values of the time series, that serve as context in order to predict the future. The sequence size of
            this tensor must be larger than the `context_length` of the model, since the model will use the larger size
            to construct lag features, i.e. additional values from the past which are added in order to serve as "extra
            context".

            The `sequence_length` here is equal to `config.context_length`

            The `past_values` is what the Transformer encoder gets as input (with optional additional features, such as
            `static_categorical_features`, `static_real_features`).

            For multivariate time series, the `num_input_channels` > 1 dimension is required and corresponds to the
            number of variates in the time series per time step.

        future_values (`torch.FloatTensor` of shape `(batch_size, prediction_length)` or `(batch_size, prediction_length, num_input_channels)`, *optional*):
            Future values of the time series, that serve as labels for the model. The `future_values` is what the
            Transformer needs during training to learn to output, given the `past_values`.

            The sequence length here is equal to `prediction_length`.

            See the demo notebook and code snippets for details.

            For multivariate time series, the `num_input_channels` > 1 dimension is required and corresponds to the
            number of variates in the time series per time step.

        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers.
"""


@dataclass
@add_start_docstrings(
    "The bare PatchTST Model outputting raw hidden-states without any specific head.",
    PATCHTST_START_DOCSTRING,
)
class PatchTSTModelOutputWithNoAttention(ModelOutput):
    """
    Base class for model's outputs, with potential hidden states.

    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, num_channels, num_patches, patch_length)`):
            Sequence of hidden-states at the output of the last layer of the model.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, num_channels, height, width)`. Hidden-states of
            the model at the output of each layer plus the optional initial embedding outputs.
        patched_input (`torch.FloatTensor` of shape `(batch_size, num_channels, num_patches, patch_length)`):
            patched input to the Transformer
        mask: (`torch.FloatTensor` of shape `(batch_size, num_channels, num_patches)`,*optional*)
            Bool masked tensor indicating which patches are masked
        loc: (`torch.FloatTensor` of shape `(batch_size, 1, num_channels)`,*optional*)
            mean of the input data (batch_size, sequence_length, num_channels) over the sequence_length
        scale: (`torch.FloatTensor` of shape `(batch_size, 1, num_channels)`,*optional*)
            std of the input data (batch_size, sequence_length, num_channels) over the sequence_length
    """

    last_hidden_state: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    patched_input: torch.FloatTensor = None
    mask: torch.FloatTensor = None
    loc: torch.FloatTensor = None
    scale: torch.FloatTensor = None


# Copied from transformers.models.time_series_transformer.modeling_time_series_transformer.nll
def nll(input: torch.distributions.Distribution, target: torch.Tensor) -> torch.Tensor:
    """
    Computes the negative log likelihood loss from input distribution with respect to target.
    """
    return -input.log_prob(target)


# Copied from transformers.models.time_series_transformer.modeling_time_series_transformer.TimeSeriesStdScaler with TimeSeries->PatchTST
class PatchTSTStdScaler(nn.Module):
    """
    Standardize features by calculating the mean and scaling along some given dimension `dim`, and then normalizes it
    by subtracting from the mean and dividing by the standard deviation.

    Args:
        dim (`int`):
            Dimension along which to calculate the mean and standard deviation.
        keepdim (`bool`, *optional*, defaults to `False`):
            Controls whether to retain dimension `dim` (of length 1) in the scale tensor, or suppress it.
        minimum_scale (`float`, *optional*, defaults to 1e-5):
            Default scale that is used for elements that are constantly zero along dimension `dim`.
    """

    def __init__(self, dim: int, keepdim: bool = False, minimum_scale: float = 1e-5):
        super().__init__()
        if not dim > 0:
            raise ValueError("Cannot compute scale along dim = 0 (batch dimension), please provide dim > 0")
        self.dim = dim
        self.keepdim = keepdim
        self.minimum_scale = minimum_scale

    @torch.no_grad()
    def forward(self, data: torch.Tensor, weights: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        denominator = weights.sum(self.dim, keepdim=self.keepdim)
        denominator = denominator.clamp_min(1.0)
        loc = (data * weights).sum(self.dim, keepdim=self.keepdim) / denominator

        variance = (((data - loc) * weights) ** 2).sum(self.dim, keepdim=self.keepdim) / denominator
        scale = torch.sqrt(variance + self.minimum_scale)
        return (data - loc) / scale, loc, scale


# Copied from transformers.models.time_series_transformer.modeling_time_series_transformer.TimeSeriesMeanScaler with TimeSeries->PatchTST
class PatchTSTMeanScaler(nn.Module):
    """
    Computes a scaling factor as the weighted average absolute value along dimension `dim`, and scales the data
    accordingly.

    Args:
        dim (`int`):
            Dimension along which to compute the scale.
        keepdim (`bool`, *optional*, defaults to `False`):
            Controls whether to retain dimension `dim` (of length 1) in the scale tensor, or suppress it.
        default_scale (`float`, *optional*, defaults to `None`):
            Default scale that is used for elements that are constantly zero. If `None`, we use the scale of the batch.
        minimum_scale (`float`, *optional*, defaults to 1e-10):
            Default minimum possible scale that is used for any item.
    """

    def __init__(
        self, dim: int = -1, keepdim: bool = True, default_scale: Optional[float] = None, minimum_scale: float = 1e-10
    ):
        super().__init__()
        self.dim = dim
        self.keepdim = keepdim
        self.minimum_scale = minimum_scale
        self.default_scale = default_scale

    @torch.no_grad()
    def forward(
        self, data: torch.Tensor, observed_indicator: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # shape: (N, [C], T=1)
        ts_sum = (data * observed_indicator).abs().sum(self.dim, keepdim=True)
        num_observed = observed_indicator.sum(self.dim, keepdim=True)

        scale = ts_sum / torch.clamp(num_observed, min=1)

        # If `default_scale` is provided, we use it, otherwise we use the scale
        # of the batch.
        if self.default_scale is None:
            batch_sum = ts_sum.sum(dim=0)
            batch_observations = torch.clamp(num_observed.sum(0), min=1)
            default_scale = torch.squeeze(batch_sum / batch_observations)
        else:
            default_scale = self.default_scale * torch.ones_like(scale)

        # apply default scale where there are no observations
        scale = torch.where(num_observed > 0, scale, default_scale)

        # ensure the scale is at least `self.minimum_scale`
        scale = torch.clamp(scale, min=self.minimum_scale)
        scaled_data = data / scale

        if not self.keepdim:
            scale = scale.squeeze(dim=self.dim)

        return scaled_data, torch.zeros_like(scale), scale


# Copied from transformers.models.time_series_transformer.modeling_time_series_transformer.TimeSeriesNOPScaler with TimeSeries->PatchTST
class PatchTSTNOPScaler(nn.Module):
    """
    Assigns a scaling factor equal to 1 along dimension `dim`, and therefore applies no scaling to the input data.

    Args:
        dim (`int`):
            Dimension along which to compute the scale.
        keepdim (`bool`, *optional*, defaults to `False`):
            Controls whether to retain dimension `dim` (of length 1) in the scale tensor, or suppress it.
    """

    def __init__(self, dim: int, keepdim: bool = False):
        super().__init__()
        self.dim = dim
        self.keepdim = keepdim

    def forward(
        self, data: torch.Tensor, observed_indicator: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        scale = torch.ones_like(data, requires_grad=False).mean(dim=self.dim, keepdim=self.keepdim)
        loc = torch.zeros_like(data, requires_grad=False).mean(dim=self.dim, keepdim=self.keepdim)
        return data, loc, scale


class PatchTSTModel(PatchTSTPreTrainedModel):
    def __init__(self, config: PatchTSTConfig):
        super().__init__(config)

        if config.scaling == "mean" or config.scaling is True:
            self.scaler = PatchTSTMeanScaler(dim=1, keepdim=True)
        elif config.scaling == "std":
            self.scaler = PatchTSTStdScaler(dim=1, keepdim=True)
        else:
            self.scaler = PatchTSTNOPScaler(dim=1, keepdim=True)

        self.patching = Patchify(
            config.context_length,
            patch_length=config.patch_length,
            stride=config.stride,
        )
        self.mask_input = config.mask_input

        if self.mask_input:
            self.masking = PatchMasking(
                mask_type=config.mask_type,
                mask_ratio=config.mask_ratio,
                mask_patches=config.mask_patches,
                mask_patch_ratios=config.mask_patch_ratios,
                channel_consistent_masking=config.channel_consistent_masking,
                unmasked_channel_indices=config.unmasked_channel_indices,
                mask_value=config.mask_value,
                seed_number=config.seed_number,
            )
        else:
            self.masking = nn.Identity()
        self.encoder = ChannelAttentionPatchTSTEncoder(config)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        past_values: torch.Tensor,
        past_observed_mask: Optional[torch.Tensor] = None,
        future_values: Optional[torch.Tensor] = None,
        output_hidden_states: Optional[bool] = None,
    ):
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        if past_observed_mask is None:
            past_observed_mask = torch.ones_like(past_values)

        # x: tensor [bs x seq_len x in_channels]
        scaled_past_values, loc, scale = self.scaler(past_values, past_observed_mask)

        # patched_values: [bs x num_input_channels x num_patches x patch_length] for pretrain
        patched_values = self.patching(scaled_past_values)
        if self.mask_input:
            masked_values, mask = self.masking(patched_values)
        else:
            masked_values, mask = self.masking(patched_values), None
        encoder_output = self.encoder(masked_values, output_hidden_states=output_hidden_states)
        return PatchTSTModelOutputWithNoAttention(
            last_hidden_state=encoder_output.last_hidden_state,
            hidden_states=encoder_output.hidden_states,
            patched_input=patched_values,
            mask=mask,
            loc=loc,
            scale=scale,
        )


class MaskPretrainHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dropout = nn.Dropout(config.dropout)
        self.linear = nn.Linear(config.d_model, config.patch_length)
        self.use_cls_token = config.use_cls_token

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: tensor [bs x nvars x num_patches x d_model]
                or [bs x nvars x (num_patches+1) x d_model] if use cls_token
        output: tensor [bs x nvars x num_patches x patch_length]
        """
        x = self.linear(self.dropout(x))  # [bs x nvars x num_patches x patch_length]
        if self.use_cls_token:
            x = x[:, :, 1:, :]  # remove the first cls token
        return x


@dataclass
class PatchTSTOutput(ModelOutput):
    """
    Output type of [`PatchTSTForPredictiontion`].

    Args:
        loss (*optional*, returned when `labels` is provided, `torch.FloatTensor` of shape `(1,)`):
            MSE loss.
        prediction_outputs (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction outputs of the time series modeling heads.
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

    loss: Optional[torch.FloatTensor] = None
    prediction_output: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


class PatchTSTForMaskPretraining(PatchTSTPreTrainedModel):
    # PatchTSTModel + Pretraining Head
    def __init__(self, config: PatchTSTConfig):
        super().__init__(config)

        config.mask_input = True
        self.model = PatchTSTModel(config=config)
        self.head = MaskPretrainHead(config)
        self.loss = torch.nn.MSELoss(reduction="none")

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        past_values: torch.Tensor,
        future_values: Optional[torch.Tensor] = None,
        output_hidden_states: Optional[bool] = None,
    ) -> PatchTSTOutput:
        """
        past_values (x): tensor [bs x sequence_length x num_input_channels ] future_values (y): labels
        """
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        # past_values: [bs x nvars x num_patches x d_model] or
        # [bs x nvars x (num_patches+1) x d_model] if use cls_token
        model_output = self.model(past_values, output_hidden_states=output_hidden_states)

        # model_output[0]: [bs x nvars x num_patches x patch_length] or
        # [bs x nvars x (num_patches+1) x patch_length] if use cls_token
        x_hat = self.head(model_output[0])

        # calculate masked_loss
        loss_val = self.loss(x_hat, model_output.patched_input)
        masked_loss = (loss_val.mean(dim=-1) * model_output.mask).sum() / (model_output.mask.sum() + 1e-10)

        return PatchTSTOutput(loss=masked_loss, prediction_output=x_hat, hidden_states=model_output.hidden_states)


class PatchTSTForClassification(PatchTSTPreTrainedModel):
    # PatchTST model + classification head
    def __init__(self, config: PatchTSTConfig):
        super().__init__(config)

        self.model = PatchTSTModel(config)
        self.head = ClassificationHead(config)
        self.loss = nn.CrossEntropyLoss()

        # Initialize weights and apply final processing
        self.post_init()

    def forward(self, past_values, labels=None, output_hidden_states: Optional[bool] = None):
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        model_output = self.model(past_values, output_hidden_states=output_hidden_states)
        y_hat = self.head(model_output[0])

        loss_val = None
        if labels is not None:
            loss_val = self.loss(y_hat, labels)
        return PatchTSTForClassificationOutput(
            loss=loss_val, prediction_logits=y_hat, hidden_states=model_output.hidden_states
        )


class ClassificationHead(nn.Module):
    def __init__(self, config: PatchTSTConfig):
        super().__init__()
        self.use_cls_token = config.use_cls_token
        self.pooling = config.pooling
        self.flatten = nn.Flatten(start_dim=1)
        self.dropout = nn.Dropout(config.head_dropout) if config.head_dropout > 0 else nn.Identity()
        self.linear = nn.Linear(config.num_input_channels * config.d_model, config.num_classes)

    def forward(self, x):
        """
        x: [bs x nvars x num_patches x d_model] or [bs x nvars x (num_patches+1) x d_model] if use cls_token output:
        [bs x n_classes]
        """
        if self.use_cls_token:
            x = x[:, :, 0, :]  # use the first output token, x: bs x nvars x d_model
        elif self.pooling == "mean":
            x = x.mean(dim=2)  # x: [bs x nvars x d_model]
        elif self.pooling == "max":
            x = x.max(dim=2)  # x: [bs x nvars x d_model]
        else:
            raise Exception(f"pooling operator {self.pooling} is not implemented yet")

        x = self.flatten(x)  # x: bs x nvars * d_model
        y = self.linear(self.dropout(x))  # y: bs x n_classes
        return y


@dataclass
class PatchTSTForClassificationOutput(ModelOutput):
    """
    Output type of [`PatchTSTForClassification`].

    Args:
        loss (*optional*, returned when `labels` is provided, `torch.FloatTensor` of shape `(1,)`):
            Total loss as the sum of the masked language modeling loss and the next sequence prediction
            (classification) loss.
        prediction_logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
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

    loss: Optional[torch.FloatTensor] = None
    prediction_logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


class PredictionHead(nn.Module):
    def __init__(self, config: PatchTSTConfig, distribution_output=None):
        super().__init__()

        self.num_output_channels = config.num_output_channels
        self.dist_output_size = config.num_output_channels * config.d_model // config.encoder_attention_heads
        self.use_cls_token = config.use_cls_token
        self.pooling = config.pooling

        head_dim = config.num_input_channels * config.d_model

        self.flatten = nn.Flatten(start_dim=1)
        if distribution_output is None:
            self.linear = nn.Linear(head_dim, config.prediction_length * config.num_output_channels)
            self.args_proj = None
        else:
            self.linear = nn.Linearr(head_dim, config.prediction_length * self.dist_output_size)
            self.args_proj = distribution_output.get_parameter_projection(self.dist_output_size)
        self.dropout = nn.Dropout(config.head_dropout) if config.head_dropout > 0 else nn.Identity()

    def forward(self, x):
        """
        x: [bs x nvars x num_patch x d_model]
            or [bs x nvars x (num_patch+1) x d_model] if use cls_token
        output: [bs x pred_len x num_output_channels]
        """
        batch_size = x.shape[0]
        if self.use_cls_token:
            x = x[:, :, 0, :]  # use the first output token, x: [bs x nvars x d_model]
        elif self.pooling == "mean":
            x = x.mean(dim=2)  # x: [bs x nvars x d_model]
        elif self.pooling == "max":
            x = x.max(dim=2)  # x: [bs x nvars x d_model]
        else:
            raise Exception(f"pooling operator {self.pooling} is not implemented yet")

        # flatten the input
        x = self.flatten(x)  # x: bs x (nvars * d_model)
        y = self.linear(self.dropout(x))  # y: bs x (pred_len * num_output_channels)
        if self.args_proj is None:
            # reshape the data
            y = y.reshape(batch_size, -1, self.num_output_channels)  # [bs x pred_len x num_output_channels]
        else:
            # reshape and project prarameters of distribution
            y = self.args_proj(y.reshape(batch_size, -1, self.dist_output_size))

        return y


class PatchTSTForPrediction(PatchTSTPreTrainedModel):
    # PatchTST model + prediction head
    def __init__(self, config: PatchTSTConfig):
        super().__init__(config)

        self.model = PatchTSTModel(config)
        if config.loss == "mse":
            self.loss = nn.MSELoss(reduction="mean")
            self.distribution_output = None
        else:
            self.loss = nll
            if config.distribution_output == "student_t":
                self.distribution_output = StudentTOutput(dim=config.num_output_channels)
            elif config.distribution_output == "normal":
                self.distribution_output = NormalOutput(dim=config.num_output_channels)
            elif config.distribution_output == "negative_binomial":
                self.distribution_output = NegativeBinomialOutput(dim=config.num_output_channels)
            else:
                raise ValueError(f"Unknown distribution output {config.distribution_output}")

        self.head = PredictionHead(config, self.distribution_output)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        past_values: torch.Tensor,
        future_values: Optional[torch.Tensor] = None,
        output_hidden_states: Optional[bool] = None,
    ):
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        model_output = self.model(past_values, output_hidden_states=output_hidden_states)
        y_hat = self.head(model_output.last_hidden_state)
        loss_val = None
        if future_values is not None:
            if self.distribution_output:
                distribution = self.distribution_output.distribution(
                    y_hat, loc=model_output.loc, scale=model_output.scale
                )
                loss_val = self.loss(distribution, future_values)
            else:
                loss_val = self.loss(y_hat * model_output.scale + model_output.loc, future_values)
        return PatchTSTOutput(loss=loss_val, prediction_output=y_hat, hidden_states=model_output.hidden_states)


@dataclass
class PatchTSTForForecastingOutput(ModelOutput):
    """
    Output type of [`PatchTSTForPredictiontion`].

    Args:
        loss (*optional*, returned when `labels` is provided, `torch.FloatTensor` of shape `(1,)`):
            MSE loss.

        forecast_outputs (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Forecasting outputs of the time series modeling heads.

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

    loss: Optional[torch.FloatTensor] = None
    forecast_outputs: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


class ForecastHead(nn.Module):
    def __init__(self, config: PatchTSTConfig):
        super().__init__()

        self.shared_projection = config.shared_projection
        self.num_input_channels = config.num_input_channels
        self.use_cls_token = config.use_cls_token
        self.pooling = config.pooling
        head_dim = config.d_model if self.pooling else config.d_model * config.num_patches

        if not self.shared_projection:
            self.linears = nn.ModuleList()
            self.dropouts = nn.ModuleList()
            self.flattens = nn.ModuleList()
            for i in range(self.num_input_channels):
                self.flattens.append(nn.Flatten(start_dim=2))
                self.linears.append(nn.Linear(head_dim, config.prediction_length))
                self.dropouts.append(nn.Dropout(config.head_dropout) if config.head_dropout > 0 else nn.Identity())
        else:
            self.flatten = nn.Flatten(start_dim=2)
            self.linear = nn.Linear(head_dim, config.prediction_length)
            self.dropout = nn.Dropout(config.head_dropout) if config.head_dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor):
        """
        x: [bs x nvars x num_patches x d_model]
            or [bs x nvars x (num_patches+1) x d_model] if use cls_token
        output: [bs x forecast_len x nvars]
        """

        if self.use_cls_token:
            y = x[:, :, 0, :]  # y: [bs x nvars x d_model]
        else:
            if self.pooling == "mean":
                y = x.mean(dim=2)  # y: [bs x nvars x d_model]
            elif self.pooling == "max":
                y = x.max(dim=2)  # y: [bs x nvars x d_model]
            else:
                y = x  # y: [bs x nvars x num_patches x d_model]

        if not self.shared_projection:
            x_out = []
            for i in range(self.num_input_channels):
                z = self.flattens[i](y[:, i, :])  # y: [bs x (d_model * num_patches)] or [bs x d_model)]
                z = self.linears[i](z)  # z: [bs x forecast_len]
                z = self.dropouts[i](z)
                x_out.append(z)
            x = torch.stack(x_out, dim=1)  # x: [bs x nvars x forecast_len]
        else:
            z = self.flatten(y)  # z: [bs x nvars x (d_model * num_patches)] or [bs x nvars x d_model)]
            z = self.dropout(z)
            x = self.linear(z)  # x: [bs x nvars x forecast_len]

        x = x.transpose(2, 1)  # [bs x forecast_len x nvars]

        return x


class PatchTSTForForecasting(PatchTSTPreTrainedModel):
    # PatchTST model + Forecasting head
    def __init__(self, config: PatchTSTConfig):
        super().__init__(config)
        self.model = PatchTSTModel(config)
        self.head = ForecastHead(config)
        self.loss = nn.MSELoss(reduction="mean")

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        past_values: torch.Tensor,
        future_values: Optional[torch.Tensor],
        past_observed_mask: Optional[torch.Tensor] = None,
        future_observed_mask: Optional[torch.Tensor] = None,
        output_hidden_states: Optional[bool] = None,
    ):
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        model_output = self.model(
            past_values, past_observed_mask=past_observed_mask, output_hidden_states=output_hidden_states
        )

        y_hat = self.head(model_output.last_hidden_state)
        y_hat = y_hat * model_output.scale + model_output.loc

        loss_val = None
        if future_values is not None:
            loss_val = self.loss(y_hat, future_values)
        return PatchTSTForForecastingOutput(
            loss=loss_val, forecast_outputs=y_hat, hidden_states=model_output.hidden_states
        )


class RegressionHead(nn.Module):
    def __init__(self, config: PatchTSTConfig):
        super().__init__()
        self.y_range = config.prediction_range
        self.use_cls_token = config.use_cls_token
        self.pooling = config.pooling
        # self.is_flatten = is_flatten

        self.flatten = nn.Flatten(start_dim=1)
        self.dropout = nn.Dropout(config.head_dropout) if config.head_dropout > 0 else nn.Identity()
        head_dim = config.num_input_channels * config.d_model
        # if is_flatten: head_dim *= num_patch
        self.linear = nn.Linear(head_dim, config.num_output_channels)

    def forward(self, past_values):
        """
        x: [bs x nvars x num_patch x d_model]
            or [bs x nvars x (num_patch+1) x d_model] if use cls_token
        output: [bs x output_dim]
        """
        if self.use_cls_token:
            past_values = past_values[:, :, 0, :]  # use the first output token, x: [bs x nvars x d_model]
        elif self.pooling == "mean":
            past_values = past_values.mean(dim=2)  # x: [bs x nvars x d_model]
        elif self.pooling == "max":
            past_values = past_values.max(dim=2)  # x: [bs x nvars x d_model]
        else:
            raise Exception(f"pooling operator {self.pooling} is not implemented yet")
        # flatten the input
        past_values = self.flatten(past_values)  # x: bs x nvars * d_model
        y = self.linear(self.dropout(past_values))  # y: bs x output_dim

        if self.y_range:
            y = torch.sigmoid(y) * (self.y_range[1] - self.y_range[0]) + self.y_range[0]

        return y


class PatchTSTForRegression(PatchTSTPreTrainedModel):
    # PatchTST model + Regression head
    def __init__(self, config: PatchTSTConfig):
        super().__init__(config)
        self.model = PatchTSTModel(config)
        self.head = RegressionHead(config)
        self.loss = nn.MSELoss(reduction="mean")

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self, past_values: torch.Tensor, labels: Optional[torch.Tensor], output_hidden_states: Optional[bool] = None
    ):
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        model_output = self.model(past_values, output_hidden_states=output_hidden_states)
        y_hat = self.head(model_output.last_hidden_state)

        loss_val = None
        if labels is not None:
            loss_val = self.loss(y_hat, labels)
        return PatchTSTOutput(loss=loss_val, prediction_output=y_hat, hidden_states=model_output.hidden_states)
