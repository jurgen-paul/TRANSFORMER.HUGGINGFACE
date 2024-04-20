# coding=utf-8
# Copyright 2024 Intel Labs and The HuggingFace Inc. team. All rights reserved.
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
""" PyTorch ZoeDepth model. """


import math
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn

from ...activations import ACT2FN
from ...file_utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    replace_return_docstrings,
)
from ...modeling_outputs import DepthEstimatorOutput
from ...modeling_utils import PreTrainedModel
from ...utils import ModelOutput, logging
from ...utils.backbone_utils import load_backbone
from .configuration_zoedepth import ZoeDepthConfig


logger = logging.get_logger(__name__)

# General docstring
_CONFIG_FOR_DOC = "ZoeDepthConfig"

# Base docstring
_CHECKPOINT_FOR_DOC = "nielsr/zoedepth-nyu"
_EXPECTED_OUTPUT_SHAPE = [1, 577, 1024]

N_MIDAS_OUT = 32


@dataclass
class ZoeDepthDepthEstimatorOutput(ModelOutput):
    """
    Extension of `DepthEstimatorOutput` to include domain logits (ZoeDepth specific).

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Classification (or regression if config.num_labels==1) loss.
        predicted_depth (`torch.FloatTensor` of shape `(batch_size, height, width)`):
            Predicted depth for each pixel.

        domain_logits (`torch.FloatTensor` of shape `(batch_size, num_domains)`):
            Logits for each domain (e.g. NYU and KITTI) in case multiple metric heads are used.

        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, num_channels, height, width)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, patch_size,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    loss: Optional[torch.FloatTensor] = None
    predicted_depth: torch.FloatTensor = None
    domain_logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None


class ZoeDepthReassembleStage(nn.Module):
    """
    This class reassembles the hidden states of the backbone into image-like feature representations at various
    resolutions.

    This happens in 3 stages:
    1. Map the N + 1 tokens to a set of N tokens, by taking into account the readout ([CLS]) token according to
       `config.readout_type`.
    2. Project the channel dimension of the hidden states according to `config.neck_hidden_sizes`.
    3. Resizing the spatial dimensions (height, width).

    Args:
        config (`[ZoeDepthConfig]`):
            Model configuration class defining the model architecture.
    """

    def __init__(self, config):
        super().__init__()

        self.readout_type = config.readout_type
        self.layers = nn.ModuleList()

        for i, factor in zip(range(len(config.neck_hidden_sizes)), config.reassemble_factors):
            self.layers.append(ZoeDepthReassembleLayer(config, channels=config.neck_hidden_sizes[i], factor=factor))

        if config.readout_type == "project":
            self.readout_projects = nn.ModuleList()
            hidden_size = config.backbone_config.hidden_size
            for _ in range(len(config.neck_hidden_sizes)):
                self.readout_projects.append(
                    nn.Sequential(nn.Linear(2 * hidden_size, hidden_size), ACT2FN[config.hidden_act])
                )

    def forward(self, hidden_states: List[torch.Tensor], patch_height=None, patch_width=None) -> List[torch.Tensor]:
        """
        Args:
            hidden_states (`List[torch.FloatTensor]`, each of shape `(batch_size, sequence_length + 1, hidden_size)`):
                List of hidden states from the backbone.
        """
        out = []

        for i, hidden_state in enumerate(hidden_states):
            # reshape to (batch_size, num_channels, height, width)
            cls_token, hidden_state = hidden_state[:, 0], hidden_state[:, 1:]
            batch_size, sequence_length, num_channels = hidden_state.shape
            if patch_height is not None and patch_width is not None:
                hidden_state = hidden_state.reshape(batch_size, patch_height, patch_width, num_channels)
            else:
                size = int(math.sqrt(sequence_length))
                hidden_state = hidden_state.reshape(batch_size, size, size, num_channels)
            hidden_state = hidden_state.permute(0, 3, 1, 2).contiguous()

            feature_shape = hidden_state.shape
            if self.readout_type == "project":
                # reshape to (batch_size, height*width, num_channels)
                hidden_state = hidden_state.flatten(2).permute((0, 2, 1))
                readout = cls_token.unsqueeze(1).expand_as(hidden_state)
                # concatenate the readout token to the hidden states and project
                hidden_state = self.readout_projects[i](torch.cat((hidden_state, readout), -1))
                # reshape back to (batch_size, num_channels, height, width)
                hidden_state = hidden_state.permute(0, 2, 1).reshape(feature_shape)
            elif self.readout_type == "add":
                hidden_state = hidden_state.flatten(2) + cls_token.unsqueeze(-1)
                hidden_state = hidden_state.reshape(feature_shape)
            hidden_state = self.layers[i](hidden_state)
            out.append(hidden_state)

        return out


class ZoeDepthReassembleLayer(nn.Module):
    def __init__(self, config, channels, factor):
        super().__init__()
        # projection
        hidden_size = config.backbone_config.hidden_size
        self.projection = nn.Conv2d(in_channels=hidden_size, out_channels=channels, kernel_size=1)

        # up/down sampling depending on factor
        if factor > 1:
            self.resize = nn.ConvTranspose2d(channels, channels, kernel_size=factor, stride=factor, padding=0)
        elif factor == 1:
            self.resize = nn.Identity()
        elif factor < 1:
            # so should downsample
            self.resize = nn.Conv2d(channels, channels, kernel_size=3, stride=int(1 / factor), padding=1)

    # Copied from transformers.models.dpt.modeling_dpt.DPTReassembleLayer.forward with DPT->ZoeDepth
    def forward(self, hidden_state):
        hidden_state = self.projection(hidden_state)
        hidden_state = self.resize(hidden_state)
        return hidden_state


# Copied from transformers.models.dpt.modeling_dpt.DPTFeatureFusionStage with DPT->ZoeDepth
class ZoeDepthFeatureFusionStage(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(len(config.neck_hidden_sizes)):
            self.layers.append(ZoeDepthFeatureFusionLayer(config))

    def forward(self, hidden_states):
        # reversing the hidden_states, we start from the last
        hidden_states = hidden_states[::-1]

        fused_hidden_states = []
        # first layer only uses the last hidden_state
        fused_hidden_state = self.layers[0](hidden_states[0])
        fused_hidden_states.append(fused_hidden_state)
        # looping from the last layer to the second
        for hidden_state, layer in zip(hidden_states[1:], self.layers[1:]):
            fused_hidden_state = layer(fused_hidden_state, hidden_state)
            fused_hidden_states.append(fused_hidden_state)

        return fused_hidden_states


# Copied from transformers.models.dpt.modeling_dpt.DPTPreActResidualLayer with DPT->ZoeDepth
class ZoeDepthPreActResidualLayer(nn.Module):
    """
    ResidualConvUnit, pre-activate residual unit.

    Args:
        config (`[ZoeDepthConfig]`):
            Model configuration class defining the model architecture.
    """

    def __init__(self, config):
        super().__init__()

        self.use_batch_norm = config.use_batch_norm_in_fusion_residual
        use_bias_in_fusion_residual = (
            config.use_bias_in_fusion_residual
            if config.use_bias_in_fusion_residual is not None
            else not self.use_batch_norm
        )

        self.activation1 = nn.ReLU()
        self.convolution1 = nn.Conv2d(
            config.fusion_hidden_size,
            config.fusion_hidden_size,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=use_bias_in_fusion_residual,
        )

        self.activation2 = nn.ReLU()
        self.convolution2 = nn.Conv2d(
            config.fusion_hidden_size,
            config.fusion_hidden_size,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=use_bias_in_fusion_residual,
        )

        if self.use_batch_norm:
            self.batch_norm1 = nn.BatchNorm2d(config.fusion_hidden_size)
            self.batch_norm2 = nn.BatchNorm2d(config.fusion_hidden_size)

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        residual = hidden_state
        hidden_state = self.activation1(hidden_state)

        hidden_state = self.convolution1(hidden_state)

        if self.use_batch_norm:
            hidden_state = self.batch_norm1(hidden_state)

        hidden_state = self.activation2(hidden_state)
        hidden_state = self.convolution2(hidden_state)

        if self.use_batch_norm:
            hidden_state = self.batch_norm2(hidden_state)

        return hidden_state + residual


# Copied from transformers.models.dpt.modeling_dpt.DPTFeatureFusionLayer with DPT->ZoeDepth
class ZoeDepthFeatureFusionLayer(nn.Module):
    """Feature fusion layer, merges feature maps from different stages.

    Args:
        config (`[ZoeDepthConfig]`):
            Model configuration class defining the model architecture.
        align_corners (`bool`, *optional*, defaults to `True`):
            The align_corner setting for bilinear upsample.
    """

    def __init__(self, config, align_corners=True):
        super().__init__()

        self.align_corners = align_corners

        self.projection = nn.Conv2d(config.fusion_hidden_size, config.fusion_hidden_size, kernel_size=1, bias=True)

        self.residual_layer1 = ZoeDepthPreActResidualLayer(config)
        self.residual_layer2 = ZoeDepthPreActResidualLayer(config)

    def forward(self, hidden_state, residual=None):
        if residual is not None:
            if hidden_state.shape != residual.shape:
                residual = nn.functional.interpolate(
                    residual, size=(hidden_state.shape[2], hidden_state.shape[3]), mode="bilinear", align_corners=False
                )
            hidden_state = hidden_state + self.residual_layer1(residual)

        hidden_state = self.residual_layer2(hidden_state)
        hidden_state = nn.functional.interpolate(
            hidden_state, scale_factor=2, mode="bilinear", align_corners=self.align_corners
        )
        hidden_state = self.projection(hidden_state)

        return hidden_state


# Copied from transformers.models.dpt.modeling_dpt.DPTPreTrainedModel with DPT->ZoeDepth,dpt->zoedepth
class ZoeDepthPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = ZoeDepthConfig
    base_model_prefix = "zoedepth"
    main_input_name = "pixel_values"
    supports_gradient_checkpointing = True

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


ZOEDEPTH_START_DOCSTRING = r"""
    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it
    as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`ViTConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

ZOEDEPTH_INPUTS_DOCSTRING = r"""
    Args:
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Pixel values. Pixel values can be obtained using [`AutoImageProcessor`]. See [`DPTImageProcessor.__call__`]
            for details.

        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~file_utils.ModelOutput`] instead of a plain tuple.
"""


class ZoeDepthNeck(nn.Module):
    """
    ZoeDepthNeck. A neck is a module that is normally used between the backbone and the head. It takes a list of tensors as
    input and produces another list of tensors as output. For ZoeDepth, it includes 2 stages:

    * ZoeDepthReassembleStage
    * ZoeDepthFeatureFusionStage.

    Args:
        config (dict): config dict.
    """

    # Copied from transformers.models.dpt.modeling_dpt.DPTNeck.__init__ with DPT->ZoeDepth
    def __init__(self, config):
        super().__init__()
        self.config = config

        # postprocessing: only required in case of a non-hierarchical backbone (e.g. ViT, BEiT)
        if config.backbone_config is not None and config.backbone_config.model_type in ["swinv2"]:
            self.reassemble_stage = None
        else:
            self.reassemble_stage = ZoeDepthReassembleStage(config)

        self.convs = nn.ModuleList()
        for channel in config.neck_hidden_sizes:
            self.convs.append(nn.Conv2d(channel, config.fusion_hidden_size, kernel_size=3, padding=1, bias=False))

        # fusion
        self.fusion_stage = ZoeDepthFeatureFusionStage(config)

    def forward(self, hidden_states: List[torch.Tensor], patch_height=None, patch_width=None) -> List[torch.Tensor]:
        """
        Args:
            hidden_states (`List[torch.FloatTensor]`, each of shape `(batch_size, sequence_length, hidden_size)` or `(batch_size, hidden_size, height, width)`):
                List of hidden states from the backbone.
        """
        if not isinstance(hidden_states, (tuple, list)):
            raise ValueError("hidden_states should be a tuple or list of tensors")

        if len(hidden_states) != len(self.config.neck_hidden_sizes):
            raise ValueError("The number of hidden states should be equal to the number of neck hidden sizes.")

        # postprocess hidden states
        if self.reassemble_stage is not None:
            hidden_states = self.reassemble_stage(hidden_states, patch_height, patch_width)

        features = [self.convs[i](feature) for i, feature in enumerate(hidden_states)]

        # fusion blocks
        output = self.fusion_stage(features)

        return output, features[-1]


class ZoeDepthRelativeDepthEstimationHead(nn.Module):
    """
    Relative depth estimation head consisting of 3 convolutional layers. It progressively halves the feature dimension and upsamples
    the predictions to the input resolution after the first convolutional layer (details can be found in DPT's paper's
    supplementary material).
    """

    def __init__(self, config):
        super().__init__()

        self.config = config

        self.projection = None
        if config.add_projection:
            self.projection = nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

        features = config.fusion_hidden_size
        self.conv1 = nn.Conv2d(features, features // 2, kernel_size=3, stride=1, padding=1)
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv2 = nn.Conv2d(features // 2, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0)

    def forward(self, hidden_states: List[torch.Tensor]) -> torch.Tensor:
        # use last features
        hidden_states = hidden_states[self.config.head_in_index]

        if self.projection is not None:
            hidden_states = self.projection(hidden_states)
            hidden_states = nn.ReLU()(hidden_states)

        hidden_states = self.conv1(hidden_states)
        hidden_states = self.upsample(hidden_states)
        hidden_states = self.conv2(hidden_states)
        hidden_states = nn.ReLU()(hidden_states)
        # we need the features here (after second conv + ReLu)
        features = hidden_states
        hidden_states = self.conv3(hidden_states)
        hidden_states = nn.ReLU()(hidden_states)

        predicted_depth = hidden_states.squeeze(dim=1)

        return predicted_depth, features


def log_binom(n, k, eps=1e-7):
    """log(nCk) using stirling approximation"""
    n = n + eps
    k = k + eps
    return n * torch.log(n) - k * torch.log(k) - (n - k) * torch.log(n - k + eps)


class LogBinomial(nn.Module):
    def __init__(self, n_classes=256, act=torch.softmax):
        """Compute log binomial distribution for n_classes

        Args:
            n_classes (`int`, *optional*, defaults to 256):
                Number of output classes.
            act (`torch.nn.Module`, *optional*, defaults to `torch.softmax`):
                Activation function to apply to the output.
        """
        super().__init__()
        self.K = n_classes
        self.act = act
        self.register_buffer("k_idx", torch.arange(0, n_classes).view(1, -1, 1, 1), persistent=False)
        self.register_buffer("K_minus_1", torch.Tensor([self.K - 1]).view(1, -1, 1, 1), persistent=False)

    def forward(self, probabilities, temperature=1.0, eps=1e-4):
        """Compute the log binomial distribution for probabilities.

        Args:
            probabilities (`torch.Tensor` of shape `(batch_size, num_channels, height, width)`):
                Tensor containing probabilities of each class.
            temperature (`float` or `torch.Tensor` of shape `(batch_size, num_channels, height, width)`, *optional*, defaults to 1):
                Temperature of distribution.
            eps (`float`, *optional*, defaults to 1e-4):
                Small number for numerical stability.

        Returns:
            `torch.Tensor` of shape `(batch_size, num_channels, height, width)`:
                Log binomial distribution logbinomial(p;t).
        """
        if probabilities.ndim == 3:
            probabilities = probabilities.unsqueeze(1)  # make it (batch_size, num_channels, height, width)

        one_minus_probabilities = torch.clamp(1 - probabilities, eps, 1)
        probabilities = torch.clamp(probabilities, eps, 1)
        y = (
            log_binom(self.K_minus_1, self.k_idx)
            + self.k_idx * torch.log(probabilities)
            + (self.K - 1 - self.k_idx) * torch.log(one_minus_probabilities)
        )
        return self.act(y / temperature, dim=1)


class ZoeDepthConditionalLogBinomial(nn.Module):
    def __init__(
        self,
        in_features,
        condition_dim,
        n_classes=256,
        bottleneck_factor=2,
        p_eps=1e-4,
        max_temp=50,
        min_temp=1e-7,
        act=torch.softmax,
    ):
        """Conditional Log Binomial distribution.

        Args:
            in_features (`int`):
                Number of input channels in the main feature.
            condition_dim (`int`):
                Number of input channels in the condition feature.
            n_classes (`int`, *optional*, defaults to 256):
                Number of classes.
            bottleneck_factor (`int`, *optional*, defaults to 2):
                Hidden dim factor.
            p_eps (`float`, *optional*, defaults to 1e-4):
                Small eps value.
            max_temp (`float`, *optional*, defaults to 50):
                Maximum temperature of output distribution.
            min_temp (`float`, *optional*, defaults to 1e-7):
                Minimum temperature of output distribution.
            act (`torch.nn.Module`, *optional*, defaults to `torch.softmax`):
                Activation function to apply to the output.

        """
        super().__init__()
        self.p_eps = p_eps
        self.max_temp = max_temp
        self.min_temp = min_temp
        self.log_binomial_transform = LogBinomial(n_classes, act=act)
        bottleneck = (in_features + condition_dim) // bottleneck_factor
        self.mlp = nn.Sequential(
            nn.Conv2d(in_features + condition_dim, bottleneck, kernel_size=1, stride=1, padding=0),
            nn.GELU(),
            # 2 for probabilities linear norm, 2 for temperature linear norm
            nn.Conv2d(bottleneck, 2 + 2, kernel_size=1, stride=1, padding=0),
            nn.Softplus(),
        )

    def forward(self, main_feature, condition_feature):
        """
        Args:
            main_feature (`torch.Tensor` of shape `(batch_size, num_channels, height, width)`):
                Main feature.
            condition_feature (torch.Tensor of shape `(batch_size, num_channels, height, width)`):
                Condition feature.

        Returns:
            `torch.Tensor`:
                Output log binomial distribution
        """
        pt = self.mlp(torch.concat((main_feature, condition_feature), dim=1))
        probabilities, temperature = pt[:, :2, ...], pt[:, 2:, ...]

        probabilities = probabilities + self.p_eps
        probabilities = probabilities[:, 0, ...] / (probabilities[:, 0, ...] + probabilities[:, 1, ...])

        temperature = temperature + self.p_eps
        temperature = temperature[:, 0, ...] / (temperature[:, 0, ...] + temperature[:, 1, ...])
        temperature = temperature.unsqueeze(1)
        temperature = (self.max_temp - self.min_temp) * temperature + self.min_temp

        return self.log_binomial_transform(probabilities, temperature)


class ZoeDepthSeedBinRegressor(nn.Module):
    def __init__(self, in_features, n_bins=16, mlp_dim=256, min_depth=1e-3, max_depth=10):
        """Bin center regressor network. Bin centers are bounded on (min_depth, max_depth) interval.

        Args:
            in_features (`int`):
                Number of input channels.
            n_bins (`int`, *optional*, defaults to 16):
                Number of bin centers.
            mlp_dim (`int`, *optional*, defaults to 256):
                Hidden dimension.
            min_depth (`float`, *optional*, defaults to 1e-3):
                Min depth value.
            max_depth (`float`, *optional*, defaults to 10):
                Max depth value.
        """
        super().__init__()
        self.min_depth = min_depth
        self.max_depth = max_depth

        self.conv1 = nn.Conv2d(in_features, mlp_dim, 1, 1, 0)
        self.act1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(mlp_dim, n_bins, 1, 1, 0)
        self.act2 = nn.ReLU(inplace=True)

    def forward(self, x):
        """
        Returns tensor of bin_width vectors (centers). One vector b for every pixel
        """
        x = self.conv1(x)
        x = self.act1(x)
        x = self.conv2(x)
        bins = self.act2(x)

        eps = 1e-3
        bins = bins + eps
        bin_widths_normed = bins / bins.sum(dim=1, keepdim=True)
        bin_widths = (
            self.max_depth - self.min_depth
        ) * bin_widths_normed  # shape (batch_size, num_channels, height, width)
        # pad has the form (left, right, top, bottom, front, back)
        bin_widths = nn.functional.pad(bin_widths, (0, 0, 0, 0, 1, 0), mode="constant", value=self.min_depth)
        bin_edges = torch.cumsum(bin_widths, dim=1)  # shape (batch_size, num_channels, height, width)

        bin_centers = 0.5 * (bin_edges[:, :-1, ...] + bin_edges[:, 1:, ...])
        return bin_widths_normed, bin_centers


class ZoeDepthSeedBinRegressorUnnormed(nn.Module):
    def __init__(self, in_features, n_bins=16, mlp_dim=256, min_depth=1e-3, max_depth=10):
        """Bin center regressor network. Bin centers are unbounded

        Args:
            in_features (`int`):
                Number of input channels.
            n_bins (`int`, *optional*, defaults to 16):
                Number of bin centers.
            mlp_dim (`int`, *optional*, defaults to 256):
                Hidden dimension.
            min_depth (`float`, *optional*):
                Not used. (for compatibility with SeedBinRegressor)
            max_depth (`float`, *optional*):
                Not used. (for compatibility with SeedBinRegressor)
        """
        super().__init__()
        self.conv1 = nn.Conv2d(in_features, mlp_dim, 1, 1, 0)
        self.act1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(mlp_dim, n_bins, 1, 1, 0)
        self.act2 = nn.Softplus()

    def forward(self, x):
        """
        Returns tensor of bin_width vectors (centers). One vector b for every pixel
        """
        x = self.conv1(x)
        x = self.act1(x)
        x = self.conv2(x)
        bin_centers = self.act2(x)

        return bin_centers, bin_centers


@torch.jit.script
def inv_attractor(dx, alpha: float = 300, gamma: int = 2):
    """Inverse attractor: dc = dx / (1 + alpha*dx^gamma), where dx = a - c, a = attractor point, c = bin center, dc = shift in bin center
    This is the default one according to the accompanying paper.

    Args:
        dx (`torch.Tensor`):
            The difference tensor dx = Ai - Cj, where Ai is the attractor point and Cj is the bin center.
        alpha (`float`, *optional*, defaults to 300):
            Proportional Attractor strength. Determines the absolute strength. Lower alpha = greater attraction.
        gamma (`int`, *optional*, defaults to 2):
            Exponential Attractor strength. Determines the "region of influence" and indirectly number of bin centers affected.
            Lower gamma = farther reach.

    Returns:
        torch.Tensor: Delta shifts - dc; New bin centers = Old bin centers + dc
    """
    return dx.div(1 + alpha * dx.pow(gamma))


class ZoeDepthAttractorLayer(nn.Module):
    def __init__(
        self,
        in_features,
        n_bins,
        n_attractors=16,
        mlp_dim=128,
        min_depth=1e-3,
        max_depth=10,
        alpha=300,
        gamma=2,
        kind="sum",
        memory_efficient=False,
    ):
        """
        Attractor layer for bin centers. Bin centers are bounded on the interval (min_depth, max_depth)
        """
        super().__init__()

        self.n_attractors = n_attractors
        self.n_bins = n_bins
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.alpha = alpha
        self.gamma = gamma
        self.kind = kind
        self.memory_efficient = memory_efficient

        # MLP to predict attractor points
        self.conv1 = nn.Conv2d(in_features, mlp_dim, 1, 1, 0)
        self.act1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(mlp_dim, n_attractors * 2, 1, 1, 0)  # x2 for linear norm
        self.act2 = nn.ReLU(inplace=True)

    def forward(self, x, prev_bin, prev_bin_embedding=None, interpolate=True):
        """
        The forward pass of the attractor layer. This layer predicts the new bin centers based on the previous bin centers
        and the attractor points (the latter are predicted by the MLP).

        Args:
            x (`torch.Tensor` of shape `(batch_size, num_channels, height, width)`):
                Feature block.
            prev_bin (`torch.Tensor` of shape `(batch_size, prev_number_of_bins, height, width)`):
                Previous bin centers normed.
            prev_bin_embedding (`torch.Tensor`, *optional*):
                Optional previous bin embeddings.
            interpolate (`bool`, *optional*, defaults to `True`):
                Whether to interpolate the previous bin embeddings to the size of the input features.

        Returns:
            `Tuple[`torch.Tensor`, `torch.Tensor`]:
                New bin centers normed and scaled.
        """
        if prev_bin_embedding is not None:
            if interpolate:
                prev_bin_embedding = nn.functional.interpolate(
                    prev_bin_embedding, x.shape[-2:], mode="bilinear", align_corners=True
                )
            x = x + prev_bin_embedding

        x = self.conv1(x)
        x = self.act1(x)
        x = self.conv2(x)
        attractors = self.act2(x)

        eps = 1e-3
        attractors = attractors + eps
        batch_size, _, height, width = attractors.shape
        attractors = attractors.view(batch_size, self.n_attractors, 2, height, width)
        attractors_normed = attractors / attractors.sum(
            dim=2, keepdim=True
        )  # batch_size, num_attractors, 2, height, width
        attractors_normed = attractors[:, :, 0, ...]  # batch_size, batch_size*num_attractors, height, width

        prev_bin = nn.functional.interpolate(prev_bin, (height, width), mode="bilinear", align_corners=True)
        bin_centers = prev_bin

        # note: only attractor_type = "exp" is supported here, since no checkpoints were released with other attractor types
        distribution = inv_attractor

        if not self.memory_efficient:
            func = {"mean": torch.mean, "sum": torch.sum}[self.kind]
            # shape (batch_size, num_bins, height, width)
            delta_c = func(distribution(attractors_normed.unsqueeze(2) - bin_centers.unsqueeze(1)), dim=1)
        else:
            delta_c = torch.zeros_like(bin_centers, device=bin_centers.device)
            for i in range(self.n_attractors):
                # shape (batch_size, num_bins, height, width)
                delta_c += distribution(attractors_normed[:, i, ...].unsqueeze(1) - bin_centers)

            if self.kind == "mean":
                delta_c = delta_c / self.n_attractors

        bin_new_centers = bin_centers + delta_c
        bin_centers = (self.max_depth - self.min_depth) * bin_new_centers + self.min_depth
        bin_centers, _ = torch.sort(bin_centers, dim=1)
        bin_centers = torch.clip(bin_centers, self.min_depth, self.max_depth)
        return bin_new_centers, bin_centers


class ZoeDepthAttractorLayerUnnormed(nn.Module):
    def __init__(
        self,
        in_features,
        n_bins,
        n_attractors=16,
        mlp_dim=128,
        min_depth=1e-3,
        max_depth=10,
        alpha=300,
        gamma=2,
        kind="sum",
        memory_efficient=True,
    ):
        """
        Attractor layer for bin centers. Bin centers are unbounded
        """
        super().__init__()

        self.n_attractors = n_attractors
        self.n_bins = n_bins
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.alpha = alpha
        self.gamma = gamma
        self.kind = kind
        self.memory_efficient = memory_efficient

        self.conv1 = nn.Conv2d(in_features, mlp_dim, 1, 1, 0)
        self.act1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(mlp_dim, n_attractors, 1, 1, 0)
        self.act2 = nn.Softplus()

    def forward(self, x, prev_bin, prev_bin_embedding=None, interpolate=True):
        """
        Args:
            x (`torch.Tensor` of shape (batch_size, num_channels, height, width)`):
                Feature block.
            prev_bin (`torch.Tensor` of shape (batch_size, prev_num_bins, height, width)`):
                Previous bin centers normed.
            prev_bin_embedding (`torch.Tensor`, *optional*):
                Optional previous bin embeddings.
            interpolate (`bool`, *optional*, defaults to `True`):
                Whether to interpolate the previous bin embeddings to the size of the input features.

        Returns:
            `Tuple[`torch.Tensor`, `torch.Tensor`]:
                New bin centers unbounded. Two outputs just to keep the API consistent with the normed version.
        """
        if prev_bin_embedding is not None:
            if interpolate:
                prev_bin_embedding = nn.functional.interpolate(
                    prev_bin_embedding, x.shape[-2:], mode="bilinear", align_corners=True
                )
            x = x + prev_bin_embedding

        x = self.conv1(x)
        x = self.act1(x)
        x = self.conv2(x)
        attractors = self.act2(x)

        height, width = attractors.shape[-2:]

        prev_bin = nn.functional.interpolate(prev_bin, (height, width), mode="bilinear", align_corners=True)
        bin_centers = prev_bin

        dist = inv_attractor

        if not self.memory_efficient:
            func = {"mean": torch.mean, "sum": torch.sum}[self.kind]
            # shape batch_size, num_bins, height, width
            delta_c = func(dist(attractors.unsqueeze(2) - bin_centers.unsqueeze(1)), dim=1)
        else:
            delta_c = torch.zeros_like(bin_centers, device=bin_centers.device)
            for i in range(self.n_attractors):
                # shape batch_size, num_bins, height, width
                delta_c += dist(attractors[:, i, ...].unsqueeze(1) - bin_centers)

            if self.kind == "mean":
                delta_c = delta_c / self.n_attractors

        bin_new_centers = bin_centers + delta_c
        bin_centers = bin_new_centers

        return bin_new_centers, bin_centers


class ZoeDepthProjector(nn.Module):
    def __init__(self, in_features, out_features, mlp_dim=128):
        """Projector MLP.

        Args:
            in_features (`int`):
                Number of input channels.
            out_features (`int`):
                Number of output channels.
            mlp_dim (`int`, *optional*, defaults to 128):
                Hidden dimension.
        """
        super().__init__()

        self.conv1 = nn.Conv2d(in_features, mlp_dim, 1, 1, 0)
        self.act = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(mlp_dim, out_features, 1, 1, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.act(x)
        x = self.conv2(x)

        return x


class ZoeDepthPatchTransformerEncoder(nn.Module):
    def __init__(self, in_channels, patch_size=10, embedding_dim=128, num_heads=4, use_class_token=False):
        """ViT-like transformer block

        Args:
            in_channels (`int`):
                Input channels.
            patch_size (`int`, *optional*, defaults to 10):
                Patch size.
            embedding_dim (`int`, *optional*, defaults to 128):
                Embedding dimension in transformer model.
            num_heads (`int`, *optional*, defaults to 4):
                Number of attention heads.
            use_class_token (`bool`, *optional*, defaults to `False`):
                Whether to use extra token at the start for global accumulation (called as "class token").
        """
        super().__init__()
        encoder_layers = nn.TransformerEncoderLayer(embedding_dim, num_heads, dim_feedforward=1024, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=4)
        self.use_class_token = use_class_token

        self.embedding_convPxP = nn.Conv2d(
            in_channels, embedding_dim, kernel_size=patch_size, stride=patch_size, padding=0
        )

    def positional_encoding_1d(self, batch_size, sequence_length, embedding_dim, device="cpu"):
        """Generate positional encodings

        Args:
            sequence_length (int): Sequence length
            embedding_dim (int): Embedding dimension

        Returns:
            torch.Tensor: Positional encodings.
        """
        position = torch.arange(0, sequence_length, dtype=torch.float32, device=device).unsqueeze(1)
        index = torch.arange(0, embedding_dim, 2, dtype=torch.float32, device=device).unsqueeze(0)
        div_term = torch.exp(index * (-torch.log(torch.tensor(10000.0, device=device)) / embedding_dim))
        pos_encoding = position * div_term
        pos_encoding = torch.cat([torch.sin(pos_encoding), torch.cos(pos_encoding)], dim=1)
        pos_encoding = pos_encoding.unsqueeze(dim=0).repeat(batch_size, 1, 1)
        return pos_encoding

    def forward(self, x):
        """Forward pass

        Args:
            x (torch.Tensor - NCHW): Input feature tensor

        Returns:
            torch.Tensor - Transformer output embeddings of shape (batch_size, sequence_length, embedding_dim)
        """
        embeddings = self.embedding_convPxP(x).flatten(2)  # .shape = n,c,s = n, embedding_dim, s
        if self.use_class_token:
            # extra special token at start ?
            embeddings = nn.functional.pad(embeddings, (1, 0))

        embeddings = embeddings.permute(0, 2, 1)
        batch_size, sequence_length, embedding_dim = embeddings.shape
        embeddings = embeddings + self.positional_encoding_1d(
            batch_size, sequence_length, embedding_dim, device=embeddings.device
        )
        x = self.transformer_encoder(embeddings)
        return x


class ZoeDepthMultipleMetricDepthEstimationHeads(nn.Module):
    """
    Multiple metric depth estimation heads. A MLP classifier is used to route between 2 different heads.
    """

    def __init__(self, config):
        super().__init__()

        bin_embedding_dim = config.bin_embedding_dim
        min_depth = config.min_depth
        max_depth = config.max_depth
        n_attractors = config.num_attractors
        attractor_alpha = config.attractor_alpha
        attractor_gamma = config.attractor_gamma
        attractor_kind = config.attractor_kind
        bin_centers_type = config.bin_centers_type
        min_temp = config.min_temp
        max_temp = config.max_temp
        bin_configurations = config.bin_configurations

        self.min_depth = min_depth
        self.max_depth = max_depth
        self.bin_centers_type = bin_centers_type
        self.min_temp = min_temp
        self.max_temp = max_temp
        self.bin_configurations = bin_configurations

        # Bottleneck convolution
        bottleneck_features = config.bottleneck_features
        self.conv2 = nn.Conv2d(bottleneck_features, bottleneck_features, kernel_size=1, stride=1, padding=0)

        # Transformer classifier on the bottleneck
        self.patch_transformer = ZoeDepthPatchTransformerEncoder(bottleneck_features, 1, 128, use_class_token=True)
        self.mlp_classifier = nn.Sequential(nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 2))

        # Regressor and attractor
        if self.bin_centers_type == "normed":
            SeedBinRegressorLayer = ZoeDepthSeedBinRegressor
            Attractor = ZoeDepthAttractorLayer
        elif self.bin_centers_type == "softplus":
            SeedBinRegressorLayer = ZoeDepthSeedBinRegressorUnnormed
            Attractor = ZoeDepthAttractorLayerUnnormed

        self.bin_centers_type = bin_centers_type
        # We have bins for each bin configuration
        # Create a map (ModuleDict) of 'name' -> seed_bin_regressor
        self.seed_bin_regressors = nn.ModuleDict(
            {
                conf["name"]: SeedBinRegressorLayer(
                    bottleneck_features,
                    conf["n_bins"],
                    mlp_dim=bin_embedding_dim // 2,
                    min_depth=conf["min_depth"],
                    max_depth=conf["max_depth"],
                )
                for conf in config.bin_configurations
            }
        )

        self.seed_projector = ZoeDepthProjector(bottleneck_features, bin_embedding_dim, mlp_dim=bin_embedding_dim // 2)
        self.projectors = nn.ModuleList(
            [
                ZoeDepthProjector(config.fusion_hidden_size, bin_embedding_dim, mlp_dim=bin_embedding_dim // 2)
                for i in range(4)
            ]
        )

        # Create a map (ModuleDict) of 'name' -> attractors (ModuleList)
        self.attractors = nn.ModuleDict(
            {
                configuration["name"]: nn.ModuleList(
                    [
                        Attractor(
                            bin_embedding_dim,
                            n_attractors[i],
                            mlp_dim=bin_embedding_dim,
                            alpha=attractor_alpha,
                            gamma=attractor_gamma,
                            kind=attractor_kind,
                            min_depth=configuration["min_depth"],
                            max_depth=configuration["max_depth"],
                        )
                        for i in range(len(n_attractors))
                    ]
                )
                for configuration in config.bin_configurations
            }
        )

        last_in = N_MIDAS_OUT
        # conditional log binomial for each bin configuration
        self.conditional_log_binomial = nn.ModuleDict(
            {
                configuration["name"]: ZoeDepthConditionalLogBinomial(
                    last_in,
                    bin_embedding_dim,
                    configuration["n_bins"],
                    bottleneck_factor=4,
                    min_temp=self.min_temp,
                    max_temp=self.max_temp,
                )
                for configuration in config.bin_configurations
            }
        )

    def forward(self, out, rel_depth):
        outconv_activation = out[0]
        bottleneck = out[1]
        feature_blocks = out[2:]

        x_d0 = self.conv2(bottleneck)
        x = x_d0

        # Predict which path to take
        embedding = self.patch_transformer(x)[:, 0, :]  # batch_size, hidden_size
        domain_logits = self.mlp_classifier(embedding)  # batch_size, 2
        domain_vote = torch.softmax(domain_logits.sum(dim=0, keepdim=True), dim=-1)  # 1, 2

        # Get the path
        names = [configuration["name"] for configuration in self.bin_configurations]
        bin_configurations_name = names[torch.argmax(domain_vote, dim=-1).squeeze().item()]

        try:
            conf = [config for config in self.bin_configurations if config["name"] == bin_configurations_name][0]
        except IndexError:
            raise ValueError(f"bin_configurations_name {bin_configurations_name} not found in bin_configurationss")

        min_depth = conf["min_depth"]
        max_depth = conf["max_depth"]

        seed_bin_regressor = self.seed_bin_regressors[bin_configurations_name]
        _, seed_bin_centers = seed_bin_regressor(x)
        if self.bin_centers_type == "normed" or self.bin_centers_type == "hybrid2":
            prev_bin = (seed_bin_centers - min_depth) / (max_depth - min_depth)
        else:
            prev_bin = seed_bin_centers
        prev_bin_embedding = self.seed_projector(x)

        attractors = self.attractors[bin_configurations_name]
        for projector, attractor, feature in zip(self.projectors, attractors, feature_blocks):
            bin_embedding = projector(feature)
            bin, bin_centers = attractor(bin_embedding, prev_bin, prev_bin_embedding, interpolate=True)
            prev_bin = bin
            prev_bin_embedding = bin_embedding

        last = outconv_activation

        bin_centers = nn.functional.interpolate(bin_centers, last.shape[-2:], mode="bilinear", align_corners=True)
        bin_embedding = nn.functional.interpolate(bin_embedding, last.shape[-2:], mode="bilinear", align_corners=True)

        conditional_log_binomial = self.conditional_log_binomial[bin_configurations_name]
        x = conditional_log_binomial(last, bin_embedding)

        # Now depth value is Sum px * cx , where cx are bin_centers from the last bin tensor
        out = torch.sum(x * bin_centers, dim=1, keepdim=True)

        return out, domain_logits


class ZoeDepthMetricDepthEstimationHead(nn.Module):
    def __init__(self, config):
        super().__init__()

        n_bins = config.n_bins
        bin_embedding_dim = config.bin_embedding_dim
        min_depth = config.min_depth
        max_depth = config.max_depth
        n_attractors = config.num_attractors
        attractor_alpha = config.attractor_alpha
        attractor_gamma = config.attractor_gamma
        attractor_kind = config.attractor_kind
        min_temp = config.min_temp
        max_temp = config.max_temp
        bin_centers_type = config.bin_centers_type

        self.min_depth = min_depth
        self.max_depth = max_depth
        self.bin_centers_type = bin_centers_type

        # Bottleneck convolution
        bottleneck_features = config.bottleneck_features
        self.conv2 = nn.Conv2d(bottleneck_features, bottleneck_features, kernel_size=1, stride=1, padding=0)

        # Regressor and attractor
        if self.bin_centers_type == "normed":
            SeedBinRegressorLayer = ZoeDepthSeedBinRegressor
            Attractor = ZoeDepthAttractorLayer
        elif self.bin_centers_type == "softplus":
            SeedBinRegressorLayer = ZoeDepthSeedBinRegressorUnnormed
            Attractor = ZoeDepthAttractorLayerUnnormed

        self.seed_bin_regressor = SeedBinRegressorLayer(
            bottleneck_features, n_bins=n_bins, min_depth=min_depth, max_depth=max_depth
        )
        self.seed_projector = ZoeDepthProjector(bottleneck_features, bin_embedding_dim)

        self.projectors = nn.ModuleList(
            [ZoeDepthProjector(config.fusion_hidden_size, bin_embedding_dim) for _ in range(4)]
        )
        self.attractors = nn.ModuleList(
            [
                Attractor(
                    bin_embedding_dim,
                    n_bins,
                    n_attractors=n_attractors[i],
                    min_depth=min_depth,
                    max_depth=max_depth,
                    alpha=attractor_alpha,
                    gamma=attractor_gamma,
                    kind=attractor_kind,
                )
                for i in range(4)
            ]
        )

        last_in = N_MIDAS_OUT + 1  # +1 for relative depth

        # use log binomial instead of softmax
        self.conditional_log_binomial = ZoeDepthConditionalLogBinomial(
            last_in, bin_embedding_dim, n_classes=n_bins, min_temp=min_temp, max_temp=max_temp
        )

    def forward(self, out, rel_depth):
        outconv_activation = out[0]
        bottleneck = out[1]
        feature_blocks = out[2:]

        x_d0 = self.conv2(bottleneck)
        x = x_d0
        _, seed_bin_centers = self.seed_bin_regressor(x)

        if self.bin_centers_type == "normed" or self.bin_centers_type == "hybrid2":
            prev_bin = (seed_bin_centers - self.min_depth) / (self.max_depth - self.min_depth)
        else:
            prev_bin = seed_bin_centers

        prev_bin_embedding = self.seed_projector(x)

        # unroll this loop for better performance
        for projector, attractor, feature in zip(self.projectors, self.attractors, feature_blocks):
            bin_embedding = projector(feature)
            bin, bin_centers = attractor(bin_embedding, prev_bin, prev_bin_embedding, interpolate=True)
            prev_bin = bin.clone()
            prev_bin_embedding = bin_embedding.clone()

        last = outconv_activation

        # concatenative relative depth with last. First interpolate relative depth to last size
        relative_conditioning = rel_depth.unsqueeze(1)
        relative_conditioning = nn.functional.interpolate(
            relative_conditioning, size=last.shape[2:], mode="bilinear", align_corners=True
        )
        last = torch.cat([last, relative_conditioning], dim=1)

        bin_embedding = nn.functional.interpolate(bin_embedding, last.shape[-2:], mode="bilinear", align_corners=True)
        x = self.conditional_log_binomial(last, bin_embedding)

        # Now depth value is Sum px * cx , where cx are bin_centers from the last bin tensor
        bin_centers = nn.functional.interpolate(bin_centers, x.shape[-2:], mode="bilinear", align_corners=True)
        out = torch.sum(x * bin_centers, dim=1, keepdim=True)

        return out, None


@add_start_docstrings(
    """
    ZoeDepth model with one or multiple metric depth estimation head(s) on top.
    """,
    ZOEDEPTH_START_DOCSTRING,
)
class ZoeDepthForDepthEstimation(ZoeDepthPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.backbone = load_backbone(config)
        self.neck = ZoeDepthNeck(config)
        self.relative_head = ZoeDepthRelativeDepthEstimationHead(config)

        self.metric_head = (
            ZoeDepthMultipleMetricDepthEstimationHeads(config)
            if config.use_multiple_heads
            else ZoeDepthMetricDepthEstimationHead(config)
        )

        # Initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings_to_model_forward(ZOEDEPTH_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=DepthEstimatorOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        pixel_values: torch.FloatTensor,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], DepthEstimatorOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, height, width)`, *optional*):
            Ground truth depth estimation maps for computing the loss.

        Returns:

        Examples:
        ```python
        >>> from transformers import AutoImageProcessor, ZoeDepthForDepthEstimation
        >>> import torch
        >>> import numpy as np
        >>> from PIL import Image
        >>> import requests

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> image_processor = AutoImageProcessor.from_pretrained("nielsr/zoedepth-nyu")
        >>> model = ZoeDepthForDepthEstimation.from_pretrained("nielsr/zoedepth-nyu")

        >>> # prepare image for the model
        >>> inputs = image_processor(images=image, return_tensors="pt")

        >>> with torch.no_grad():
        ...     outputs = model(**inputs)
        ...     predicted_depth = outputs.predicted_depth

        >>> # interpolate to original size
        >>> prediction = torch.nn.functional.interpolate(
        ...     predicted_depth.unsqueeze(1),
        ...     size=image.size[::-1],
        ...     mode="bicubic",
        ...     align_corners=False,
        ... )

        >>> # visualize the prediction
        >>> output = prediction.squeeze().cpu().numpy()
        >>> formatted = (output * 255 / np.max(output)).astype("uint8")
        >>> depth = Image.fromarray(formatted)
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions

        outputs = self.backbone.forward_with_filtered_kwargs(
            pixel_values, output_hidden_states=output_hidden_states, output_attentions=output_attentions
        )
        hidden_states = outputs.feature_maps

        _, _, height, width = pixel_values.shape
        patch_size = self.config.backbone_config.patch_size
        patch_height = height // patch_size
        patch_width = width // patch_size

        hidden_states, features = self.neck(hidden_states, patch_height, patch_width)

        out = [features] + hidden_states

        relative_depth, features = self.relative_head(hidden_states)

        out = [features] + out

        metric_depth, domain_logits = self.metric_head(out, relative_depth)
        metric_depth = metric_depth.squeeze(dim=1)

        loss = None
        if labels is not None:
            raise NotImplementedError("Training is not implemented yet")

        if not return_dict:
            output = (
                (
                    metric_depth,
                    domain_logits,
                )
                + outputs[1:]
                if domain_logits is not None
                else (metric_depth,) + outputs[1:]
            )
            return ((loss,) + output) if loss is not None else output

        return ZoeDepthDepthEstimatorOutput(
            loss=loss,
            predicted_depth=metric_depth,
            domain_logits=domain_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
