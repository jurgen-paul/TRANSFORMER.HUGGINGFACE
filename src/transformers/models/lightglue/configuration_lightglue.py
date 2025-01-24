#                🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨
#           This file was automatically generated from src/transformers/models/lightglue/modular_lightglue.py.
#               Do NOT edit this file manually as any edits will be overwritten by the generation of
#             the file from the modular. If any change should be done, please apply the change to the
#                          modular_lightglue.py file directly. One of our CI enforces this.
#                🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨
from typing import TYPE_CHECKING

from ...configuration_utils import PretrainedConfig
from ..auto import CONFIG_MAPPING


if TYPE_CHECKING:
    from ..superpoint import SuperPointConfig


class LightGlueConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`LightGlueForKeypointMatching`]. It is used to
    instantiate a LightGlue model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the LightGlue
    [stevenbucaille/lightglue_superpoint](https://huggingface.co/stevenbucaille/lightglue_superpoint) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        keypoint_detector_config (`Union[AutoConfig, dict]`,  *optional*, defaults to `SuperPointConfig`):
            The config object or dictionary of the keypoint detector.
        descriptor_dim (`int`, *optional*, defaults to 256):
            The dimension of the descriptors.
        num_hidden_layers (`int`, *optional*, defaults to 9):
            The number of self and cross attention layers.
        num_attention_heads (`int`, *optional*, defaults to 4):
            The number of heads in the multi-head attention.
        num_key_value_heads (`int`, *optional*):
            This is the number of key_value heads that should be used to implement Grouped Query Attention. If
            `num_key_value_heads=num_attention_heads`, the model will use Multi Head Attention (MHA), if
            `num_key_value_heads=1` the model will use Multi Query Attention (MQA) otherwise GQA is used. When
            converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed
            by meanpooling all the original heads within that group. For more details checkout [this
            paper](https://arxiv.org/pdf/2305.13245.pdf). If it is not specified, will default to
            `num_attention_heads`.
        depth_confidence (`float`, *optional*, defaults to 0.95):
            The confidence threshold used to perform early stopping
        width_confidence (`float`, *optional*, defaults to 0.99):
            The confidence threshold used to prune points
        filter_threshold (`float`, *optional*, defaults to 0.1):
            The confidence threshold used to filter matches
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        hidden_act (`str`, *optional*, defaults to `"gelu"`):
            The activation function to be used in the hidden layers.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        attention_bias (`bool`, *optional*, defaults to `True`):
            Whether to use a bias in the query, key, value and output projection layers during self-attention.

    Examples:
        ```python
        >>> from transformers import LightGlueConfig, LightGlueForKeypointMatching

        >>> # Initializing a LightGlue style configuration
        >>> configuration = LightGlueConfig()

        >>> # Initializing a model from the LightGlue style configuration
        >>> model = LightGlueForKeypointMatching(configuration)

        >>> # Accessing the model configuration
        >>> configuration = model.config
        ```
    """

    model_type = "lightglue"

    def __init__(
        self,
        keypoint_detector_config: "SuperPointConfig" = None,
        descriptor_dim: int = 256,
        num_hidden_layers: int = 9,
        num_attention_heads: int = 4,
        num_key_value_heads=None,
        depth_confidence: float = 0.95,
        width_confidence: float = 0.99,
        filter_threshold: float = 0.1,
        initializer_range: float = 0.02,
        hidden_act: str = "gelu",
        attention_dropout=0.0,
        attention_bias=True,
        **kwargs,
    ):
        if descriptor_dim % num_attention_heads != 0:
            raise ValueError("descriptor_dim % num_heads is different from zero")

        self.descriptor_dim = descriptor_dim
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads

        # for backward compatibility
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.num_key_value_heads = num_key_value_heads

        self.depth_confidence = depth_confidence
        self.width_confidence = width_confidence
        self.filter_threshold = filter_threshold
        self.initializer_range = initializer_range

        if isinstance(keypoint_detector_config, dict):
            keypoint_detector_config["model_type"] = (
                keypoint_detector_config["model_type"] if "model_type" in keypoint_detector_config else "superpoint"
            )
            keypoint_detector_config = CONFIG_MAPPING[keypoint_detector_config["model_type"]](
                **keypoint_detector_config
            )
        if keypoint_detector_config is None:
            keypoint_detector_config = CONFIG_MAPPING["superpoint"]()

        self.keypoint_detector_config = keypoint_detector_config

        self.hidden_size = descriptor_dim
        self.hidden_act = hidden_act
        self.attention_dropout = attention_dropout
        self.attention_bias = attention_bias
        super().__init__(**kwargs)


__all__ = ["LightGlueConfig"]
