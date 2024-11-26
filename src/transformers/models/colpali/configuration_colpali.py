# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team.
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
"""ColPali model configuration"""

from copy import deepcopy

from ...modeling_utils import PretrainedConfig
from ..auto import CONFIG_MAPPING


class ColPaliConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`ColPaliForRetrieval`]. It is used to instantiate an
    ColPaliForRetrieval according to the specified arguments, defining the model architecture following the methodology from
    the "ColPali: Efficient Document Retrieval with Vision Language Models" paper.

    Instantiating a configuration with the defaults will yield the same configuration used in the ColPali paper, i.e. the one
    from [vidore/colpali-v1.2](https://huggingface.co/vidore/colpali-v1.2).

    The ColPali config is very similar to [`PaligemmaConfig`], but with an extra attribute defining the embedding dimension.

    Note that contrarily to what the class name suggests (actually the name refers to the ColPali **methodology**), you can
    use a different VLM backbone model than PaliGemma by passing the corresponding VLM configuration to the class constructor.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        vlm_backbone_config (`PretrainedConfig`, *optional*):
            Configuration of the VLM backbone model.
        embedding_dim (`int`, *optional*, defaults to 128):
            Dimension of the multi-vector embeddings produced by the model.

    Example:

    ```python
    from transformers.models.colpali import ColPaliConfig, ColPaliForRetrieval

    config = ColPaliConfig()
    model = ColPaliForRetrieval(config)
    ```
    """

    def __init__(
        self,
        vlm_backbone_config=None,
        embedding_dim: int = 128,
        **kwargs,
    ):
        if isinstance(vlm_backbone_config, dict):
            vlm_backbone_config = deepcopy(vlm_backbone_config)
            vlm_backbone_config["model_type"] = (
                vlm_backbone_config["model_type"] if "model_type" in vlm_backbone_config else "paligemma"
            )
            vlm_backbone_config = CONFIG_MAPPING[vlm_backbone_config["model_type"]](**vlm_backbone_config)
        elif vlm_backbone_config is None:
            vlm_backbone_config = CONFIG_MAPPING["paligemma"]()
        self.vlm_backbone_config = vlm_backbone_config
        self.embedding_dim = embedding_dim

        super().__init__(**kwargs)

    def ignore_index(self):
        raise AttributeError("Not needed for ColPali")


__all__ = ["ColPaliConfig"]
