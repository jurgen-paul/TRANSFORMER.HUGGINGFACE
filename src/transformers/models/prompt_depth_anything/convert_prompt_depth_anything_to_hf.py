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
"""Convert Prompt Depth Anything checkpoints from the original repository. URL:
https://github.com/DepthAnything/PromptDA"""

import argparse
from pathlib import Path

import requests
import torch
from huggingface_hub import hf_hub_download
from PIL import Image

from transformers import (
    Dinov2Config,
    PromptDepthAnythingConfig,
    PromptDepthAnythingForDepthEstimation,
    PromptDepthAnythingImageProcessor,
)
from transformers.utils import logging


logging.set_verbosity_info()
logger = logging.get_logger(__name__)


def get_dpt_config(model_name):
    if "small" in model_name or "vits" in model_name:
        out_indices = [3, 6, 9, 12]
        backbone_config = Dinov2Config.from_pretrained(
            "facebook/dinov2-small", out_indices=out_indices, apply_layernorm=True, reshape_hidden_states=False
        )
        fusion_hidden_size = 64
        neck_hidden_sizes = [48, 96, 192, 384]
    elif "base" in model_name or "vitb" in model_name:
        out_indices = [3, 6, 9, 12]
        backbone_config = Dinov2Config.from_pretrained(
            "facebook/dinov2-base", out_indices=out_indices, apply_layernorm=True, reshape_hidden_states=False
        )
        fusion_hidden_size = 128
        neck_hidden_sizes = [96, 192, 384, 768]
    elif "large" in model_name or "vitl" in model_name:
        out_indices = [5, 12, 18, 24]
        backbone_config = Dinov2Config.from_pretrained(
            "facebook/dinov2-large", out_indices=out_indices, apply_layernorm=True, reshape_hidden_states=False
        )
        fusion_hidden_size = 256
        neck_hidden_sizes = [256, 512, 1024, 1024]
    else:
        raise NotImplementedError(f"Model not supported: {model_name}")

    depth_estimation_type = "metric"
    max_depth = None

    config = PromptDepthAnythingConfig(
        reassemble_hidden_size=backbone_config.hidden_size,
        patch_size=backbone_config.patch_size,
        backbone_config=backbone_config,
        fusion_hidden_size=fusion_hidden_size,
        neck_hidden_sizes=neck_hidden_sizes,
        depth_estimation_type=depth_estimation_type,
        max_depth=max_depth,
    )

    return config


def create_rename_keys(config):
    rename_keys = []

    # fmt: off
    # stem
    rename_keys.append(("pretrained.cls_token", "backbone.embeddings.cls_token"))
    rename_keys.append(("pretrained.mask_token", "backbone.embeddings.mask_token"))
    rename_keys.append(("pretrained.pos_embed", "backbone.embeddings.position_embeddings"))
    rename_keys.append(("pretrained.patch_embed.proj.weight", "backbone.embeddings.patch_embeddings.projection.weight"))
    rename_keys.append(("pretrained.patch_embed.proj.bias", "backbone.embeddings.patch_embeddings.projection.bias"))

    # Transfomer encoder
    for i in range(config.backbone_config.num_hidden_layers):
        rename_keys.append((f"pretrained.blocks.{i}.ls1.gamma", f"backbone.encoder.layer.{i}.layer_scale1.lambda1"))
        rename_keys.append((f"pretrained.blocks.{i}.ls2.gamma", f"backbone.encoder.layer.{i}.layer_scale2.lambda1"))
        rename_keys.append((f"pretrained.blocks.{i}.norm1.weight", f"backbone.encoder.layer.{i}.norm1.weight"))
        rename_keys.append((f"pretrained.blocks.{i}.norm1.bias", f"backbone.encoder.layer.{i}.norm1.bias"))
        rename_keys.append((f"pretrained.blocks.{i}.norm2.weight", f"backbone.encoder.layer.{i}.norm2.weight"))
        rename_keys.append((f"pretrained.blocks.{i}.norm2.bias", f"backbone.encoder.layer.{i}.norm2.bias"))
        rename_keys.append((f"pretrained.blocks.{i}.mlp.fc1.weight", f"backbone.encoder.layer.{i}.mlp.fc1.weight"))
        rename_keys.append((f"pretrained.blocks.{i}.mlp.fc1.bias", f"backbone.encoder.layer.{i}.mlp.fc1.bias"))
        rename_keys.append((f"pretrained.blocks.{i}.mlp.fc2.weight", f"backbone.encoder.layer.{i}.mlp.fc2.weight"))
        rename_keys.append((f"pretrained.blocks.{i}.mlp.fc2.bias", f"backbone.encoder.layer.{i}.mlp.fc2.bias"))
        rename_keys.append((f"pretrained.blocks.{i}.attn.proj.weight", f"backbone.encoder.layer.{i}.attention.output.dense.weight"))
        rename_keys.append((f"pretrained.blocks.{i}.attn.proj.bias", f"backbone.encoder.layer.{i}.attention.output.dense.bias"))

    # Head
    rename_keys.append(("pretrained.norm.weight", "backbone.layernorm.weight"))
    rename_keys.append(("pretrained.norm.bias", "backbone.layernorm.bias"))

    # activation postprocessing (readout projections + resize blocks)
    # Depth Anything does not use CLS token => readout_projects not required

    for i in range(4):
        rename_keys.append((f"depth_head.projects.{i}.weight", f"neck.reassemble_stage.layers.{i}.projection.weight"))
        rename_keys.append((f"depth_head.projects.{i}.bias", f"neck.reassemble_stage.layers.{i}.projection.bias"))

        if i != 2:
            rename_keys.append((f"depth_head.resize_layers.{i}.weight", f"neck.reassemble_stage.layers.{i}.resize.weight"))
            rename_keys.append((f"depth_head.resize_layers.{i}.bias", f"neck.reassemble_stage.layers.{i}.resize.bias"))

    # refinenet (tricky here)
    mapping = {1:3, 2:2, 3:1, 4:0}

    for i in range(1, 5):
        j = mapping[i]
        rename_keys.append((f"depth_head.scratch.refinenet{i}.out_conv.weight", f"neck.fusion_stage.layers.{j}.projection.weight"))
        rename_keys.append((f"depth_head.scratch.refinenet{i}.out_conv.bias", f"neck.fusion_stage.layers.{j}.projection.bias"))
        rename_keys.append((f"depth_head.scratch.refinenet{i}.resConfUnit1.conv1.weight", f"neck.fusion_stage.layers.{j}.residual_layer1.convolution1.weight"))
        rename_keys.append((f"depth_head.scratch.refinenet{i}.resConfUnit1.conv1.bias", f"neck.fusion_stage.layers.{j}.residual_layer1.convolution1.bias"))
        rename_keys.append((f"depth_head.scratch.refinenet{i}.resConfUnit1.conv2.weight", f"neck.fusion_stage.layers.{j}.residual_layer1.convolution2.weight"))
        rename_keys.append((f"depth_head.scratch.refinenet{i}.resConfUnit1.conv2.bias", f"neck.fusion_stage.layers.{j}.residual_layer1.convolution2.bias"))
        rename_keys.append((f"depth_head.scratch.refinenet{i}.resConfUnit2.conv1.weight", f"neck.fusion_stage.layers.{j}.residual_layer2.convolution1.weight"))
        rename_keys.append((f"depth_head.scratch.refinenet{i}.resConfUnit2.conv1.bias", f"neck.fusion_stage.layers.{j}.residual_layer2.convolution1.bias"))
        rename_keys.append((f"depth_head.scratch.refinenet{i}.resConfUnit2.conv2.weight", f"neck.fusion_stage.layers.{j}.residual_layer2.convolution2.weight"))
        rename_keys.append((f"depth_head.scratch.refinenet{i}.resConfUnit2.conv2.bias", f"neck.fusion_stage.layers.{j}.residual_layer2.convolution2.bias"))
        rename_keys.append((f"depth_head.scratch.refinenet{i}.resConfUnit_depth.0.weight", f"neck.fusion_stage.layers.{j}.residual_layer_depth.convolution1.weight"))
        rename_keys.append((f"depth_head.scratch.refinenet{i}.resConfUnit_depth.0.bias", f"neck.fusion_stage.layers.{j}.residual_layer_depth.convolution1.bias"))
        rename_keys.append((f"depth_head.scratch.refinenet{i}.resConfUnit_depth.2.weight", f"neck.fusion_stage.layers.{j}.residual_layer_depth.convolution2.weight"))
        rename_keys.append((f"depth_head.scratch.refinenet{i}.resConfUnit_depth.2.bias", f"neck.fusion_stage.layers.{j}.residual_layer_depth.convolution2.bias"))
        rename_keys.append((f"depth_head.scratch.refinenet{i}.resConfUnit_depth.4.weight", f"neck.fusion_stage.layers.{j}.residual_layer_depth.convolution3.weight"))
        rename_keys.append((f"depth_head.scratch.refinenet{i}.resConfUnit_depth.4.bias", f"neck.fusion_stage.layers.{j}.residual_layer_depth.convolution3.bias"))

    # scratch convolutions
    for i in range(4):
        rename_keys.append((f"depth_head.scratch.layer{i+1}_rn.weight", f"neck.convs.{i}.weight"))

    # head
    rename_keys.append(("depth_head.scratch.output_conv1.weight", "head.conv1.weight"))
    rename_keys.append(("depth_head.scratch.output_conv1.bias", "head.conv1.bias"))
    rename_keys.append(("depth_head.scratch.output_conv2.0.weight", "head.conv2.weight"))
    rename_keys.append(("depth_head.scratch.output_conv2.0.bias", "head.conv2.bias"))
    rename_keys.append(("depth_head.scratch.output_conv2.2.weight", "head.conv3.weight"))
    rename_keys.append(("depth_head.scratch.output_conv2.2.bias", "head.conv3.bias"))

    return rename_keys


# we split up the matrix of each encoder layer into queries, keys and values
def read_in_q_k_v(state_dict, config):
    hidden_size = config.backbone_config.hidden_size
    for i in range(config.backbone_config.num_hidden_layers):
        # read in weights + bias of input projection layer (in original implementation, this is a single matrix + bias)
        in_proj_weight = state_dict.pop(f"pretrained.blocks.{i}.attn.qkv.weight")
        in_proj_bias = state_dict.pop(f"pretrained.blocks.{i}.attn.qkv.bias")
        # next, add query, keys and values (in that order) to the state dict
        state_dict[f"backbone.encoder.layer.{i}.attention.attention.query.weight"] = in_proj_weight[:hidden_size, :]
        state_dict[f"backbone.encoder.layer.{i}.attention.attention.query.bias"] = in_proj_bias[:hidden_size]
        state_dict[f"backbone.encoder.layer.{i}.attention.attention.key.weight"] = in_proj_weight[
            hidden_size : hidden_size * 2, :
        ]
        state_dict[f"backbone.encoder.layer.{i}.attention.attention.key.bias"] = in_proj_bias[
            hidden_size : hidden_size * 2
        ]
        state_dict[f"backbone.encoder.layer.{i}.attention.attention.value.weight"] = in_proj_weight[-hidden_size:, :]
        state_dict[f"backbone.encoder.layer.{i}.attention.attention.value.bias"] = in_proj_bias[-hidden_size:]


def rename_key(dct, old, new):
    val = dct.pop(old)
    dct[new] = val


name_to_checkpoint = {
    "promptda_vits": "model.ckpt",
    "promptda_vits_transparent": "model.ckpt",
    "promptda_vitl": "model.ckpt",
}


@torch.no_grad()
def convert_dpt_checkpoint(model_name, pytorch_dump_folder_path, push_to_hub, verify_logits):
    """
    Copy/paste/tweak model's weights to our DPT structure.
    """

    # define DPT configuration
    config = get_dpt_config(model_name)

    model_name_to_repo = {
        "promptda_vits": "depth-anything/promptda_vits",
        "promptda_vits_transparent": "depth-anything/promptda_vits_transparent",
        "promptda_vitl": "depth-anything/promptda_vitl",
    }

    # load original state_dict
    repo_id = model_name_to_repo[model_name]
    filename = name_to_checkpoint[model_name]
    filepath = hf_hub_download(
        repo_id=repo_id,
        filename=f"{filename}",
    )

    state_dict = torch.load(filepath, map_location="cpu")["state_dict"]
    state_dict = {key[9:]: state_dict[key] for key in state_dict}
    # rename keys
    rename_keys = create_rename_keys(config)
    for src, dest in rename_keys:
        rename_key(state_dict, src, dest)
    # read in qkv matrices
    read_in_q_k_v(state_dict, config)

    # load HuggingFace model
    model = PromptDepthAnythingForDepthEstimation(config)
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    processor = PromptDepthAnythingImageProcessor(
        do_resize=True,
        size=756,
        ensure_multiple_of=14,
        keep_aspect_ratio=True,
        do_rescale=True,
        do_normalize=True,
        image_mean=[0.485, 0.456, 0.406],
        image_std=[0.229, 0.224, 0.225],
    )
    url = "https://github.com/DepthAnything/PromptDA/blob/main/assets/example_images/image.jpg?raw=true"
    image = Image.open(requests.get(url, stream=True).raw)

    prompt_depth_url = (
        "https://github.com/DepthAnything/PromptDA/blob/main/assets/example_images/arkit_depth.png?raw=true"
    )
    prompt_depth = Image.open(requests.get(prompt_depth_url, stream=True).raw)

    inputs = processor(image, return_tensors="pt", prompt_depth=prompt_depth)

    # Verify forward pass
    with torch.no_grad():
        outputs = model(**inputs)
        predicted_depth = outputs.predicted_depth

    print("Shape of predicted depth:", predicted_depth.shape)
    print("First values:", predicted_depth[0, :3, :3])

    # assert logits
    if verify_logits:
        expected_shape = torch.Size([1, 756, 1008])
        if model_name == "promptda_vits":
            expected_slice = torch.tensor(
                [[3.0100, 3.0016, 3.0219], [3.0046, 3.0137, 3.0275], [3.0083, 3.0191, 3.0292]]
            )
        elif model_name == "promptda_vits_transparent":
            expected_slice = torch.tensor(
                [[3.0058, 3.0397, 3.0460], [3.0314, 3.0393, 3.0504], [3.0326, 3.0465, 3.0545]]
            )
        elif model_name == "promptda_vitl":
            expected_slice = torch.tensor(
                [[3.1336, 3.1358, 3.1363], [3.1368, 3.1267, 3.1414], [3.1397, 3.1385, 3.1448]]
            )
        else:
            raise ValueError("Not supported")
        assert predicted_depth.shape == torch.Size(expected_shape)
        assert torch.allclose(predicted_depth[0, :3, :3], expected_slice, atol=5e-3)  # 5mm tolerance
        print("Looks ok!")

    if pytorch_dump_folder_path is not None:
        Path(pytorch_dump_folder_path).mkdir(exist_ok=True)
        print(f"Saving model and processor to {pytorch_dump_folder_path}")
        model.save_pretrained(pytorch_dump_folder_path)
        processor.save_pretrained(pytorch_dump_folder_path)

    if push_to_hub:
        print("Pushing model and processor to hub...")
        model.push_to_hub(repo_id=f"{model_name.title()}-hf")
        processor.push_to_hub(repo_id=f"{model_name.title()}-hf")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--model_name",
        default="promptda_vits",
        type=str,
        choices=name_to_checkpoint.keys(),
        help="Name of the model you'd like to convert.",
    )
    parser.add_argument(
        "--pytorch_dump_folder_path",
        default=None,
        type=str,
        help="Path to the output PyTorch model directory.",
    )
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Whether to push the model to the hub after conversion.",
    )
    parser.add_argument(
        "--verify_logits",
        action="store_false",
        required=False,
        help="Whether to verify the logits after conversion.",
    )

    args = parser.parse_args()
    convert_dpt_checkpoint(args.model_name, args.pytorch_dump_folder_path, args.push_to_hub, args.verify_logits)
