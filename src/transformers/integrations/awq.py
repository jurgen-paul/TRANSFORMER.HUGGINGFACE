# Copyright 2023 The HuggingFace Team. All rights reserved.
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
"AWQ (Activation aware Weight Quantization) integration file"
from ..activations import ACT2FN
from ..utils import is_auto_awq_available, is_torch_available
from ..utils.quantization_config import AwqBackendPackingMethod, AWQLinearVersion


if is_torch_available():
    import torch
    import torch.nn as nn


AWQ_FUSED_MAPPINGS = {
    "mistral": {
        "attention": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "mlp": ["gate_proj", "up_proj", "down_proj"],
        "layernorm": ["input_layernorm", "post_attention_layernorm", "norm"],
        "use_alibi": False,
    },
    "llama": {
        "attention": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "mlp": ["gate_proj", "up_proj", "down_proj"],
        "layernorm": ["input_layernorm", "post_attention_layernorm", "norm"],
        "use_alibi": False,
    },
}


def replace_with_awq_linear(
    model,
    modules_to_not_convert=None,
    quantization_config=None,
    current_key_name=None,
    has_been_replaced=False,
) -> bool:
    """
    Public method that recursively replaces the Linear layers of the given model with AWQ quantized layers.
    `accelerate` is needed to use this method. Returns the converted model and a boolean that indicates if the
    conversion has been successfull or not.

    During the module replacement, we also infer the backend to use through the `quantization_config` object.

    Args:
        model (`torch.nn.Module`):
            The model to convert, can be any `torch.nn.Module` instance.
        quantization_config (`AwqConfig`):
            The quantization config object that contains the quantization parameters.
        modules_to_not_convert (`list`, *optional*):
            A list of modules to not convert. If a module name is in the list (e.g. `lm_head`), it will not be
            converted.
        current_key_name (`list`, *optional*):
            A list that contains the current key name. This is used for recursion and should not be passed by the user.
        has_been_replaced (`bool`, *optional*):
            A boolean that indicates if the conversion has been successful or not. This is used for recursion and
            should not be passed by the user.
    """
    if modules_to_not_convert is None:
        modules_to_not_convert = []

    backend = quantization_config.backend

    if not is_auto_awq_available():
        raise ValueError(
            "AWQ (either `autoawq` or `llmawq`) is not available. Please install it with `pip install autoawq` or check out the installation guide in https://github.com/mit-han-lab/llm-awq"
        )

    if backend == AwqBackendPackingMethod.AUTOAWQ:
        from awq.modules.linear import WQLinear_GEMM, WQLinear_GEMV
    elif backend == AwqBackendPackingMethod.LLMAWQ:
        from awq.quantize.qmodule import WQLinear

    if backend == AwqBackendPackingMethod.AUTOAWQ:
        target_cls = WQLinear_GEMM if quantization_config.version == AWQLinearVersion.GEMM else WQLinear_GEMV
    else:
        target_cls = WQLinear

    for name, module in model.named_children():
        if current_key_name is None:
            current_key_name = []
        current_key_name.append(name)

        if isinstance(module, nn.Linear) and name not in modules_to_not_convert:
            # Check if the current key is not in the `modules_to_not_convert`
            if not any(key in ".".join(current_key_name) for key in modules_to_not_convert):
                in_features = module.in_features
                out_features = module.out_features

                model._modules[name] = target_cls(
                    w_bit=quantization_config.bits,
                    group_size=quantization_config.group_size,
                    in_features=in_features,
                    out_features=out_features,
                    bias=module.bias is not None,
                    dev=module.weight.device,
                )
                has_been_replaced = True

                # Force requires grad to False to avoid unexpected errors
                model._modules[name].requires_grad_(False)
        if len(list(module.children())) > 0:
            _, has_been_replaced = replace_with_awq_linear(
                module,
                modules_to_not_convert=modules_to_not_convert,
                current_key_name=current_key_name,
                quantization_config=quantization_config,
                has_been_replaced=has_been_replaced,
            )
        # Remove the last key for recursion
        current_key_name.pop(-1)
    return model, has_been_replaced


def get_fusing_mapping(model, quantization_config):
    """
    Returns the fusing mapping given the quantization config and the model

    Args:
        model (`~PreTrainedModel`):
            The model to fuse - note this model should have been converted into AWQ format beforehand.
        quantization_config (`~transformers.quantization_config.AWQConfig`):
            The quantization configuration to use.
    """
    # Always default to `quantization_config.fusing_mapping`
    if quantization_config.fusing_mapping is not None:
        current_fused_mapping = quantization_config.fusing_mapping
        current_fused_mapping["max_seq_len"] = quantization_config.fuse_max_seq_len
    elif model.config.model_type in AWQ_FUSED_MAPPINGS:
        current_fused_mapping = AWQ_FUSED_MAPPINGS[model.config.model_type]

        # Handle hidden_size, num_attention_heads, num_key_value_heads on our own.
        hidden_size = model.config.hidden_size
        num_attention_heads = model.config.num_attention_heads
        num_key_value_heads = getattr(model.config, "num_key_value_heads", num_attention_heads)

        # Fill `current_fused_mapping` with the expected values
        current_fused_mapping["hidden_size"] = hidden_size
        current_fused_mapping["num_attention_heads"] = num_attention_heads
        current_fused_mapping["num_key_value_heads"] = num_key_value_heads
        current_fused_mapping["max_seq_len"] = quantization_config.fuse_max_seq_len
    else:
        raise ValueError(
            "Fusing mapping not found either on the quantization config or the supported `AWQ_FUSED_MAPPINGS`. Please pass a `fused_mapping` argument"
            " in the `quantization_config` or raise an issue on transformers https://github.com/huggingface/transformers to add its support."
        )
    return current_fused_mapping


def fuse_awq_modules(model, quantization_config):
    """
    Optionally fuse some modules in the model to speedup inference.

    Args:
        model (`~PreTrainedModel`):
            The model to fuse - note this model should have been converted into AWQ format beforehand.
        quantization_config (`~transformers.quantization_config.AWQConfig`):
            The quantization configuration to use.
    """
    backend = quantization_config.backend
    fusing_mapping = get_fusing_mapping(model, quantization_config)

    if backend == AwqBackendPackingMethod.AUTOAWQ:
        from awq.modules.fused.attn import QuantAttentionFused
        from awq.modules.fused.mlp import QuantFusedMLP
        from awq.modules.fused.norm import FasterTransformerRMSNorm
        from awq.modules.linear import WQLinear_GEMM, WQLinear_GEMV
    else:
        raise ValueError("Fusing is only supported for the AutoAWQ backend")

    for name, module in model.named_modules():
        # Replace layer norms
        for module_name in fusing_mapping["layernorm"]:
            if hasattr(module, module_name):
                old_module = getattr(module, module_name)
                module._modules[module_name] = FasterTransformerRMSNorm(
                    old_module.weight,
                    old_module.variance_epsilon,
                ).to(old_module.weight.device)
                del old_module
        # Replace MLP layers
        if hasattr(module, fusing_mapping["mlp"][0]):
            gate_proj = getattr(module, fusing_mapping["mlp"][0])
            up_proj = getattr(module, fusing_mapping["mlp"][1])
            down_proj = getattr(module, fusing_mapping["mlp"][2])

            previous_device = gate_proj.qweight.device
            activation_fn = ACT2FN[model.config.hidden_act]
            new_module = QuantFusedMLP(gate_proj, down_proj, up_proj, activation_fn)

            parent_name, child_name = name.rsplit(".", 1)
            parent = model.get_submodule(parent_name)
            setattr(parent, child_name, new_module.to(previous_device))

            del gate_proj, up_proj, down_proj

        # Replace attention layers
        # inside fusing_mapping["attention"] we should have (in correct order): q, k, v, o layer
        if hasattr(module, fusing_mapping["attention"][0]):
            # First, we pack the QKV layers together
            q_proj = getattr(module, fusing_mapping["attention"][0])
            previous_device = q_proj.qweight.device

            if isinstance(q_proj, WQLinear_GEMV):
                target_cls = WQLinear_GEMV
                cat_dim = 0
            else:
                target_cls = WQLinear_GEMM
                cat_dim = 1

            k_proj = getattr(module, fusing_mapping["attention"][1])
            v_proj = getattr(module, fusing_mapping["attention"][2])
            o_proj = getattr(module, fusing_mapping["attention"][3])

            bias = torch.cat([q_proj.bias, k_proj.bias, v_proj.bias], dim=0) if q_proj.bias is not None else None

            qkv_layer = target_cls(
                q_proj.w_bit,
                q_proj.group_size,
                q_proj.in_features,
                q_proj.out_features + k_proj.out_features + v_proj.out_features,
                q_proj.bias is not None,
                next(iter(module.state_dict().values())).device,
            )

            qkv_layer.qweight = torch.cat([q_proj.qweight, k_proj.qweight, v_proj.qweight], dim=cat_dim)
            qkv_layer.qzeros = torch.cat([q_proj.qzeros, k_proj.qzeros, v_proj.qzeros], dim=cat_dim)
            qkv_layer.scales = torch.cat([q_proj.scales, k_proj.scales, v_proj.scales], dim=cat_dim)

            if isinstance(qkv_layer, WQLinear_GEMV):
                qkv_layer.split_k_iters = q_proj.split_k_iters

            qkv_layer.bias = bias

            fused_attention_layer = QuantAttentionFused(
                fusing_mapping["hidden_size"],
                fusing_mapping["num_attention_heads"],
                fusing_mapping["num_key_value_heads"],
                qkv_layer,
                o_proj,
                previous_device,
                fusing_mapping["max_seq_len"],
                use_alibi=fusing_mapping["use_alibi"],
            )

            fused_attention_layer.is_hf_transformers = True

            parent_name, child_name = name.rsplit(".", 1)
            parent = model.get_submodule(parent_name)
            setattr(parent, child_name, fused_attention_layer.to(previous_device))

            del q_proj, k_proj, v_proj, o_proj

    return model
