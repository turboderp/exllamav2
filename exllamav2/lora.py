from __future__ import annotations
from exllamav2.config import ExLlamaV2Config
from exllamav2.linear import ExLlamaV2Linear
import os, json
from safetensors.torch import load_file as safe_load_file
from torch import load as load_file
import torch
from exllamav2.compat import safe_move_tensor

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from exllamav2.model import ExLlamaV2

class ExLlamaV2Lora:

    model: ExLlamaV2

    lora_config_path: str
    lora_path: str

    lora_config_path: str
    lora_path: str
    lora_r: int
    lora_alpha: float
    lora_scaling: float
    config: ExLlamaV2Config
    bias_ignored: bool

    tensors: dict
    target_modules: dict

    @staticmethod
    def from_directory(model, directory, lora_scaling = 1.0):
        config_path = os.path.join(directory, "adapter_config.json")
        lora_path_bin = os.path.join(directory, "adapter_model.bin")
        lora_path_st = os.path.join(directory, "adapter_model.safetensors")
        if os.path.exists(lora_path_bin): return ExLlamaV2Lora(model, config_path, lora_path_bin, lora_scaling)
        if os.path.exists(lora_path_st): return ExLlamaV2Lora(model, config_path, lora_path_st, lora_scaling)
        raise ValueError(f"No LoRA found in {directory}")

    @torch.inference_mode
    def __init__(self,
                 model: ExLlamaV2,
                 lora_config_path: str,
                 lora_path: str,
                 lora_scaling: float = 1.0):

        self.lora_config_path = lora_config_path
        self.lora_path = lora_path
        self.model = model
        self.config = model.config
        self.tensors = {}
        self.target_modules = {}
        self.bias_ignored = False
        self.lora_scaling = lora_scaling
        self.embed_tokens = None
        self.lm_head = None

        # Compatibility check

        assert not self.model.config.arch.residual_stream_fp32, \
            "LoRAs not (yet) supported for models with FP32 residual stream"

        # Grab relevant items from LoRA config

        with open(lora_config_path, encoding = "utf8") as f:
            read_config = json.load(f)

        self.lora_r = read_config["r"]
        self.lora_alpha = float(read_config["lora_alpha"])
        self.lora_scaling *= self.lora_alpha / self.lora_r

        if "fan_in_fan_out" in read_config and read_config["fan_in_fan_out"]:
            raise ValueError(" ## Error: fan_in_fan_out mode not supported.")

        # Load LoRA weights

        if self.lora_path.endswith(".safetensors"):
            f = safe_load_file(self.lora_path, device = "cpu")
        else:
            f = load_file(self.lora_path, map_location = "cpu")

        for key in f.keys():
            if any(key.endswith(x) for x in [".original_module.weight", ".modules_to_save.weight"]):
                continue
            tensor = f[key]

            # Find target
            if key.endswith(f'{self.config.arch.lm_head_key}.weight'):
                if tensor.dtype == torch.bfloat16:
                    tensor = tensor.to(torch.float16)
                elif tensor.dtype == torch.float32:
                    tensor = tensor.to(torch.float16)
                target_module = self.model.modules_dict["lm_head"]
                tensor = safe_move_tensor(tensor, target_module.device())
                self.lm_head = torch.nn.Linear(target_module.in_features, tensor.shape[0], bias = False, device = "meta")
                self.lm_head.weight = torch.nn.Parameter(tensor, requires_grad=False)
                continue
            elif key.endswith(f'embed_tokens.weight'):
                if tensor.dtype == torch.bfloat16:
                    tensor = tensor.to(torch.float16)
                elif tensor.dtype == torch.float32:
                    tensor = tensor.to(torch.float16)
                target_module = self.model.modules_dict["model.embed_tokens"]
                tensor = safe_move_tensor(tensor, target_module.device())
                self.embed_tokens = torch.nn.Embedding(tensor.shape[0], self.config.hidden_size, self.config.pad_token_id, device = "meta")
                weight = torch.nn.Parameter(tensor, requires_grad=False)
                if self.model.config.scale_emb != 1:
                    weight *= self.model.config.scale_emb
                self.embed_tokens.weight = weight
                continue

            i = key.find("model.layers.")
            if i == -1: raise ValueError(f" ## Error: unsupported layer in {self.lora_path}: {key}")

            target_key = key[i:]
            ks = target_key.split(".")
            decoder_idx = int(ks[2])
            decoder_part = ks[3]
            decoder_layer = ".".join(ks[4:-2])
            lora_half = ks[-2]

            if lora_half == "bias":
                epsilon = 1e-6
                if torch.max(tensor) > epsilon or torch.max(tensor) < -epsilon:
                    raise ValueError(f" ## Error: unsupported bias target {self.lora_path}: {key}")
                self.bias_ignored = True
                continue

            target_module = self.model.modules_dict["model.layers." + str(decoder_idx) + "." + decoder_part + "." + decoder_layer]
            # if decoder_part == "self_attn": target_module = target_module.self_attn
            # elif decoder_part == "mlp": target_module = target_module.mlp
            # else: raise ValueError(f" ## Error: unsupported layer in {self.lora_path}: {key}")

            # if   decoder_layer == "q_proj": target_module = target_module.q_proj
            # elif decoder_layer == "k_proj": target_module = target_module.k_proj
            # elif decoder_layer == "v_proj": target_module = target_module.v_proj
            # elif decoder_layer == "o_proj": target_module = target_module.o_proj
            # elif decoder_layer == "gate_proj": target_module = target_module.gate_proj
            # elif decoder_layer == "up_proj": target_module = target_module.up_proj
            # elif decoder_layer == "down_proj": target_module = target_module.down_proj
            # else: raise ValueError(f" ## Error: unsupported layer in {self.lora_path}: {key}")

            # Check that shape is compatible

            assert isinstance(target_module, ExLlamaV2Linear)

            if lora_half == "lora_A":
                in_features = tensor.shape[1]
                out_features = None
            elif lora_half == "lora_B":
                in_features = None
                out_features = tensor.shape[0]
            else: raise ValueError(f" ## Error: unsupported layer in {self.lora_path}: {key}")

            if (in_features and in_features != target_module.in_features) or (out_features and out_features != target_module.out_features):
                raise ValueError(f" ## Error: incompatible tensor shape in {self.lora_path}: {key}")

            # For efficiency, transpose adapter instead of transposing state during inference

            tensor = tensor.T.contiguous()

            # Pre-scale

            if lora_half == "lora_B" and self.lora_scaling != 1.0: tensor.mul_(self.lora_scaling)

            # Check that dtype is compatible, or convert

            if tensor.dtype == torch.bfloat16:
                tensor = tensor.to(torch.float16)

            elif tensor.dtype == torch.float32:
                tensor = tensor.to(torch.float16)

            elif tensor.dtype == torch.float16:
                pass

            else: raise ValueError(f" ## Error: unsupported tensor dtype in {self.lora_path}")

            # Move to target device

            tensor = safe_move_tensor(tensor, target_module.device())
            if lora_half == "lora_A": target_module.lora_a_tensors[self] = tensor
            if lora_half == "lora_B": target_module.lora_b_tensors[self] = tensor

            # Store adapter tensor

            self.tensors[target_key] = tensor
            self.target_modules[target_key] = target_module

        self.model.update_loras()


    def unload(self):

        for k, v in self.target_modules.items():
            if self in v.lora_a_tensors: del v.lora_a_tensors[self]
            if self in v.lora_b_tensors: del v.lora_b_tensors[self]

        self.tensors = {}
        self.target_modules = {}

        self.model.update_loras()





