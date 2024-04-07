from __future__ import annotations
import torch
from exllamav2.fasttensors import STFile
from exllamav2.architecture import ExLlamaV2ArchParams
import os, glob, json
from typing import Any, Dict, List, TypeVar, Union, cast


T = TypeVar('T')
no_default = object()

def read(input_dict: dict[str, Any], expected_type: type, keys: str | list[str], default = no_default) -> T:

    if isinstance(keys, str): keys = [keys]

    for key in keys:

        key_split = key.split("->")
        for subk in key_split[:-1]:
            input_dict = input_dict.get(subk, None)
            if not input_dict:
                key = None
                break
        if key is None: continue
        key = key_split[-1]

        x = input_dict.get(key, None)
        if x is not None:

            if expected_type == float and isinstance(x, int):
                x = float(x)
            if expected_type == int and isinstance(x, float) and x == int(x):
                x = int(x)

            if isinstance(x, expected_type):
                return cast(T, x)
            else:
                raise TypeError(f"Value for {key} is not of expected type {expected_type}")

    if default != no_default: return default
    raise ValueError(f"Missing any of the following keys: {keys}")


class ExLlamaV2Config:

    model_dir: str | None                       # Directory containing model files

    max_seq_len: int                            # Maximum sequence length. Sequences longer than this will throw an exception
    max_batch_size: int                         # Maximum size of batches to process
    max_input_len: int                          # Maximum length of input IDs in a single forward pass. Sequences longer than this will be processed in multiple steps
    max_attention_size: int                     # Sequences will be processed in chunks to keep the size of the attention weights matrix <= this
    max_output_len: int | None                  # Maximum number of output tokens per forward pass

    scale_pos_emb: float                        # Factor by which to scale positional embeddings, e.g. for 4096-token sequence use a scaling factor of 2.0, requires finetuned model or LoRA
    scale_alpha_value: float                    # Alpha value for NTK RoPE scaling. Similar to compress_pos_emb but works without finetuned model

    no_flash_attn: bool                         # Implementation will automatically use flash-attn-2 when available
    fasttensors: bool                           # Experimental, Linux only
    load_in_q4: bool                            # Load float linear layers in Q4 format (for test/dev purposes, not performant)

    max_dq_size: int                            # Max number of elements to dequantize at once

    # Loaded/set by .prepare():

    architecture: str
    arch: ExLlamaV2ArchParams

    model_config: str
    tensor_file_map: dict
    tensor_files: list

    tokenizer_path: str

    bos_token_id: int
    eos_token_id: int
    pad_token_id: int

    hidden_size: int
    initializer_range: float
    intermediate_size: int
    num_attention_heads: int
    num_key_value_heads: int
    num_key_value_groups: int
    num_hidden_layers: int
    norm_eps: float | None
    vocab_size: int
    rotary_embedding_base: float
    head_dim: int
    num_experts: int | None
    num_experts_per_token: int | None
    logit_scale: float
    use_qk_norm: bool

    checkpoint_fused_mlp: bool


    def __init__(self,
                 model_dir: str | None = None):
        """
        :param model_dir:
            If specified, initialize ExLlamaV2Config with values read from model config.
        """

        self.max_batch_size = 1
        self.max_input_len = 2048
        self.max_attention_size = 2048**2
        self.max_output_len = None
        self.scale_pos_emb = 1.0
        self.scale_alpha_value = 1.0

        self.no_flash_attn = False
        self.fasttensors = False
        self.load_in_q4 = False

        if model_dir is not None:
            self.model_dir = model_dir
            self.prepare()
        else:
            self.model_dir = None

        self.max_dq_size = 512*(1024**2)

    # Set low-mem options

    def set_low_mem(self):

        self.max_input_len = 1024
        self.max_attention_size = 1024 ** 2
        self.max_output_len = 1024


    # Populate config with required files from model_dir

    def prepare(self, no_tensors: bool = False):

        assert self.model_dir is not None, "No model_dir specified in ExLlamaV2Config"
        assert os.path.exists(self.model_dir), "Can't find " + self.model_dir

        # Load config.json

        self.model_config = os.path.join(self.model_dir, "config.json")
        assert os.path.exists(self.model_config), "Can't find " + self.model_config

        with open(self.model_config, encoding = "utf8") as f:
            read_config = json.load(f)

        # Model architecture

        assert len(read_config["architectures"]) == 1, "Multiple architectures defined in config.json"
        self.architecture = read_config["architectures"][0]
        self.arch = ExLlamaV2ArchParams(self.architecture, read_config)

        # Vocab params

        self.bos_token_id = read(read_config, int, "bos_token_id", None)  # 1
        self.eos_token_id = read(read_config, int, "eos_token_id", None)  # 2
        self.pad_token_id = read(read_config, int, "pad_token_id", None)  # 0
        self.vocab_size = read(read_config, int, "vocab_size")

        # Standard params

        self.initializer_range = read(read_config, float, ["initializer_range"])
        self.num_hidden_layers = read(read_config, int, ["num_hidden_layers", "n_layers"])

        # Norm params

        if self.arch.norm_eps_key:
            self.norm_eps = read(read_config, float, self.arch.norm_eps_key)
        else:
            self.norm_eps = 1e-5  # Torch default

        # Model dimensions

        self.hidden_size = read(read_config, int, ["hidden_size", "d_model"])

        # Attn params

        self.num_attention_heads = read(read_config, int, ["num_attention_heads", "n_heads"])
        self.head_dim = read(read_config, int, "head_dim", self.hidden_size // self.num_attention_heads)

        self.num_key_value_heads = read(read_config, int, ["num_key_value_heads", "attn_config->kv_n_heads"], self.num_attention_heads)
        self.num_key_value_groups = self.num_attention_heads // self.num_key_value_heads
        self.use_qk_norm = read(read_config, bool, ["use_qk_norm"], False)

        # MLP params

        self.intermediate_size = read(read_config, int, ["intermediate_size", "ffn_config->ffn_hidden_size"])
        self.num_experts = read(read_config, int, ["num_local_experts", "ffn_config->moe_num_experts"], None)
        self.num_experts_per_token = read(read_config, int,["num_experts_per_tok", "ffn_config->moe_top_k"], None)

        # Logit scale

        self.logit_scale = read(read_config, float, "logit_scale", 1)

        # Positional embeddings

        self.rotary_embedding_base = read(read_config, float, ["rope_theta", "attn_config->rope_theta"], 10000.0)

        self.max_seq_len = read(read_config, int,["max_sequence_length",
                                                  "model_max_length",
                                                  "max_position_embeddings",
                                                  "max_seq_len"], 2048)

        rs = read(read_config, dict, "rope_scaling", None)
        if rs and "factor" in rs:
            factor = rs["factor"]
            scaling_type = rs.get("type", None)
            if scaling_type == "linear":
                self.scale_pos_emb = factor
            # elif scaling_type == "yarn":
            #     self.scale_alpha_value = factor

        # Create map of model tensors

        if no_tensors: return

        self.tensor_file_map = {}

        st_pattern = os.path.join(self.model_dir, "*.safetensors")
        self.tensor_files = glob.glob(st_pattern)

        if len(self.tensor_files) == 0:
            raise ValueError(f" ## No .safetensors files found in {self.model_dir}")

        for st_file in self.tensor_files:
            f = STFile.open(st_file, fast = self.fasttensors, keymap = self.arch.keymap)
            for key in f.get_dict():
                self.tensor_file_map[key] = st_file

        # For loading checkpoints with fused MLP layers

        if "model.layers.0.mlp.down_proj.weight" not in self.tensor_file_map and \
            "model.layers.0.mlp.swiglu.w12.weight" in self.tensor_file_map:
            self.checkpoint_fused_mlp = True
            self.arch.make_fused_mlp()
        else:
            self.checkpoint_fused_mlp = False

        # Make sure we found all the layers we need

        expect_keys = self.arch.expect_keys.copy()

        if not self.num_experts or self.num_experts == 1:
            per_layer_keys = self.arch.layer_keys
        else:
            per_layer_keys = set()
            for expert_idx in range(self.num_experts):
                for k in self.arch.layer_keys:
                    skt = [sk.replace(".*.", f".{expert_idx}.") for sk in k]
                    per_layer_keys.add(tuple(skt))
            per_layer_keys = list(per_layer_keys)

        for layer_idx in range(self.num_hidden_layers):
            for ks in per_layer_keys:
                prefixes = [f"model.layers.{layer_idx}.{k}" for k in ks]
                expect_keys.append(prefixes)

        all_keys = set(self.tensor_file_map.keys())
        suffixes = [".q_weight", ".qweight", ".weight", ""]

        for prefixes in expect_keys:
            match = False
            for prefix in prefixes:
                for suffix in suffixes:
                    if (prefix + suffix) in all_keys:
                        match = True
                        break
                    if match: break
                if match: break
            if not match:
                raise ValueError(f" ## Could not find {prefix}.* in model")

        x = 0