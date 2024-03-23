from __future__ import annotations
import torch
from exllamav2.fasttensors import STFile
from exllamav2.architecture import ExLlamaV2ArchParams
import os, glob, json

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
    initializer_range: int
    intermediate_size: int
    num_attention_heads: int
    num_key_value_heads: int
    num_key_value_groups: int
    num_hidden_layers: int
    norm_eps: float
    vocab_size: int
    rotary_embedding_base: float
    head_dim: int
    num_experts: int | None
    num_experts_per_token: int | None
    logit_scale: float

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

        self.bos_token_id = read_config.get("bos_token_id", 1)
        self.eos_token_id = read_config.get("eos_token_id", 2)
        self.pad_token_id = read_config.get("pad_token_id", 0)
        self.vocab_size = read_config["vocab_size"]

        # Standard params

        self.initializer_range = read_config["initializer_range"]
        self.num_hidden_layers = read_config["num_hidden_layers"]

        # Norm params

        self.norm_eps = read_config[self.arch.norm_eps_key]

        # Model dimensions

        self.hidden_size = read_config["hidden_size"]

        # Attn params

        self.num_attention_heads = read_config["num_attention_heads"]
        self.head_dim = read_config.get("head_dim", self.hidden_size // self.num_attention_heads)

        if "num_key_value_heads" in read_config:
            self.num_key_value_heads = read_config["num_key_value_heads"]
            self.num_key_value_groups = self.num_attention_heads // self.num_key_value_heads
        else:
            self.num_key_value_heads = self.num_attention_heads
            self.num_key_value_groups = 1

        # MLP params

        self.intermediate_size = read_config["intermediate_size"]
        self.num_experts = read_config.get("num_local_experts", None)
        self.num_experts_per_token = read_config.get("num_experts_per_tok", None)

        # Logit scale

        self.logit_scale = read_config.get("logit_scale", 1)

        # Positional embeddings

        self.rotary_embedding_base = read_config.get("rope_theta", 10000.0)

        self.max_seq_len = read_config.get("max_sequence_length",
                           read_config.get("max_position_embeddings",
                           2048))

        rs = read_config.get("rope_scaling", None)
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
            f = STFile.open(st_file, fast = self.fasttensors)
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

        for layer_idx in range(self.num_hidden_layers):
            for ks in self.arch.layer_keys:
                prefixes = [f"model.layers.{layer_idx}.{k}" for k in ks]
                expect_keys.append(prefixes)

        for prefixes in expect_keys:
            for prefix in prefixes:
                if any(key.startswith(prefix) for key in self.tensor_file_map):
                    break
            else:
                raise ValueError(f" ## Could not find {prefix}.* in model")

