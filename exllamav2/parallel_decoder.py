import torch
import torch.nn.functional as F
from torch import nn
from exllamav2.module import ExLlamaV2Module
from exllamav2.attn import ExLlamaV2Attention
from exllamav2.mlp import ExLlamaV2MLP
from exllamav2.rmsnorm import ExLlamaV2RMSNorm
from exllamav2.layernorm import ExLlamaV2LayerNorm
from exllamav2.ext import exllamav2_ext as ext_c, none_tensor
from exllamav2 import ext

class ExLlamaV2ParallelDecoder(ExLlamaV2Module):

    layer_idx: int
    input_layernorm: ExLlamaV2RMSNorm or ExLlamaV2LayerNorm or None

    attn: ExLlamaV2Attention
    mlp: ExLlamaV2MLP

    name: str = "ParallelDecoder"
    submodules: list

    def __init__(self, model, key, layer_idx):
        super().__init__(model, key)

        self.layer_idx = layer_idx

        hidden_size = self.model.config.hidden_size
        intermediate_size = self.model.config.intermediate_size

        if self.model.config.arch.norm == "layernorm":
            self.input_layernorm = ExLlamaV2LayerNorm(model, key + self.model.config.arch.norm_key_1)
        elif self.model.config.arch.norm == "rmsnorm":
            self.input_layernorm = ExLlamaV2RMSNorm(model, key + self.model.config.arch.norm_key_1)

        self.attn = ExLlamaV2Attention(model, key, layer_idx, has_norm = False, has_residual = False)
        self.mlp = ExLlamaV2MLP(model, key, layer_idx, has_norm = False, has_residual = False)

        self.submodules = self.attn.submodules + self.mlp.submodules


    def numel(self):

        return self.attn.numel() + \
               self.mlp.numel()


    def load(self):

        self.input_layernorm.load()
        self.attn.load()
        self.mlp.load()


    def unload(self):

        self.input_layernorm.unload()
        self.attn.unload()
        self.mlp.unload()


    def weight_footprint(self):

        return \
            self.input_layernorm.weight_footprint() + \
            self.attn.weight_footprint() + \
            self.mlp.weight_footprint()


    def scratch_space_fixed(self):

        return max(self.attn.scratch_space_fixed(), self.mlp.scratch_space_fixed())


    def scratch_space(self):

        return max(self.attn.scratch_space(), self.mlp.scratch_space())



    def set_device_idx(self, idx):
        super().set_device_idx(idx)

        self.input_layernorm.set_device_idx(idx)
        self.attn.set_device_idx(idx)
        self.mlp.set_device_idx(idx)


    def forward(self, hidden_states, cache = None, attn_params = None, past_len = None, intermediates = False, loras = None):

        if intermediates:
            return self.forward_interm(hidden_states, cache, attn_params, past_len, intermediates, loras)

        a = self.input_layernorm.forward(hidden_states)
        b = a.clone()
        a = self.attn.forward(a, cache, attn_params, past_len, intermediates, loras)
        b = self.mlp.forward(b, cache, attn_params, past_len, intermediates, loras)
        hidden_states += a
        hidden_states += b
        return hidden_states


    def forward_interm(self, hidden_states, cache = None, attn_params = None, past_len = None, intermediates = False, loras = None, position_offsets = None):

        a = self.input_layernorm.forward(hidden_states)
        b = a.clone()
        post_norm = a.clone()
        res_a = self.attn.forward(a, cache, attn_params, past_len, True, loras)
        res_b = self.mlp.forward(b, cache, attn_params, past_len, True, loras)
        hidden_states += res_a["hidden_states"]
        hidden_states += res_b["hidden_states"]

        if intermediates:
            return {"post_norm": post_norm,
                    "attn_output": res_a["attn_output"],
                    "pre_down": res_b["pre_down"],
                    "hidden_states_attn": res_a["hidden_states"],
                    "hidden_states_mlp": res_b["hidden_states"],
                    "hidden_states": hidden_states}
        else:
            return hidden_states


    def update_loras(self):

        self.attn.update_loras()
        self.mlp.update_loras()


    def is_quant(self):
        return self.attn.is_quant() and self.mlp.is_quant()


    def rank_reduce(self, k):

        self.attn.rank_reduce(k)
        self.mlp.rank_reduce(k)
