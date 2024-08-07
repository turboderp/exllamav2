from __future__ import annotations
import torch
import torch.nn.functional as F
from exllamav2.module import ExLlamaV2Module
from exllamav2.attn import ExLlamaV2Attention
from exllamav2.cache import ExLlamaV2CacheBase
from exllamav2.mlp import ExLlamaV2MLP
from exllamav2.rmsnorm import ExLlamaV2RMSNorm
from exllamav2.lora import ExLlamaV2Lora
from exllamav2.layernorm import ExLlamaV2LayerNorm
from exllamav2.ext import exllamav2_ext as ext_c, none_tensor

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from exllamav2.model import ExLlamaV2

class ExLlamaV2ParallelDecoder(ExLlamaV2Module):

    name: str = "ParallelDecoder"

    layer_idx: int
    input_layernorm: ExLlamaV2RMSNorm | ExLlamaV2LayerNorm | None

    attn: ExLlamaV2Attention
    mlp: ExLlamaV2MLP

    def __init__(self,
                 model: ExLlamaV2,
                 key: str,
                 layer_idx: int):
        super().__init__(model, key)

        self.layer_idx = layer_idx

        if self.model.config.arch.norm == "layernorm":
            self.input_layernorm = ExLlamaV2LayerNorm(model, key + self.model.config.arch.norm_key_1)
        elif self.model.config.arch.norm == "rmsnorm":
            self.input_layernorm = ExLlamaV2RMSNorm(model, key + self.model.config.arch.norm_key_1)

        self.attn = ExLlamaV2Attention(model, key, layer_idx, has_norm = False, has_residual = False)
        self.mlp = ExLlamaV2MLP(model, key, layer_idx, has_norm = False, has_residual = False)

        self.submodules = self.attn.submodules + self.mlp.submodules


    def numel(self) -> int:

        return self.attn.numel() + \
               self.mlp.numel() + \
               self.input_layernorm.numel()


    @torch.inference_mode
    def load(self):

        self.input_layernorm.load()
        self.attn.load()
        self.mlp.load()


    def unload(self):

        self.input_layernorm.unload()
        self.attn.unload()
        self.mlp.unload()


    def weight_footprint(self) -> int:

        return \
            self.input_layernorm.weight_footprint() + \
            self.attn.weight_footprint() + \
            self.mlp.weight_footprint()


    def scratch_space_fixed(self) -> int:

        return max(self.attn.scratch_space_fixed(), self.mlp.scratch_space_fixed())


    def scratch_space(self) -> int:

        return max(self.attn.scratch_space(), self.mlp.scratch_space())


    def set_device_idx(self, idx: int | None):
        super().set_device_idx(idx)

        self.input_layernorm.set_device_idx(idx)
        self.attn.set_device_idx(idx)
        self.mlp.set_device_idx(idx)


    def forward(self,
                hidden_states: torch.Tensor,
                cache: ExLlamaV2CacheBase | None = None,
                attn_params: ExLlamaV2Attention.Params | None = None,
                past_len: int | None = None,
                intermediates: bool = False,
                loras: list[ExLlamaV2Lora] | None = None,
                **kwargs) -> torch.Tensor | dict[str: torch.Tensor]:

        if intermediates:
            return self.forward_interm(hidden_states,
                                       cache,
                                       attn_params,
                                       past_len,
                                       intermediates,
                                       loras,
                                       **kwargs)

        a = self.input_layernorm.forward(hidden_states)
        b = a.clone()
        a = self.attn.forward(a, cache, attn_params, past_len, intermediates, loras, **kwargs)
        b = self.mlp.forward(b, cache, attn_params, past_len, intermediates, loras, **kwargs)
        hidden_states += a
        hidden_states += b
        return hidden_states


    def forward_interm(self,
                       hidden_states: torch.Tensor,
                       cache: ExLlamaV2CacheBase | None = None,
                       attn_params: ExLlamaV2Attention.Params | None = None,
                       past_len: int | None = None,
                       intermediates: bool = False,
                       loras: list[ExLlamaV2Lora] | None = None,
                       **kwargs) -> torch.Tensor | dict[str: torch.Tensor]:

        a = self.input_layernorm.forward(hidden_states)
        b = a.clone()
        post_norm = a.clone()
        res_a = self.attn.forward(a, cache, attn_params, past_len, True, loras, **kwargs)
        res_b = self.mlp.forward(b, cache, attn_params, past_len, True, loras, **kwargs)
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


    def is_quant(self) -> bool:
        return self.attn.is_quant() and self.mlp.is_quant()


    def rank_reduce(self, k: float):

        self.attn.rank_reduce(k)
        self.mlp.rank_reduce(k)
