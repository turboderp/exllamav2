from __future__ import annotations
import torch
import torch.nn.functional as F
from torch import nn
from exllamav2.module import ExLlamaV2Module
from exllamav2.rmsnorm import ExLlamaV2RMSNorm
from exllamav2.layernorm import ExLlamaV2LayerNorm
from exllamav2.linear import ExLlamaV2Linear
from exllamav2.ext import exllamav2_ext as ext_c, none_tensor
from exllamav2.lora import ExLlamaV2Lora

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from exllamav2.model import ExLlamaV2


class ExLlamaV2MLP(ExLlamaV2Module):

    name: str = "MLP"

    layer_idx: int
    post_attention_layernorm: ExLlamaV2RMSNorm | ExLlamaV2LayerNorm | None
    gate_proj: ExLlamaV2Linear | None
    up_proj: ExLlamaV2Linear | None
    down_proj: ExLlamaV2Linear | None

    q_handle: int | None

    temp_lora_size: int

    has_norm: bool
    has_residual: bool

    def __init__(self,
                 model: ExLlamaV2,
                 key: str,
                 layer_idx: int,
                 has_norm: bool = True,
                 has_residual: bool = True):

        super().__init__(model, key)

        self.layer_idx = layer_idx
        self.has_norm = has_norm
        self.has_residual = has_residual

        self.q_handle = None
        self.temp_lora_size = 0

        hidden_size = self.model.config.hidden_size
        intermediate_size = self.model.config.intermediate_size

        if self.has_norm:
            if self.model.config.arch.norm == "layernorm":
                self.post_attention_layernorm = ExLlamaV2LayerNorm(model, key + self.model.config.arch.norm_key_2)
            elif self.model.config.arch.norm == "rmsnorm":
                self.post_attention_layernorm = ExLlamaV2RMSNorm(model, key + self.model.config.arch.norm_key_2)
        else:
            self.post_attention_layernorm = None

        self.up_proj = ExLlamaV2Linear(model, key + self.model.config.arch.mlp_key_up, hidden_size, intermediate_size, self.model.config.arch.mlp_bias)
        self.down_proj = ExLlamaV2Linear(model, key + self.model.config.arch.mlp_key_down, intermediate_size, hidden_size, self.model.config.arch.mlp_bias)

        self.submodules = [self.up_proj,
                           self.down_proj]
        if self.has_norm:
            self.submodules += [self.post_attention_layernorm]

        if self.model.config.arch.mlp_gate:
            self.gate_proj = ExLlamaV2Linear(model, key + self.model.config.arch.mlp_key_gate, hidden_size, intermediate_size, self.model.config.arch.mlp_bias)
            self.submodules += [self.gate_proj]
        else:
            self.gate_proj = None


    def numel(self) -> int:

        numel = self.up_proj.numel() + \
                self.down_proj.numel()

        if self.model.config.arch.mlp_gate:
            numel += self.gate_proj.numel()

        if self.post_attention_layernorm is not None:
            numel += self.post_attention_layernorm.numel()

        return numel


    def load(self):

        if self.post_attention_layernorm is not None:
            self.post_attention_layernorm.load()

        if self.model.config.checkpoint_fused_mlp:
            w12 = self.load_weight(self.key + self.model.config.arch.fused_mlp_key_12)
            w1 = nn.Parameter(w12[:self.model.config.intermediate_size, :].contiguous())
            w2 = nn.Parameter(w12[self.model.config.intermediate_size:, :].contiguous())
            w3 = self.load_weight(self.key + self.model.config.arch.fused_mlp_key_3)
            self.gate_proj.load(w1)
            self.up_proj.load(w2)
            self.down_proj.load(w3)
        else:
            if self.gate_proj is not None: self.gate_proj.load()
            self.up_proj.load()
            self.down_proj.load()

        if self.up_proj.is_quant():
            assert self.gate_proj is None or self.gate_proj.is_quant()
            assert self.up_proj.is_quant(), "Partially quantized MLP layer"
            device_tensors = self.model.get_device_tensors(self.device_idx)
            device_tensors.begin_scratch_alloc()

            if self.has_norm:
                norm_weight = self.post_attention_layernorm.weight if self.post_attention_layernorm.weight is not None else none_tensor
                norm_bias = self.post_attention_layernorm.bias if self.post_attention_layernorm.bias is not None else none_tensor
                is_rms = isinstance(self.post_attention_layernorm, ExLlamaV2RMSNorm)
                eps = self.post_attention_layernorm.variance_epsilon
            else:
                norm_weight = none_tensor
                norm_bias = none_tensor
                is_rms = False
                eps = 0

            self.q_handle = ext_c.make_q_mlp(norm_weight,
                                             norm_bias,
                                             is_rms,
                                             eps,
                                             0 if self.gate_proj is None else self.gate_proj.q_handle,
                                             self.up_proj.q_handle,
                                             self.down_proj.q_handle,
                                             device_tensors.get_scratch_slice(self.temp_state_size()),
                                             device_tensors.get_scratch_slice(self.temp_a_size()),
                                             device_tensors.get_scratch_slice(self.temp_b_size()),
                                             device_tensors.get_scratch_slice(self.temp_dq_size()),
                                             self.model.config.max_input_len * self.model.config.max_batch_size,
                                             self.model.config.arch.mlp_act_func == "gelu",
                                             self.has_residual)


    def unload(self):

        if self.q_handle is not None:
            ext_c.free_q_mlp(self.q_handle)
            self.q_handle = None

        if self.post_attention_layernorm is not None: self.post_attention_layernorm.unload()
        if self.gate_proj is not None: self.gate_proj.unload()
        self.up_proj.unload()
        self.down_proj.unload()


    def weight_footprint(self) -> int:

        if self.model.config.checkpoint_fused_mlp:
            fp = 3 * self.model.config.intermediate_size * self.model.config.hidden_size * 2
        else:
            fp = self.up_proj.weight_footprint() + \
                 self.down_proj.weight_footprint()
            if self.gate_proj is not None:
                fp += self.gate_proj.weight_footprint()

        if self.post_attention_layernorm is not None:
            fp += self.post_attention_layernorm.weight_footprint()

        return fp


    def scratch_space_fixed(self) -> int:

        return self.temp_state_size() + \
               self.temp_a_size() + \
               self.temp_b_size() + \
               self.temp_dq_size()


    def scratch_space(self) -> int:

        assert self.model.config.intermediate_size >= self.model.config.hidden_size
        return self.temp_state_size() + \
               self.temp_a_size() + \
               self.temp_b_size() + \
               self.temp_dq_size()


    def temp_state_size(self) -> int:

        return self.model.config.max_input_len * self.model.config.max_batch_size * self.model.config.hidden_size * 2 + 128


    def temp_a_size(self) -> int:

        return self.model.config.max_input_len * self.model.config.max_batch_size * self.model.config.intermediate_size * 2 + 128


    def temp_b_size(self) -> int:

        return self.model.config.max_input_len * self.model.config.max_batch_size * self.model.config.intermediate_size * 2 + 128


    def temp_dq_size(self) -> int:

        return max(0 if self.gate_proj is None else self.gate_proj.temp_dq_size(),
                   self.up_proj.temp_dq_size(),
                   self.down_proj.temp_dq_size())


    def set_device_idx(self, idx: int):
        super().set_device_idx(idx)

        if self.post_attention_layernorm is not None:
            self.post_attention_layernorm.set_device_idx(idx)
        if self.gate_proj is not None: self.gate_proj.set_device_idx(idx)
        self.up_proj.set_device_idx(idx)
        self.down_proj.set_device_idx(idx)


    def forward(self,
                hidden_states: torch.Tensor,
                cache = None,
                attn_params = None,
                past_len = None,
                intermediates: bool = False,
                loras: list[ExLlamaV2Lora] | None = None,
                **kwargs) -> torch.Tensor | dict[str: torch.Tensor]:

        if self.q_handle is None or intermediates:
            return self.forward_torch(hidden_states, cache, attn_params, past_len, intermediates, loras = loras, **kwargs)

        if loras is None or self.temp_lora_size == 0:
            pass_loras = []
            pass_lora_temp = none_tensor
        else:
            pass_loras = [id(x) for x in loras]
            pass_lora_temp = torch.empty((self.temp_lora_size,), dtype = torch.half, device = hidden_states.device)

        ext_c.q_mlp_forward_(self.q_handle,
                             hidden_states.view(-1, hidden_states.shape[-1]),
                             pass_loras,
                             pass_lora_temp)

        return hidden_states


    def forward_torch(self,
                      hidden_states: torch.Tensor,
                      cache = None,
                      attn_params = None,
                      past_len = None,
                      intermediates: bool = False,
                      loras: list[ExLlamaV2Lora] | None = None,
                      **kwargs) -> torch.Tensor | dict[str: torch.Tensor]:

        residual = hidden_states
        post_norm = self.post_attention_layernorm.forward(hidden_states) \
            if self.has_norm else hidden_states

        if self.gate_proj is not None:
            gate = self.gate_proj.forward(post_norm, loras = loras)
            if self.model.config.arch.mlp_act_func == "silu":
                y = F.silu(gate)
            elif self.model.config.arch.mlp_act_func == "gelu":
                y = F.gelu(gate)
            up = self.up_proj.forward(post_norm, loras = loras)
            y *= up
            y.clamp_(min = -65504.0, max = 65504.0)
        else:
            up = self.up_proj.forward(post_norm, loras = loras)
            if self.model.config.arch.mlp_act_func == "silu":
                y = F.silu(up)
            elif self.model.config.arch.mlp_act_func == "gelu":
                y = F.gelu(up)

        down = self.down_proj.forward(y, loras = loras)
        hidden_states = down + residual if self.has_residual else down

        if intermediates:
            return {"post_norm": post_norm,
                    "pre_down": y,
                    "hidden_states": hidden_states}
        else:
            return hidden_states


    def update_loras(self):

        if self.q_handle is None: return

        if self.gate_proj is None:
            gate_proj_lora_a = {}
            gate_proj_lora_b = {}
        else:
            gate_proj_lora_a = { id(k): v for k, v in self.gate_proj.lora_a_tensors.items() }
            gate_proj_lora_b = { id(k): v for k, v in self.gate_proj.lora_b_tensors.items() }

        up_proj_lora_a = { id(k): v for k, v in self.up_proj.lora_a_tensors.items() }
        up_proj_lora_b = { id(k): v for k, v in self.up_proj.lora_b_tensors.items() }
        down_proj_lora_a = { id(k): v for k, v in self.down_proj.lora_a_tensors.items() }
        down_proj_lora_b = { id(k): v for k, v in self.down_proj.lora_b_tensors.items() }

        temp_lora_size = ext_c.q_mlp_set_loras(self.q_handle,
                                               gate_proj_lora_a,
                                               gate_proj_lora_b,
                                               up_proj_lora_a,
                                               up_proj_lora_b,
                                               down_proj_lora_a,
                                               down_proj_lora_b)

        self.temp_lora_size = temp_lora_size * self.model.config.max_batch_size * self.model.config.max_input_len


    def is_quant(self):
        return self.q_handle is not None


    def rank_reduce(self, k):

        if self.gate_proj is not None: self.gate_proj.rank_reduce(k)
        self.up_proj.rank_reduce(k)
        self.down_proj.rank_reduce(k)
