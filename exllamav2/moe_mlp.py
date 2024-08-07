from __future__ import annotations
import torch
import torch.nn.functional as F
from exllamav2.module import ExLlamaV2Module
from exllamav2.rmsnorm import ExLlamaV2RMSNorm
from exllamav2.layernorm import ExLlamaV2LayerNorm
from exllamav2.linear import ExLlamaV2Linear
from exllamav2.lora import ExLlamaV2Lora
from exllamav2.ext import exllamav2_ext as ext_c, none_tensor

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from exllamav2.model import ExLlamaV2

class ExLlamaV2MoEMLP(ExLlamaV2Module):

    name: str = "MoE MLP"

    layer_idx: int
    post_attention_layernorm: ExLlamaV2RMSNorm | ExLlamaV2LayerNorm
    w1: list
    w2: list
    w3: list
    gate: ExLlamaV2Linear
    num_experts: int
    num_experts_per_token: int

    q_handle: int | None

    temp_lora_size: int

    def __init__(self,
                 model: ExLlamaV2,
                 key: str,
                 layer_idx: int):
        super().__init__(model, key)

        cfg = self.model.config

        self.layer_idx = layer_idx

        self.q_handle = None
        self.temp_lora_size = 0

        hidden_size = cfg.hidden_size
        intermediate_size = cfg.intermediate_size
        self.num_experts = cfg.num_experts
        self.num_experts_per_token = cfg.num_experts_per_token

        if cfg.arch.norm == "layernorm":
            self.post_attention_layernorm = ExLlamaV2LayerNorm(model, key + self.model.config.arch.norm_key_2)
        elif cfg.arch.norm == "rmsnorm":
            self.post_attention_layernorm = ExLlamaV2RMSNorm(model, key + self.model.config.arch.norm_key_2)

        w1_key = key + cfg.arch.mlp_key_gate
        w2_key = key + cfg.arch.mlp_key_down
        w3_key = key + cfg.arch.mlp_key_up
        w1_f_key = w1_key.replace(".*.", ".")
        w2_f_key = w2_key.replace(".*.", ".")
        w3_f_key = w3_key.replace(".*.", ".")

        gate_key = cfg.arch.mlp_key_expert_gate

        self.w1 = []
        self.w2 = []
        self.w3 = []

        bu = 0
        # bd = 0
        for e in range(self.num_experts):
            au = bu
            # ad = bd
            bu += intermediate_size
            # bd += hidden_size
            w1 = ExLlamaV2Linear(model, w1_key.replace("*", str(e)), hidden_size, intermediate_size, cfg.arch.mlp_bias, f_key = w1_f_key, f_beg = au, f_end = bu)
            w2 = ExLlamaV2Linear(model, w2_key.replace("*", str(e)), intermediate_size, hidden_size, cfg.arch.mlp_bias, f_key = w2_f_key, f_beg = au, f_end = bu)
            w3 = ExLlamaV2Linear(model, w3_key.replace("*", str(e)), hidden_size, intermediate_size, cfg.arch.mlp_bias, f_key = w3_f_key, f_beg = au, f_end = bu)
            self.w1.append(w1)
            self.w2.append(w2)
            self.w3.append(w3)

        self.gate = ExLlamaV2Linear(model, key + gate_key, hidden_size, self.num_experts, False, pad32 = False)

        self.submodules = [self.post_attention_layernorm,
                           self.gate] + \
                           self.w1 + \
                           self.w2 + \
                           self.w3


    def numel(self) -> int:

        return sum(l.numel() for l in self.w1 + self.w2 + self.w3)


    @torch.inference_mode
    def load(self):

        self.post_attention_layernorm.load()
        self.gate.load()
        for e in range(self.num_experts):
            self.w1[e].load()
            self.w2[e].load()
            self.w3[e].load()

        if self.w1[0].is_quant():
            device_context = self.model.get_device_context(self.device_idx)
            device_context.begin_scratch_alloc()
            self.q_handle = ext_c.make_q_moe_mlp(self.post_attention_layernorm.weight,
                                                 self.post_attention_layernorm.bias if self.post_attention_layernorm.bias is not None else none_tensor,
                                                 isinstance(self.post_attention_layernorm, ExLlamaV2RMSNorm),
                                                 self.post_attention_layernorm.variance_epsilon,
                                                 self.gate.linear.weight,
                                                 self.num_experts,
                                                 self.num_experts_per_token,
                                                 [w.q_handle for w in self.w1],
                                                 [w.q_handle for w in self.w2],
                                                 [w.q_handle for w in self.w3],
                                                 device_context.get_scratch_slice(self.temp_state_size()),
                                                 device_context.get_scratch_slice(self.temp_gathered_state_size()),
                                                 device_context.get_scratch_slice(self.temp_a_size()),
                                                 device_context.get_scratch_slice(self.temp_b_size()),
                                                 device_context.get_scratch_slice(self.temp_logit_size()),
                                                 device_context.get_scratch_slice(self.temp_dq_size()),
                                                 self.model.config.max_input_len * self.model.config.max_batch_size,
                                                 self.model.config.arch.mlp_act_func == "gelu")


    def unload(self):
        if self.q_handle is not None:
            ext_c.free_q_moe_mlp(self.q_handle)
            self.q_handle = None

        self.post_attention_layernorm.unload()
        self.gate.unload()
        for e in range(self.num_experts):
            self.w1[e].unload()
            self.w2[e].unload()
            self.w3[e].unload()


    def weight_footprint(self) -> int:

        return self.post_attention_layernorm.weight_footprint() + \
               self.gate.weight_footprint() + \
               sum(self.w1[e].weight_footprint() for e in range(self.num_experts)) + \
               sum(self.w2[e].weight_footprint() for e in range(self.num_experts)) + \
               sum(self.w3[e].weight_footprint() for e in range(self.num_experts))


    def scratch_space_fixed(self) -> int:

        return self.temp_state_size() + \
               self.temp_gathered_state_size() + \
               self.temp_a_size() + \
               self.temp_b_size() + \
               self.temp_logit_size() + \
               self.temp_dq_size()


    def scratch_space(self) -> int:

        assert self.model.config.intermediate_size >= self.model.config.hidden_size
        return self.temp_state_size() + \
               self.temp_gathered_state_size() + \
               self.temp_a_size() + \
               self.temp_b_size() + \
               self.temp_logit_size() + \
               self.temp_dq_size()


    def temp_state_size(self) -> int:

        return self.model.config.max_input_len * self.model.config.max_batch_size * self.model.config.hidden_size * 2 + 128


    def temp_gathered_state_size(self) -> int:

        return self.model.config.max_input_len * self.model.config.max_batch_size * self.model.config.hidden_size * 2 + 128


    def temp_a_size(self) -> int:

        return self.model.config.max_input_len * self.model.config.max_batch_size * self.model.config.intermediate_size * 2 + 128


    def temp_b_size(self) -> int:

        return self.model.config.max_input_len * self.model.config.max_batch_size * self.model.config.intermediate_size * 2 + 128


    def temp_dq_size(self) -> int:

        return max(self.w1[0].temp_dq_size(),
                   self.w2[0].temp_dq_size(),
                   self.w3[0].temp_dq_size())


    def temp_logit_size(self) -> int:

        return self.model.config.max_input_len * self.model.config.max_batch_size * self.model.config.num_experts * 2 + 128


    def set_device_idx(self, idx: int | None):
        super().set_device_idx(idx)

        self.post_attention_layernorm.set_device_idx(idx)
        self.gate.set_device_idx(idx)
        for e in range(self.num_experts):
            self.w1[e].set_device_idx(idx)
            self.w2[e].set_device_idx(idx)
            self.w3[e].set_device_idx(idx)


    def forward(self,
                hidden_states: torch.Tensor,
                cache = None,
                attn_params = None,
                past_len = None,
                intermediates: bool = False,
                loras: list[ExLlamaV2Lora] | None = None,
                **kwargs) -> torch.Tensor | dict[str: torch.Tensor]:

        batch_size, sequence_length, hidden_dim = hidden_states.shape

        # TODO: LoRA currently uses the Torch codepath. Needs conditional (early-exit) kernels with output scaling
        # for the LoRA matmuls in order to work with the C++ path

        if self.q_handle is None or intermediates or batch_size * sequence_length > 4 or self.num_experts not in [4, 8, 16] or (loras is not None and len(loras) > 0):
            return self.forward_torch(hidden_states, cache, attn_params, past_len, intermediates, loras = loras, **kwargs)

        # if loras is None or self.temp_lora_size == 0:
        #     pass_loras = []
        #     pass_lora_temp = none_tensor
        # else:
        #     pass_loras = [id(x) for x in loras]
        #     pass_lora_temp = torch.empty((self.temp_lora_size,), dtype = torch.half, device = hidden_states.device)

        # ref = self.forward_torch(hidden_states, cache, attn_params, intermediates, loras = loras)
        # ext_c.q_moe_mlp_forward_(self.q_handle, hidden_states.view(-1, hidden_states.shape[-1]), pass_loras, pass_lora_temp)
        ext_c.q_moe_mlp_forward_(self.q_handle, hidden_states.view(-1, hidden_states.shape[-1]))

        return hidden_states


    def forward_torch(self,
                      hidden_states: torch.Tensor,
                      cache = None,
                      attn_params = None,
                      past_len = None,
                      intermediates = False,
                      loras: list[ExLlamaV2Lora] | None = None,
                      **kwargs) -> torch.Tensor | dict[str: torch.Tensor]:

        residual = hidden_states

        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)

        # Layernorm

        hidden_states = self.post_attention_layernorm.forward(hidden_states)
        if intermediates: result = { "post_norm": hidden_states }

        # Get router logits

        router_logits = self.gate.forward(hidden_states, loras = loras)  #[:, :self.num_experts]

        # Get routing weights and select top K experts

        routing_weights = F.softmax(router_logits, dim = -1, dtype = torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, self.num_experts_per_token, dim = -1)
        routing_weights /= routing_weights.sum(dim = -1, keepdim = True)
        routing_weights = routing_weights.to(hidden_states.dtype)

        final_hidden_states = torch.zeros((batch_size * sequence_length, hidden_dim), dtype = hidden_states.dtype, device = hidden_states.device)

        expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes = self.num_experts).permute(2, 1, 0)

        for expert_idx in range(self.num_experts):
            idx, top_x = torch.where(expert_mask[expert_idx])

            if top_x.shape[0] == 0: continue  # Skip experts that weren't selected at all

            top_x_list = top_x.tolist()
            idx_list = idx.tolist()

            current_state = hidden_states[None, top_x_list].reshape(-1, hidden_dim)

            gate = self.w1[expert_idx].forward(current_state, loras = loras)
            up = self.w3[expert_idx].forward(current_state, loras = loras)

            if self.model.config.arch.mlp_act_func == "silu":
                current_hidden_states = F.silu(gate)
            elif self.model.config.arch.mlp_act_func == "gelu":
                current_hidden_states = F.gelu(gate)
            current_hidden_states *= up
            if intermediates: result[f"pre_down.{expert_idx}"] = current_hidden_states

            current_hidden_states = self.w2[expert_idx].forward(current_hidden_states, loras = loras)
            current_hidden_states *= routing_weights[top_x_list, idx_list, None]

            final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))

        final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
        final_hidden_states += residual

        if intermediates:
            result["hidden_states"] = final_hidden_states
            return result
        else:
            return final_hidden_states


    def update_loras(self):
        pass
        # if self.q_handle is None: return
        #
        # w1_lora_a = []
        # w1_lora_b = []
        # w2_lora_a = []
        # w2_lora_b = []
        # w3_lora_a = []
        # w3_lora_b = []
        # for i in range(self.model.config.num_experts):
        #     w1_lora_a.append({ id(k): v for k, v in self.w1[i].lora_a_tensors.items() })
        #     w1_lora_b.append({ id(k): v for k, v in self.w1[i].lora_b_tensors.items() })
        #     w2_lora_a.append({ id(k): v for k, v in self.w2[i].lora_a_tensors.items() })
        #     w2_lora_b.append({ id(k): v for k, v in self.w2[i].lora_b_tensors.items() })
        #     w3_lora_a.append({ id(k): v for k, v in self.w3[i].lora_a_tensors.items() })
        #     w3_lora_b.append({ id(k): v for k, v in self.w3[i].lora_b_tensors.items() })
        #
        # temp_lora_size = ext_c.q_moe_mlp_set_loras(self.q_handle,
        #                                            w1_lora_a,
        #                                            w1_lora_b,
        #                                            w2_lora_a,
        #                                            w2_lora_b,
        #                                            w3_lora_a,
        #                                            w3_lora_b)
        #
        # self.temp_lora_size = temp_lora_size * self.model.config.max_batch_size * self.model.config.max_input_len


    def is_quant(self):
        return self.q_handle is not None


    def rank_reduce(self, k):

        for e in range(self.num_experts):
            self.w1[e].rank_reduce(k)
            self.w2[e].rank_reduce(k)
            self.w3[e].rank_reduce(k)
