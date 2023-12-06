import torch
import torch.nn.functional as F
from exllamav2.module import ExLlamaV2Module
from exllamav2.rmsnorm import ExLlamaV2RMSNorm
from exllamav2.linear import ExLlamaV2Linear
from exllamav2.ext import exllamav2_ext as ext_c, none_tensor
from exllamav2 import ext
from exllamav2.quip_linear import QuipLinear

class ExLlamaV2MLP(ExLlamaV2Module):

    layer_idx: int
    post_attention_layernorm: ExLlamaV2RMSNorm
    gate_proj: ExLlamaV2Linear or None = None
    up_proj: ExLlamaV2Linear or None = None
    down_proj: ExLlamaV2Linear

    name: str = "MLP"
    submodules: list

    q_handle: int or None = None

    temp_lora_size: int = 0

    def __init__(self, model, key, layer_idx):
        super().__init__(model, key)

        self.layer_idx = layer_idx

        hidden_size = self.model.config.hidden_size
        intermediate_size = self.model.config.intermediate_size

        self.post_attention_layernorm = ExLlamaV2RMSNorm(model, key + ".post_attention_layernorm")
        self.submodules = [self.post_attention_layernorm]

        if 'quip_params' in model.config:
            self.upgate_proj = QuipLinear(model,
                                          key + ".mlp.upgate_proj",
                                          hidden_size,
                                          intermediate_size * 2)
            self.down_proj = QuipLinear(model,
                                        key + ".mlp.down_proj",
                                        model.config.quip_params['ocs_down_size'] if model.config.quip_params['outlier_channel_split'] else self.intermediate_size,
                                        hidden_size)
            self.submodules += [self.upgate_proj, self.down_proj]
        else:
            self.gate_proj = ExLlamaV2Linear(model, key + ".mlp.gate_proj", hidden_size, intermediate_size, False)
            self.up_proj = ExLlamaV2Linear(model, key + ".mlp.up_proj", hidden_size, intermediate_size, False)
            self.down_proj = ExLlamaV2Linear(model, key + ".mlp.down_proj", intermediate_size, hidden_size, False)
            self.submodules += [self.gate_proj, self.up_proj, self.down_proj]


    def load(self):

        self.post_attention_layernorm.load()
        self.upgate_proj.load() if self.upgate_proj is not None else None
        self.gate_proj.load() if self.gate_proj is not None else None
        self.up_proj.load() if self.gate_proj is not None else None

        if self.gate_proj is not None and self.gate_proj.is_quant():
            assert self.up_proj.is_quant() and self.down_proj.is_quant(), "Partially quantized MLP layer"
            device_tensors = self.model.get_device_tensors(self.device_idx)
            device_tensors.begin_scratch_alloc()
            self.q_handle = ext_c.make_q_mlp(self.post_attention_layernorm.weight,
                                            self.post_attention_layernorm.variance_epsilon,
                                            self.gate_proj.q_handle,
                                            self.up_proj.q_handle,
                                            self.down_proj.q_handle,
                                            device_tensors.get_scratch_slice(self.temp_state_size()),
                                            device_tensors.get_scratch_slice(self.temp_a_size()),
                                            device_tensors.get_scratch_slice(self.temp_b_size()),
                                            device_tensors.get_scratch_slice(self.temp_dq_size()),
                                            self.model.config.max_input_len * self.model.config.max_batch_size)


    def unload(self):
        if self.q_handle is not None:
            ext_c.free_q_mlp(self.q_handle)
            self.q_handle = None

        self.post_attention_layernorm.unload()
        self.upgate_proj.unload() if self.upgate_proj is not None else None
        self.gate_proj.unload() if self.gate_proj is not None else None
        self.up_proj.unload() if self.up_proj is not None else None
        self.down_proj.unload()


    def weight_footprint(self):

        return self.post_attention_layernorm.weight_footprint() + \
               self.gate_proj.weight_footprint() if self.gate_proj is not None else 0 + \
               self.up_proj.weight_footprint() if self.up_proj is not None else 0 + \
               self.upgate_proj.weight_footprint() if self.upgate_proj is not None else 0 + \
               self.down_proj.weight_footprint()


    def scratch_space_fixed(self):

        return self.temp_state_size() + \
               self.temp_a_size() + \
               self.temp_b_size() + \
               self.temp_dq_size()


    def scratch_space(self):

        assert self.model.config.intermediate_size >= self.model.config.hidden_size
        return self.temp_state_size() + \
               self.temp_a_size() + \
               self.temp_b_size() + \
               self.temp_dq_size()


    def temp_state_size(self):

        return self.model.config.max_input_len * self.model.config.max_batch_size * self.model.config.hidden_size * 2 + 128


    def temp_a_size(self):

        return self.model.config.max_input_len * self.model.config.max_batch_size * self.model.config.intermediate_size * 2 + 128


    def temp_b_size(self):

        return self.model.config.max_input_len * self.model.config.max_batch_size * self.model.config.intermediate_size * 2 + 128


    def temp_dq_size(self):

        return max(self.gate_proj.temp_dq_size() if self.gate_proj is not None else 0,
                   self.up_proj.temp_dq_size() if self.up_proj is not None else 0,
                   self.upgate_proj.temp_dq_size() if self.upgate_proj is not None else 0,
                   self.down_proj.temp_dq_size())


    def set_device_idx(self, idx):
        super().set_device_idx(idx)

        self.post_attention_layernorm.set_device_idx(idx)
        self.gate_proj.set_device_idx(idx) if self.gate_proj is not None else None
        self.up_proj.set_device_idx(idx) if self.gate_proj is not None else None
        self.down_proj.set_device_idx(idx)

    def forward(self, hidden_states, cache = None, attn_mask = None, past_len = None, intermediates = False, loras = None, position_offsets = None):

        if self.q_handle is None or intermediates:
            return self.forward_torch(hidden_states, cache, attn_mask, intermediates, loras = loras)

        if loras is None or self.temp_lora_size == 0:
            pass_loras = []
            pass_lora_temp = ext.none_tensor
        else:
            pass_loras = [id(x) for x in loras]
            pass_lora_temp = torch.empty((self.temp_lora_size,), dtype = torch.half, device = hidden_states.device)

        ext_c.q_mlp_forward_(self.q_handle,
                             hidden_states.view(-1, hidden_states.shape[-1]),
                             pass_loras,
                             pass_lora_temp)

        return hidden_states


    def forward_torch(self, hidden_states, cache = None, attn_mask = None, intermediates = False, loras = None, position_offsets = None):

        residual = hidden_states
        post_norm = self.post_attention_layernorm.forward(hidden_states)

        if self.upgate_proj is not None:
            # QuiP forward path
            upgate = self.upgate_proj(post_norm.to(torch.float32), loras = loras)
            gate = self.gate_scale * upgate[..., self.intermediate_size:(self.intermediate_size * 2)]
            up = self.up_scale * upgate[...,0:self.intermediate_size]
            down = self.down_scale * self.down_proj(F.silu(gate) * up, loras = loras).half()
        else:
            gate = self.gate_proj.forward(post_norm, loras = loras)
            y = F.silu(gate)
            up = self.up_proj.forward(post_norm, loras = loras)
            y *= up
            down = self.down_proj.forward(y, loras = loras)

        hidden_states = down + residual

        if intermediates:
            return {"post_norm": post_norm,
                    "gate": gate,
                    "up": up,
                    "pre_down": y,
                    "down": down,
                    "hidden_states": hidden_states}
        else:
            return hidden_states


    def update_loras(self):

        if self.q_handle is None: return

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
        return self.q_handle is not None or self.upgate_proj is not None


