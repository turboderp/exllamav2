import torch
import torch.nn.functional as F
from exllamav2.module import ExLlamaV2Module
from exllamav2.rmsnorm import ExLlamaV2RMSNorm
from exllamav2.linear import ExLlamaV2Linear
from exllamav2.ext import exllamav2_ext as ext_c, none_tensor

class ExLlamaV2MLP(ExLlamaV2Module):

    layer_idx: int
    post_attention_layernorm: ExLlamaV2RMSNorm
    gate_proj: ExLlamaV2Linear
    up_proj: ExLlamaV2Linear
    down_proj: ExLlamaV2Linear

    name: str = "MLP"
    submodules: list

    q_handle: int or None = None

    def __init__(self, model, key, layer_idx):
        super().__init__(model, key)

        self.layer_idx = layer_idx

        hidden_size = self.model.config.hidden_size
        intermediate_size = self.model.config.intermediate_size

        self.post_attention_layernorm = ExLlamaV2RMSNorm(model, key + ".post_attention_layernorm")
        self.gate_proj = ExLlamaV2Linear(model, key + ".mlp.gate_proj", hidden_size, intermediate_size, False)
        self.up_proj = ExLlamaV2Linear(model, key + ".mlp.up_proj", hidden_size, intermediate_size, False)
        self.down_proj = ExLlamaV2Linear(model, key + ".mlp.down_proj", intermediate_size, hidden_size, False)

        self.submodules = [self.post_attention_layernorm,
                           self.gate_proj,
                           self.up_proj,
                           self.down_proj]


    def load(self):

        self.post_attention_layernorm.load()
        self.gate_proj.load()
        self.up_proj.load()
        self.down_proj.load()

        if self.gate_proj.is_quant():
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

        self.post_attention_layernorm.unload()
        self.gate_proj.unload()
        self.up_proj.unload()
        self.down_proj.unload()


    def weight_footprint(self):

        return self.post_attention_layernorm.weight_footprint() + \
               self.gate_proj.weight_footprint() + \
               self.up_proj.weight_footprint() + \
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

        return max(self.gate_proj.temp_dq_size(),
                   self.up_proj.temp_dq_size(),
                   self.down_proj.temp_dq_size())


    def set_device_idx(self, idx):
        super().set_device_idx(idx)

        self.post_attention_layernorm.set_device_idx(idx)
        self.gate_proj.set_device_idx(idx)
        self.up_proj.set_device_idx(idx)
        self.down_proj.set_device_idx(idx)

    def forward(self, hidden_states, cache = None, attn_mask = None, past_len = None, intermediates = False):

        if self.q_handle is None or intermediates:
            return self.forward_torch(hidden_states, cache, attn_mask, intermediates)

        ext_c.q_mlp_forward_(self.q_handle, hidden_states.view(-1, hidden_states.shape[-1]))
        return hidden_states


    def forward_torch(self, hidden_states, cache = None, attn_mask = None, intermediates = False):

        residual = hidden_states
        post_norm = self.post_attention_layernorm.forward(hidden_states)

        gate = self.gate_proj.forward(post_norm)
        y = F.silu(gate)
        up = self.up_proj.forward(post_norm)
        y *= up
        down = self.down_proj.forward(y)

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


class ExLlamaV2QMLP(ExLlamaV2Module):

    layer_idx: int

    q_post_attention_layernorm: int or None = None
    q_gate_proj: int or None = None
    q_up_proj: int or None = None
    q_down_proj: int or None = None

    q_tensors: dict or None = None

    def __init__(self, original):
        super().__init__(original.model, original.key)

        q_post_attention_layernorm = original.post_attention_layernorm.q_handle
        q_gate_proj = original.gate_proj.q_handle
        q_up_proj = original.up_proj.q_handle
        q_down_proj = original.down_proj.q_handle

        q_tensors = {}
        q_tensors += original.post_attention_layernorm.q_tensors
        q_tensors += original.gate_proj.q_tensors
        q_tensors += original.up_proj.q_tensors
        q_tensors += original.down_proj.q_tensors

