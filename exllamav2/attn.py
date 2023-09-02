import torch
from torch import nn
import torch.nn.functional as F
from exllamav2.module import ExLlamaV2Module
from exllamav2.rmsnorm import ExLlamaV2RMSNorm
from exllamav2.linear import ExLlamaV2Linear
from exllamav2.cache import ExLlamaV2Cache
import math
from exllamav2 import ext
from exllamav2.ext import exllamav2_ext as ext_c, none_tensor
import sys


class ExLlamaV2Attention(ExLlamaV2Module):

    layer_idx: int
    input_layernorm: ExLlamaV2RMSNorm
    q_proj: ExLlamaV2Linear
    k_proj: ExLlamaV2Linear
    v_proj: ExLlamaV2Linear
    o_proj: ExLlamaV2Linear

    name: str = "Attention"
    submodules: list

    q_handle: int or None = None

    temp_state: torch.tensor
    temp_q: torch.tensor
    temp_k: torch.tensor
    temp_v: torch.tensor
    temp_o: torch.tensor
    temp_dq: torch.tensor
    # temp_kv: torch.tensor

    def __init__(self, model, key, layer_idx):
        super().__init__(model, key)

        self.layer_idx = layer_idx

        hidden_size = self.model.config.hidden_size

        self.input_layernorm = ExLlamaV2RMSNorm(model, key + ".input_layernorm")
        self.q_proj = ExLlamaV2Linear(model, key + ".self_attn.q_proj", hidden_size, self.model.config.num_attention_heads * self.model.config.head_dim, False)
        self.k_proj = ExLlamaV2Linear(model, key + ".self_attn.k_proj", hidden_size, self.model.config.num_key_value_heads * self.model.config.head_dim, False)
        self.v_proj = ExLlamaV2Linear(model, key + ".self_attn.v_proj", hidden_size, self.model.config.num_key_value_heads * self.model.config.head_dim, False)
        self.o_proj = ExLlamaV2Linear(model, key + ".self_attn.o_proj", self.model.config.num_attention_heads * self.model.config.head_dim, hidden_size, False)

        self.submodules = [self.input_layernorm,
                           self.q_proj,
                           self.k_proj,
                           self.v_proj,
                           self.o_proj]


    def load(self):

        self.input_layernorm.load()
        self.q_proj.load()
        self.k_proj.load()
        self.v_proj.load()
        self.o_proj.load()

        if self.q_proj.is_quant():

            assert self.k_proj.is_quant() and self.v_proj.is_quant() and self.o_proj.is_quant(), "Partially quantized attention layer"

            device_tensors = self.model.get_device_tensors(self.device_idx)
            device_tensors.begin_scratch_alloc()
            self.temp_state = device_tensors.get_scratch_slice(self.temp_state_size())
            self.temp_q = device_tensors.get_scratch_slice(self.temp_q_size())
            self.temp_k = device_tensors.get_scratch_slice(self.temp_k_size())
            self.temp_v = device_tensors.get_scratch_slice(self.temp_v_size())
            self.temp_dq = device_tensors.get_scratch_slice(self.temp_dq_size())
            # self.temp_kv = device_tensors.get_scratch_slice(self.temp_kv_size()) if self.model.config.num_attention_heads != self.model.config.num_key_value_heads else None

            self.q_handle = ext_c.make_q_attn(self.input_layernorm.weight,
                                              self.input_layernorm.variance_epsilon,
                                              self.q_proj.q_handle,
                                              self.k_proj.q_handle,
                                              self.v_proj.q_handle,
                                              self.o_proj.q_handle,
                                              self.temp_state,
                                              self.temp_q,
                                              self.temp_k,
                                              self.temp_v,
                                              self.temp_dq,
                                              self.model.config.max_input_len * self.model.config.max_batch_size,
                                              self.model.config.hidden_size,
                                              self.model.config.num_attention_heads,
                                              self.model.config.num_key_value_heads,
                                              self.model.config.head_dim,
                                              self.model.config.max_seq_len)


    def unload(self):

        self.input_layernorm.unload()
        self.q_proj.unload()
        self.k_proj.unload()
        self.v_proj.unload()
        self.o_proj.unload()


    def weight_footprint(self):

        return self.input_layernorm.weight_footprint() + \
               self.q_proj.weight_footprint() + \
               self.k_proj.weight_footprint() + \
               self.v_proj.weight_footprint() + \
               self.o_proj.weight_footprint()


    def scratch_space(self):

        return self.temp_state_size() + \
               self.temp_q_size() + \
               self.temp_k_size() + \
               self.temp_v_size() + \
               self.temp_dq_size() + \
               self.temp_attn_size()


    def temp_state_size(self):

        return self.model.config.max_input_len * self.model.config.max_batch_size * self.model.config.hidden_size * 2 + 128


    def temp_q_size(self):

        return self.model.config.max_input_len * self.model.config.max_batch_size * self.model.config.num_attention_heads * self.model.config.head_dim * 2 + 128


    def temp_k_size(self):

        return self.model.config.max_input_len * self.model.config.max_batch_size * self.model.config.num_key_value_heads * self.model.config.head_dim * 2 + 128


    def temp_v_size(self):

        return self.model.config.max_input_len * self.model.config.max_batch_size * self.model.config.num_key_value_heads * self.model.config.head_dim * 2 + 128


    def temp_dq_size(self):

        return max(self.q_proj.temp_dq_size(),
                   self.k_proj.temp_dq_size(),
                   self.v_proj.temp_dq_size(),
                   self.o_proj.temp_dq_size())


    def temp_kv_size(self):

        return self.model.config.max_seq_len * self.model.config.max_batch_size * self.model.config.num_attention_heads * self.model.config.head_dim * 2 + 128


    def temp_attn_size(self):

        return self.model.config.max_attention_size * self.model.config.max_batch_size * 2 + 128


    def set_device_idx(self, idx):
        super().set_device_idx(idx)

        self.input_layernorm.set_device_idx(idx)
        self.q_proj.set_device_idx(idx)
        self.k_proj.set_device_idx(idx)
        self.v_proj.set_device_idx(idx)
        self.o_proj.set_device_idx(idx)


    def repeat_kv(self, hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:

        if n_rep == 1: return hidden_states

        batch, num_key_value_heads, slen, head_dim = hidden_states.shape
        hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        hidden_states = hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
        return hidden_states


    def forward(self, hidden_states, cache = None, attn_mask = None, intermediates = False):

        if self.q_handle is None or intermediates or cache is None:
            return self.forward_torch(hidden_states, cache, attn_mask, intermediates)

        residual = hidden_states

        batch_size = hidden_states.shape[0]
        q_len = hidden_states.shape[1]
        past_len = cache.current_seq_len

        constants = self.model.get_device_tensors(self.device_idx)

        ext_c.q_attn_forward_(self.q_handle,
                              hidden_states,
                              batch_size,
                              q_len,
                              past_len,
                              cache.key_states[self.layer_idx],
                              cache.value_states[self.layer_idx],
                              constants.sin,
                              constants.cos,
                              self.temp_q,
                              attn_mask if attn_mask is not None else ext.none_tensor)
                              # self.temp_kv if self.temp_kv is not None else ext.none_tensor)

        return hidden_states


    def forward_torch(self, hidden_states, cache = None, attn_mask = None, intermediates = False):

        num_attention_heads = self.model.config.num_attention_heads
        num_key_value_heads = self.model.config.num_key_value_heads
        num_key_value_groups = self.model.config.num_key_value_groups
        head_dim = self.model.config.head_dim
        hidden_size = self.model.config.hidden_size

        bsz, q_len, _ = hidden_states.size()
        past_len = 0 if cache is None else cache.current_seq_len

        residual = hidden_states
        post_norm = self.input_layernorm.forward(hidden_states)

        # Project q, k, v

        query_states_im = self.q_proj.forward(post_norm)
        key_states_im = self.k_proj.forward(post_norm)
        value_states_im = self.v_proj.forward(post_norm)

        if intermediates:

            query_states = query_states_im.clone()
            key_states = key_states_im.clone()
            value_states = value_states_im.clone()

        else:

            query_states = query_states_im
            key_states = key_states_im
            value_states = value_states_im

        query_states = query_states.view(bsz, q_len, num_attention_heads, head_dim)
        key_states = key_states.view(bsz, q_len, num_key_value_heads, head_dim)
        value_states = value_states.view(bsz, q_len, num_key_value_heads, head_dim)

        # Apply position embeddings

        constants = self.model.get_device_tensors(self.device_idx, scratch = False)

        ext_c.rope_(query_states, constants.sin, constants.cos, past_len, self.model.config.num_attention_heads, self.model.config.head_dim)
        ext_c.rope_(key_states, constants.sin, constants.cos, past_len, self.model.config.num_key_value_heads, self.model.config.head_dim)

        # Add keys and values to cache

        if cache is not None:

            new_keys = cache.key_states[self.layer_idx].narrow(1, past_len, q_len)
            new_values = cache.value_states[self.layer_idx].narrow(1, past_len, q_len)
            new_keys.copy_(key_states)
            new_values.copy_(value_states)

            # Key/value tensors with past

            key_states = cache.key_states[self.layer_idx].narrow(1, 0, past_len + q_len)
            value_states = cache.value_states[self.layer_idx].narrow(1, 0, past_len + q_len)

        # Attention

        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        key_states = self.repeat_kv(key_states, self.model.config.num_key_value_groups)
        key_states = key_states.transpose(-1, -2)

        attn_weights = torch.matmul(query_states, key_states)

        attn_weights /= math.sqrt(head_dim)
        if attn_mask is not None: attn_weights = attn_weights + attn_mask
        attn_weights = nn.functional.softmax(attn_weights, dim = -1, dtype = torch.float16)

        value_states = self.repeat_kv(value_states, self.model.config.num_key_value_groups)
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2)

        # Output projection

        attn_output = attn_output.reshape(bsz, q_len, hidden_size)
        attn_proj = self.o_proj.forward(attn_output)

        # Add residual connection

        hidden_states = attn_proj + residual

        if intermediates:
            return {"post_norm": post_norm,
                    "query_states": query_states_im,
                    "key_states": key_states_im,
                    "value_states": value_states_im,
                    "attn_output": attn_output,
                    "attn_proj": attn_proj,
                    "hidden_states": hidden_states}
        else:
            return hidden_states



    def forward_old(self, hidden_states, cache = None, attn_mask = None, intermediates = False):

        num_attention_heads = self.model.config.num_attention_heads
        head_dim = self.model.config.head_dim
        hidden_size = self.model.config.hidden_size

        bsz, q_len, _ = hidden_states.size()
        past_len = 0 if cache is None else cache.current_seq_len

        residual = hidden_states
        post_norm = self.input_layernorm.forward(hidden_states)

        # Project q, k, v

        query_states_im = self.q_proj.forward(post_norm)
        key_states_im = self.k_proj.forward(post_norm)
        value_states_im = self.v_proj.forward(post_norm)

        if intermediates:

            query_states = query_states_im.clone()
            key_states = key_states_im.clone()
            value_states = value_states_im.clone()

        else:

            query_states = query_states_im
            key_states = key_states_im
            value_states = value_states_im

        query_states = query_states.view(bsz, q_len, num_attention_heads, head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, num_attention_heads, head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, num_attention_heads, head_dim).transpose(1, 2)

        # Apply position embeddings

        constants = self.model.get_device_tensors(self.device_idx)

        cos_emb = constants.cos.narrow(2, past_len, q_len)
        sin_emb = constants.sin.narrow(2, past_len, q_len)

        def rotate_half(x):
            half_size = x.shape[-1] // 2
            x1 = x.narrow(-1, 0, half_size)
            x2 = x.narrow(-1, half_size, half_size)
            return torch.cat((-x2, x1), dim = -1)

        query_states_r = rotate_half(query_states)
        query_states_r.mul_(sin_emb)
        query_states.mul_(cos_emb)
        query_states.add_(query_states_r)

        key_states_r = rotate_half(key_states)
        key_states_r.mul_(sin_emb)
        key_states.mul_(cos_emb)
        key_states.add_(key_states_r)

        # Add keys and values to cache

        if cache is not None:

            new_keys = cache.key_states[self.layer_idx].narrow(2, past_len, q_len)
            new_values = cache.value_states[self.layer_idx].narrow(2, past_len, q_len)
            new_keys.copy_(key_states)
            new_values.copy_(value_states)

            # Key/value tensors with past

            key_states = cache.key_states[self.layer_idx].narrow(2, 0, past_len + q_len)
            value_states = cache.value_states[self.layer_idx].narrow(2, 0, past_len + q_len)

        # Attention

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3))
        attn_weights /= math.sqrt(head_dim)
        if attn_mask is not None: attn_weights = attn_weights + attn_mask
        attn_weights = nn.functional.softmax(attn_weights, dim = -1, dtype = torch.float16)
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2)

        # Output projection

        attn_output = attn_output.reshape(bsz, q_len, hidden_size)
        attn_proj = self.o_proj.forward(attn_output)

        # Add residual connection

        hidden_states = attn_proj + residual

        if intermediates:
            return {"post_norm": post_norm,
                    "query_states": query_states_im,
                    "key_states": key_states_im,
                    "value_states": value_states_im,
                    "attn_output": attn_output,
                    "attn_proj": attn_proj,
                    "hidden_states": hidden_states}
        else:
            return hidden_states
