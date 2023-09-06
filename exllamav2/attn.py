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
# from flash_attn import flash_attn_func
# import xformers.ops as xops


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
            # self.temp_q = device_tensors.get_scratch_slice(self.temp_q_size())
            # self.temp_k = device_tensors.get_scratch_slice(self.temp_k_size())
            # self.temp_v = device_tensors.get_scratch_slice(self.temp_v_size())
            self.temp_dq = device_tensors.get_scratch_slice(self.temp_dq_size())
            # self.temp_kv = device_tensors.get_scratch_slice(self.temp_kv_size()) if self.model.config.num_attention_heads != self.model.config.num_key_value_heads else None

            self.q_handle = ext_c.make_q_attn(self.input_layernorm.weight,
                                              self.input_layernorm.variance_epsilon,
                                              self.q_proj.q_handle,
                                              self.k_proj.q_handle,
                                              self.v_proj.q_handle,
                                              self.o_proj.q_handle,
                                              self.temp_state,
                                              # self.temp_q,
                                              # self.temp_k,
                                              # self.temp_v,
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


    def forward(self, hidden_states, cache = None, attn_mask = None, past_len = None, intermediates = False):

        if self.q_handle is None or intermediates:
            return self.forward_torch(hidden_states, cache, attn_mask, past_len, intermediates)

        batch_size = hidden_states.shape[0]
        q_len = hidden_states.shape[1]

        # past_len = 0
        # if cache is not None:
        #     if isinstance(cache, ExLlamaV2Cache):
        #         past_len = cache.current_seq_len
        #     if isinstance(cache, list):
        #         past_len = [c.current_seq_len for c in cache]

        num_attention_heads = self.model.config.num_attention_heads
        num_key_value_heads = self.model.config.num_key_value_heads
        num_key_value_groups = self.model.config.num_key_value_groups
        head_dim = self.model.config.head_dim
        hidden_size = self.model.config.hidden_size

        constants = self.model.get_device_tensors(self.device_idx)

        q_shape = hidden_states.shape[:-1] + (self.q_proj.out_features,)
        k_shape = hidden_states.shape[:-1] + (self.k_proj.out_features,)
        v_shape = hidden_states.shape[:-1] + (self.v_proj.out_features,)
        q_states = torch.empty(q_shape, device = hidden_states.device, dtype = torch.half)
        k_states = torch.empty(k_shape, device = hidden_states.device, dtype = torch.half)
        v_states = torch.empty(v_shape, device = hidden_states.device, dtype = torch.half)

        # RMS norm, Q/K/V projections, position embeddings

        ext_c.q_attn_forward_1(self.q_handle,
                               hidden_states,
                               batch_size,
                               q_len,
                               -1 if isinstance(past_len, tuple) else past_len,
                               past_len[0] if isinstance(past_len, tuple) else ext.none_tensor,
                               q_states,
                               k_states,
                               v_states,
                               constants.sin,
                               constants.cos)

        q_states = q_states.view(batch_size, q_len, num_attention_heads, head_dim)
        k_states = k_states.view(batch_size, q_len, num_key_value_heads, head_dim)
        v_states = v_states.view(batch_size, q_len, num_key_value_heads, head_dim)

        # Regular (batched) attention with optional padding mask

        if cache is None or isinstance(cache, ExLlamaV2Cache):

            # Add keys and values to cache

            if cache is not None:

                # TODO: For batch_size == 1, cached K, V states are contiguous and consecutive, so they can be written directly into the cache

                batch_keys = cache.key_states[self.layer_idx].narrow(0, 0, batch_size)
                batch_values = cache.value_states[self.layer_idx].narrow(0, 0, batch_size)

                new_keys = batch_keys.narrow(1, past_len, q_len)
                new_values = batch_values.narrow(1, past_len, q_len)
                new_keys.copy_(k_states)
                new_values.copy_(v_states)

                # Key/value tensors with past

                k_states = batch_keys.narrow(1, 0, past_len + q_len)
                v_states = batch_values.narrow(1, 0, past_len + q_len)

            # Torch matmul attention

            q_states = q_states.transpose(1, 2)
            k_states = k_states.transpose(1, 2)
            v_states = v_states.transpose(1, 2)

            k_states = self.repeat_kv(k_states, num_key_value_groups)
            k_states = k_states.transpose(-1, -2)

            attn_weights = torch.matmul(q_states, k_states)

            attn_weights /= math.sqrt(head_dim)
            if attn_mask is not None: attn_weights = attn_weights + attn_mask
            attn_weights = nn.functional.softmax(attn_weights, dim = -1, dtype = torch.float16)

            v_states = self.repeat_kv(v_states, num_key_value_groups)
            attn_output = torch.matmul(attn_weights, v_states)
            attn_output = attn_output.transpose(1, 2)
            attn_output = attn_output.reshape((batch_size, q_len, hidden_size))

            # Flash Attention 2.0

            # attn_output = flash_attn_func(q_states, k_states, v_states, causal = True)
            # attn_output = attn_output.reshape((batch_size, q_len, hidden_size));

            # xformers memory_efficient_attention

            # attn_output = xops.memory_efficient_attention(q_states, k_states, v_states, attn_bias = xops.LowerTriangularMask())
            # attn_output = attn_output.reshape((batch_size, q_len, hidden_size));

            # Torch SDP attention:

            # q_states = q_states.transpose(1, 2)
            # k_states = k_states.transpose(1, 2)
            # v_states = v_states.transpose(1, 2)
            #
            # # k_states = self.repeat_kv(k_states, num_key_value_groups)
            # # v_states = self.repeat_kv(v_states, num_key_value_groups)
            #
            # attn_output = F.scaled_dot_product_attention(q_states, k_states, v_states, attn_mask = attn_mask, is_causal = False)
            # attn_output = attn_output.transpose(1, 2)
            # attn_output = attn_output.reshape((batch_size, q_len, hidden_size))

        # Multiple caches

        else:

            attn_outputs = []
            for i in range(len(cache)):

                # TODO: Once nested tensors are finalized in Torch, this could all be batched, probably

                # Add keys and values to cache

                batch_keys = cache[i].key_states[self.layer_idx]
                batch_values = cache[i].value_states[self.layer_idx]

                new_keys = batch_keys.narrow(1, past_len[1][i], q_len)
                new_values = batch_values.narrow(1, past_len[1][i], q_len)
                new_keys.copy_(k_states.narrow(0, i, 1))
                new_values.copy_(v_states.narrow(0, i, 1))

                # Key/value tensors with past

                k_states_b = batch_keys.narrow(1, 0, past_len[1][i] + q_len)
                v_states_b = batch_values.narrow(1, 0, past_len[1][i] + q_len)

                # Torch matmul attention

                q_states_b = q_states.transpose(1, 2).narrow(0, i, 1)
                k_states_b = k_states_b.transpose(1, 2)
                v_states_b = v_states_b.transpose(1, 2)

                k_states_b = self.repeat_kv(k_states_b, num_key_value_groups)
                k_states_b = k_states_b.transpose(-1, -2)

                attn_weights = torch.matmul(q_states_b, k_states_b)

                attn_weights /= math.sqrt(head_dim)
                if attn_mask is not None: attn_weights = attn_weights + attn_mask[i]
                attn_weights = nn.functional.softmax(attn_weights, dim = -1, dtype = torch.float16)

                v_states_b = self.repeat_kv(v_states_b, num_key_value_groups)
                attn_output_b = torch.matmul(attn_weights, v_states_b)
                attn_outputs.append(attn_output_b)

            attn_output = torch.cat(attn_outputs, dim = 0)
            attn_output = attn_output.transpose(1, 2)
            attn_output = attn_output.reshape((batch_size, q_len, hidden_size))

            # Output projection

        ext_c.q_attn_forward_2(self.q_handle,
                               hidden_states,
                               attn_output,
                               batch_size,
                               q_len)

        return hidden_states


    def forward_torch(self, hidden_states, cache = None, attn_mask = None, past_len = None, intermediates = False):

        num_attention_heads = self.model.config.num_attention_heads
        num_key_value_heads = self.model.config.num_key_value_heads
        num_key_value_groups = self.model.config.num_key_value_groups
        head_dim = self.model.config.head_dim
        hidden_size = self.model.config.hidden_size

        batch_size, q_len, _ = hidden_states.size()
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

        query_states = query_states.view(batch_size, q_len, num_attention_heads, head_dim)
        key_states = key_states.view(batch_size, q_len, num_key_value_heads, head_dim)
        value_states = value_states.view(batch_size, q_len, num_key_value_heads, head_dim)

        # Apply position embeddings

        constants = self.model.get_device_tensors(self.device_idx, scratch = False)

        ext_c.rope_(query_states, constants.sin, constants.cos, past_len, num_attention_heads, head_dim)
        ext_c.rope_(key_states, constants.sin, constants.cos, past_len, num_key_value_heads, head_dim)

        # Add keys and values to cache

        if cache is not None:

            new_keys = cache.key_states[self.layer_idx].narrow(1, past_len, q_len).narrow(0, 0, batch_size)
            new_values = cache.value_states[self.layer_idx].narrow(1, past_len, q_len).narrow(0, 0, batch_size)
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

        attn_output = attn_output.reshape(batch_size, q_len, hidden_size)
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


