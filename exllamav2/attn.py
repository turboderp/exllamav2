import torch
from torch import nn
import torch.nn.functional as F
from exllamav2.module import ExLlamaV2Module
from exllamav2.rmsnorm import ExLlamaV2RMSNorm
from exllamav2.linear import ExLlamaV2Linear
from exllamav2.cache import ExLlamaV2CacheBase
from exllamav2.embedding import ExLlamaV2Embedding
import math
from exllamav2 import ext
from exllamav2.ext import exllamav2_ext as ext_c
from exllamav2.quip_linear import QuipLinear
# import xformers.ops as xops
# from exllamav2.util import list_live_tensors, set_snapshot, diff_snapshot, print_vram_usage_peak

# Detect flash-attn

has_flash_attn = False
try:
    import flash_attn
    flash_attn_ver = [int(t) for t in flash_attn.__version__.split(".") if t.isdigit()]
    is_ampere_or_newer_gpu = any(torch.cuda.get_device_properties(i).major >= 8 for i in range(torch.cuda.device_count()))
    
    if flash_attn_ver >= [2, 2, 1] and is_ampere_or_newer_gpu:
        from flash_attn import flash_attn_func
        has_flash_attn = True
except ModuleNotFoundError:
    pass

class ExLlamaV2Attention(ExLlamaV2Module):

    layer_idx: int
    input_layernorm: ExLlamaV2RMSNorm or None
    q_proj: ExLlamaV2Linear or None
    k_proj: ExLlamaV2Linear or None
    v_proj: ExLlamaV2Linear or None
    o_proj: ExLlamaV2Linear
    qkv_proj: ExLlamaV2Linear or None
    k_scale: torch.tensor or None
    o_scale: torch.tensor or None
    q_scale: torch.tensor or None
    v_scale: torch.tensor or None

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

    temp_lora_size: int = 0


    def __init__(self, model, key, layer_idx):
        super().__init__(model, key)

        self.layer_idx = layer_idx

        hidden_size = self.model.config.hidden_size

        self.input_layernorm = ExLlamaV2RMSNorm(model, key + ".input_layernorm")
        self.submodules = [self.input_layernorm]

        if model.config.is_quip:
            self.qkv_proj = QuipLinear(model,
                                       key + ".self_attn.qkv_proj",
                                       hidden_size,
                                       (self.model.config.num_attention_heads * self.model.config.head_dim) +
                                       (self.model.config.num_key_value_heads * self.model.config.head_dim) +
                                       (self.model.config.num_key_value_heads * self.model.config.head_dim))

            self.o_proj = QuipLinear(model, key + ".self_attn.o_proj",
                                     self.model.config.num_attention_heads * self.model.config.head_dim,
                                     hidden_size)
            self.submodules += [self.qkv_proj, self.o_proj]
        else:
            self.q_proj = ExLlamaV2Linear(model, key + ".self_attn.q_proj", hidden_size, self.model.config.num_attention_heads * self.model.config.head_dim, False)
            self.k_proj = ExLlamaV2Linear(model, key + ".self_attn.k_proj", hidden_size, self.model.config.num_key_value_heads * self.model.config.head_dim, False)
            self.v_proj = ExLlamaV2Linear(model, key + ".self_attn.v_proj", hidden_size, self.model.config.num_key_value_heads * self.model.config.head_dim, False)
            self.o_proj = ExLlamaV2Linear(model, key + ".self_attn.o_proj", self.model.config.num_attention_heads * self.model.config.head_dim, hidden_size, False)
            self.submodules += [self.q_proj, self.k_proj, self.v_proj, self.o_proj]


    def load(self):
        if self.model.config.is_quip:
            w = self.load_weight()
            self.k_scale = w['k_scale'].to(self.device_idx)
            self.o_scale = w['o_scale'].to(self.device_idx)
            self.q_scale = w['q_scale'].to(self.device_idx)
            self.v_scale = w['v_scale'].to(self.device_idx)

        qkv_embed = self.model.config.qkv_embed and self.layer_idx == 0

        if hasattr(self, 'input_layernorm') and self.input_layernorm is not None: self.input_layernorm.load()
        if hasattr(self, 'q_proj') and self.q_proj is not None: self.q_proj.load()
        if hasattr(self, 'k_proj') and self.k_proj is not None: self.k_proj.load()
        if hasattr(self, 'v_proj') and self.v_proj is not None: self.v_proj.load()
        if hasattr(self, 'qkv_proj') and self.qkv_proj is not None: self.qkv_proj.load()
        self.o_proj.load()

        if hasattr(self, 'q_proj') and self.q_proj is not None and self.q_proj.is_quant():

            assert self.k_proj.is_quant() and self.v_proj.is_quant() and self.o_proj.is_quant(), "Partially quantized attention layer"

            device_tensors = self.model.get_device_tensors(self.device_idx)
            device_tensors.begin_scratch_alloc()
            self.temp_state = device_tensors.get_scratch_slice(self.temp_state_size())
            # self.temp_q = device_tensors.get_scratch_slice(self.temp_q_size())
            # self.temp_k = device_tensors.get_scratch_slice(self.temp_k_size())
            # self.temp_v = device_tensors.get_scratch_slice(self.temp_v_size())
            self.temp_dq = device_tensors.get_scratch_slice(self.temp_dq_size())
            # self.temp_kv = device_tensors.get_scratch_slice(self.temp_kv_size()) if self.model.config.num_attention_heads != self.model.config.num_key_value_heads else None

            self.q_handle = ext_c.make_q_attn(self.input_layernorm.weight if not qkv_embed else ext.none_tensor,
                                              self.input_layernorm.variance_epsilon if not qkv_embed else 0.0,
                                              self.q_proj.q_handle if not qkv_embed else 0,
                                              self.k_proj.q_handle if not qkv_embed else 0,
                                              self.v_proj.q_handle if not qkv_embed else 0,
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

        if qkv_embed:

            embedding = self.model.modules[0]
            assert isinstance(embedding, ExLlamaV2Embedding)
            q = self.q_proj.get_weight_tensor_dq() if hasattr(self, 'q_proj') and self.q_proj is not None else None
            k = self.k_proj.get_weight_tensor_dq() if hasattr(self, 'k_proj') and self.k_proj is not None else None
            v = self.v_proj.get_weight_tensor_dq() if hasattr(self, 'v_proj') and self.v_proj is not None else None
            norm = self.input_layernorm
            embedding.make_qkv(norm, q, k, v)

            if hasattr(self, 'q_proj') and self.q_proj is not None: self.q_proj.unload(); self.q_proj = None
            if hasattr(self, 'k_proj') and self.v_proj is not None: self.k_proj.unload(); self.k_proj = None
            if hasattr(self, 'v_proj') and self.v_proj is not None: self.v_proj.unload(); self.v_proj = None
            self.input_layernorm.unload(); self.input_layernorm = None


    def unload(self):
        if self.q_handle is not None:
            ext_c.free_q_attn(self.q_handle)
            self.q_handle = None

        if hasattr(self, 'qkv_proj') and self.qkv_proj is not None: self.qkv_proj.unload()
        if hasattr(self, 'input_layernorm') and self.input_layernorm is not None: self.input_layernorm.unload()
        if hasattr(self, 'q_proj') and self.q_proj is not None: self.q_proj.unload()
        if hasattr(self, 'k_proj') and self.k_proj is not None: self.k_proj.unload()
        if hasattr(self, 'v_proj') and self.v_proj is not None: self.v_proj.unload()
        self.o_proj.unload()


    def weight_footprint(self, qkv_embed = False):

        if self.layer_idx == 0 and self.model.config.qkv_embed:

            return self.o_proj.weight_footprint()

        else:

            return self.input_layernorm.weight_footprint() + \
                   self.q_proj.weight_footprint() if hasattr(self, 'q_proj') and self.q_proj is not None else 0 + \
                   self.k_proj.weight_footprint() if hasattr(self, 'k_proj') and self.k_proj is not None else 0 + \
                   self.v_proj.weight_footprint() if hasattr(self, 'v_proj') and self.v_proj is not None else 0 + \
                   self.qkv_proj.weight_footprint() if hasattr(self, 'qkv_proj') and self.qkv_proj is not None else 0 + \
                   self.o_proj.weight_footprint()


    def scratch_space_fixed(self):

        return self.temp_state_size() + \
               self.temp_dq_size()


    def scratch_space(self):

        return self.temp_state_size() + \
               self.temp_q_size() + \
               self.temp_k_size() + \
               self.temp_v_size() + \
               self.temp_dq_size() + \
               self.temp_kv_size()
               # self.temp_attn_size() +  # Accounted for separately in model.set_device_map()


    def temp_state_size(self):

        return self.model.config.max_input_len * self.model.config.max_batch_size * self.model.config.hidden_size * 2 + 128


    def temp_q_size(self):

        return self.model.config.max_input_len * self.model.config.max_batch_size * self.model.config.num_attention_heads * self.model.config.head_dim * 2 + 128


    def temp_k_size(self):

        return self.model.config.max_input_len * self.model.config.max_batch_size * self.model.config.num_key_value_heads * self.model.config.head_dim * 2 + 128


    def temp_v_size(self):

        return self.model.config.max_input_len * self.model.config.max_batch_size * self.model.config.num_key_value_heads * self.model.config.head_dim * 2 + 128


    def temp_dq_size(self):
        return max(self.q_proj.temp_dq_size() if hasattr(self, 'q_proj') and self.q_proj is not None else 0,
                   self.k_proj.temp_dq_size() if hasattr(self, 'k_proj') and self.k_proj is not None else 0,
                   self.v_proj.temp_dq_size() if hasattr(self, 'v_proj') and self.v_proj is not None else 0,
                   self.qkv_proj.temp_dq_size() if hasattr(self, 'qkv_proj') and self.qkv_proj is not None else 0,
                   self.o_proj.temp_dq_size())


    def temp_kv_size(self):

        if self.model.config.num_key_value_heads == self.model.config.num_attention_heads: return 0
        return 2 * self.model.config.max_seq_len * self.model.config.max_batch_size * self.model.config.num_attention_heads * self.model.config.head_dim * 2 + 128


    def temp_attn_size(self):
        global has_flash_attn

        att_max = min(self.model.config.max_attention_size, self.model.config.max_seq_len ** 2)

        if has_flash_attn and not self.model.config.no_flash_attn:
            eff = self.model.config.max_attention_size ** 0.5 / 190  # based on supposed memory savings listed in flash-attn repo + some fudging
            att_max //= eff

        return 2 * att_max * self.model.config.num_attention_heads * 2 + 128


    def set_device_idx(self, idx):
        super().set_device_idx(idx)

        self.input_layernorm.set_device_idx(idx)
        if hasattr(self, 'q_proj') and self.q_proj is not None: self.q_proj.set_device_idx(idx)
        if hasattr(self, 'k_proj') and self.k_proj is not None: self.k_proj.set_device_idx(idx)
        if hasattr(self, 'v_proj') and self.v_proj is not None: self.v_proj.set_device_idx(idx)
        if hasattr(self, 'qkv_proj') and self.qkv_proj is not None: self.qkv_proj.set_device_idx(idx)
        self.o_proj.set_device_idx(idx)


    def repeat_kv(self, hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:

        if n_rep == 1: return hidden_states

        batch, num_key_value_heads, slen, head_dim = hidden_states.shape
        hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        hidden_states = hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
        return hidden_states


    def forward(self, hidden_states, cache = None, attn_mask = None, past_len = None, intermediates = False, loras = None, position_offsets = None):
        global has_flash_attn

        qkv_embed = self.model.config.qkv_embed and self.layer_idx == 0

        if self.q_handle is None or intermediates:
            return self.forward_torch(hidden_states, cache, attn_mask, past_len, intermediates, loras = loras, position_offsets = position_offsets)

        if qkv_embed:
            batch_size = hidden_states[0].shape[0]
            q_len = hidden_states[0].shape[1]
        else:
            batch_size = hidden_states.shape[0]
            q_len = hidden_states.shape[1]

        direct = (batch_size == 1 and cache is not None and isinstance(cache, ExLlamaV2CacheBase)) and not qkv_embed

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

        if not qkv_embed:

            q_shape = hidden_states.shape[:-1] + (self.q_proj.out_features,)
            k_shape = hidden_states.shape[:-1] + (self.k_proj.out_features,)
            v_shape = hidden_states.shape[:-1] + (self.v_proj.out_features,)
            q_states = torch.empty(q_shape, device = hidden_states.device, dtype = torch.half)

            # If conditions are right we can write the K/V projections directly into the cache

            if direct:

                batch_keys, batch_values = cache.get_kv_state(self.layer_idx, batch_size, 0, past_len)
                k_states = batch_keys.narrow(0, 0, batch_size).narrow(1, past_len, q_len)
                v_states = batch_values.narrow(0, 0, batch_size).narrow(1, past_len, q_len)

            else:

                k_states = torch.empty(k_shape, device = hidden_states.device, dtype = torch.half)
                v_states = torch.empty(v_shape, device = hidden_states.device, dtype = torch.half)

            # RMS norm, Q/K/V projections, position embeddings

            if loras is None or self.temp_lora_size == 0:
                pass_loras = []
                pass_lora_temp = ext.none_tensor
            else:
                pass_loras = [id(x) for x in loras]
                pass_lora_temp = torch.empty((self.temp_lora_size,), dtype = torch.half, device = hidden_states.device)

            if isinstance(past_len, tuple):
                pass_past_len_1 = -1
                pass_past_len_2 = past_len[0]
            elif position_offsets is not None:
                pass_past_len_1 = past_len
                pass_past_len_2 = position_offsets
            else:
                pass_past_len_1 = past_len
                pass_past_len_2 = ext.none_tensor

            ext_c.q_attn_forward_1(self.q_handle,
                                   hidden_states,
                                   batch_size,
                                   q_len,
                                   pass_past_len_1,
                                   pass_past_len_2,
                                   q_states,
                                   k_states,
                                   v_states,
                                   constants.sin,
                                   constants.cos,
                                   pass_loras,
                                   pass_lora_temp)

        # Alternative, for embedded QKV

        else:

            q_states = hidden_states[1]
            k_states = hidden_states[2]
            v_states = hidden_states[3]
            hidden_states = hidden_states[0]

            offset_tensor = position_offsets if position_offsets is not None else ext.none_tensor
            ext_c.rope_(q_states, constants.sin, constants.cos, past_len, num_attention_heads, head_dim, offset_tensor)
            ext_c.rope_(k_states, constants.sin, constants.cos, past_len, num_key_value_heads, head_dim, offset_tensor)

        # Shape for attention

        q_states = q_states.view(batch_size, q_len, num_attention_heads, head_dim)
        k_states = k_states.view(batch_size, q_len, num_key_value_heads, head_dim)
        v_states = v_states.view(batch_size, q_len, num_key_value_heads, head_dim)

        # Regular (batched) attention with optional padding mask

        if cache is None or isinstance(cache, ExLlamaV2CacheBase):

            # Add keys and values to cache

            if cache is not None:

                if direct:

                    k_states = batch_keys.narrow(0, 0, batch_size).narrow(1, 0, past_len + q_len)
                    v_states = batch_values.narrow(0, 0, batch_size).narrow(1, 0, past_len + q_len)

                else:

                    batch_keys, batch_values = cache.get_kv_state(self.layer_idx, batch_size, 0, past_len)
                    new_keys = batch_keys.narrow(0, 0, batch_size).narrow(1, past_len, q_len)
                    new_values = batch_values.narrow(0, 0, batch_size).narrow(1, past_len, q_len)
                    new_keys.copy_(k_states)
                    new_values.copy_(v_states)

                    # Key/value tensors with past

                    k_states = batch_keys.narrow(1, 0, past_len + q_len)
                    v_states = batch_values.narrow(1, 0, past_len + q_len)

            # Torch matmul attention

            if self.model.config.no_flash_attn or not has_flash_attn:

                q_states = q_states.transpose(1, 2)
                k_states = k_states.transpose(1, 2)
                v_states = v_states.transpose(1, 2)

                k_states = self.repeat_kv(k_states, num_key_value_groups)
                k_states = k_states.transpose(-1, -2)

                attn_weights = torch.matmul(q_states, k_states)
                k_states = None
                q_states = None

                attn_weights /= math.sqrt(head_dim)
                if attn_mask is not None: attn_weights = attn_weights + attn_mask
                attn_weights = nn.functional.softmax(attn_weights, dim = -1, dtype = torch.float16)

                v_states = self.repeat_kv(v_states, num_key_value_groups)
                attn_output = torch.matmul(attn_weights, v_states)
                v_states = None

                attn_output = attn_output.transpose(1, 2)
                attn_output = attn_output.reshape((batch_size, q_len, hidden_size))

            # Flash Attention 2

            else:

                attn_output = flash_attn_func(q_states, k_states, v_states, causal = True)
                attn_output = attn_output.reshape((batch_size, q_len, hidden_size))

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

            # Update 8-bit cache

            if cache is not None:
                cache.store_kv_state(self.layer_idx, batch_size, past_len, q_len)

        # Multiple caches

        else:

            attn_outputs = []
            for i in range(len(cache)):

                # TODO: Once nested tensors are finalized in Torch, this could all be batched, probably

                # Add keys and values to cache

                batch_keys, batch_values = cache[i].get_kv_state(self.layer_idx, batch_size, 0, past_len)
                new_keys = batch_keys.narrow(1, past_len[1][i], q_len)
                new_values = batch_values.narrow(1, past_len[1][i], q_len)
                new_keys.copy_(k_states.narrow(0, i, 1))
                new_values.copy_(v_states.narrow(0, i, 1))

                # Key/value tensors with past

                k_states_b = batch_keys.narrow(1, 0, past_len[1][i] + q_len)
                v_states_b = batch_values.narrow(1, 0, past_len[1][i] + q_len)

                # Torch matmul attention

                # TODO: enable flash-attn

                q_states_b = q_states.transpose(1, 2).narrow(0, i, 1)
                k_states_b = k_states_b.transpose(1, 2)
                v_states_b = v_states_b.transpose(1, 2)

                k_states_b = self.repeat_kv(k_states_b, num_key_value_groups)
                k_states_b = k_states_b.transpose(-1, -2)

                attn_weights = torch.matmul(q_states_b, k_states_b)
                q_states_b = None
                k_states_b = None

                attn_weights /= math.sqrt(head_dim)
                if attn_mask is not None: attn_weights = attn_weights + attn_mask[i]
                attn_weights = nn.functional.softmax(attn_weights, dim = -1, dtype = torch.float16)

                v_states_b = self.repeat_kv(v_states_b, num_key_value_groups)
                attn_output_b = torch.matmul(attn_weights, v_states_b)
                v_states_b = None

                attn_outputs.append(attn_output_b)

            q_states = None
            k_states = None
            v_states = None

            attn_output = torch.cat(attn_outputs, dim = 0)
            attn_output = attn_output.transpose(1, 2)
            attn_output = attn_output.reshape((batch_size, q_len, hidden_size))

        # Output projection

        ext_c.q_attn_forward_2(self.q_handle,
                               hidden_states,
                               attn_output,
                               batch_size,
                               q_len,
                               pass_loras,
                               pass_lora_temp)

        attn_output = None
        attn_weights = None

        return hidden_states


    def forward_torch(self, hidden_states, cache = None, attn_mask = None, past_len = None, intermediates = False, loras = None, position_offsets = None):

        num_attention_heads = self.model.config.num_attention_heads
        num_key_value_heads = self.model.config.num_key_value_heads
        num_key_value_groups = self.model.config.num_key_value_groups
        head_dim = self.model.config.head_dim
        hidden_size = self.model.config.hidden_size

        qkv_embed = self.model.config.qkv_embed and self.layer_idx == 0

        if not qkv_embed: batch_size, q_len, _ = hidden_states.size()
        else: batch_size, q_len, _ = hidden_states[0].size()

        past_len = 0 if cache is None else cache.current_seq_len

        # Project q, k, v

        if not qkv_embed:

            residual = hidden_states
            post_norm = self.input_layernorm.forward(hidden_states)

            if self.model.config.is_quip:
                qkv_states = self.qkv_proj.forward(post_norm.to(torch.float32), loras = loras)
                query_states = self.q_scale * qkv_states[..., 0:(num_attention_heads * head_dim)]
                key_states = self.k_scale * qkv_states[..., (
                    num_attention_heads * head_dim):(
                        (num_attention_heads * head_dim) +
                        (num_key_value_heads * head_dim))]
                value_states = self.v_scale * qkv_states[..., (
                    (num_attention_heads * head_dim) +
                    (num_key_value_heads * head_dim)):(
                        (num_attention_heads * head_dim) +
                        (num_key_value_heads * head_dim) +
                        (num_key_value_heads * head_dim))]
            else:
                query_states_im = self.q_proj.forward(post_norm, loras = loras)
                key_states_im = self.k_proj.forward(post_norm, loras = loras)
                value_states_im = self.v_proj.forward(post_norm, loras = loras)

                if intermediates:

                    query_states = query_states_im.clone()
                    key_states = key_states_im.clone()
                    value_states = value_states_im.clone()

                else:

                    query_states = query_states_im
                    key_states = key_states_im
                    value_states = value_states_im



        # Alternative, for embedded QKV

        else:

            residual = hidden_states[0]
            query_states = hidden_states[1]
            key_states = hidden_states[2]
            value_states = hidden_states[3]

        # Apply position embeddings

        query_states = query_states.view(batch_size, q_len, num_attention_heads, head_dim)
        key_states = key_states.view(batch_size, q_len, num_key_value_heads, head_dim)
        value_states = value_states.view(batch_size, q_len, num_key_value_heads, head_dim)

        constants = self.model.get_device_tensors(self.device_idx, scratch = False)

        offset_tensor = position_offsets if position_offsets is not None else ext.none_tensor
        ext_c.rope_(query_states, constants.sin, constants.cos, past_len, num_attention_heads, head_dim, offset_tensor)
        ext_c.rope_(key_states, constants.sin, constants.cos, past_len, num_key_value_heads, head_dim, offset_tensor)

        # Add keys and values to cache

        if cache is not None:

            batch_keys, batch_values = cache.get_kv_state(self.layer_idx, batch_size, 0, past_len)
            new_keys = batch_keys.narrow(1, past_len, q_len).narrow(0, 0, batch_size)
            new_values = batch_values.narrow(1, past_len, q_len).narrow(0, 0, batch_size)
            new_keys.copy_(key_states)
            new_values.copy_(value_states)

            # Key/value tensors with past

            key_states = batch_keys.narrow(1, 0, past_len + q_len).narrow(0, 0, batch_size)
            value_states = batch_values.narrow(1, 0, past_len + q_len).narrow(0, 0, batch_size)

        # Torch matmul attention

        if self.model.config.no_flash_attn or not has_flash_attn:

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
            attn_output = attn_output.reshape((batch_size, q_len, hidden_size))

        # Flash Attention 2

        else:

            attn_output = flash_attn_func(query_states, key_states, value_states, causal = True)
            attn_output = attn_output.reshape((batch_size, q_len, hidden_size))

        # Update 8-bit cache
        # TODO: Only update changed positions of the cache

        if cache is not None:
            cache.store_kv_state(self.layer_idx, batch_size, past_len, q_len)

        # Output projection

        attn_proj = self.o_proj.forward(attn_output, loras = loras)

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


    def update_loras(self):

        if self.q_handle is None: return

        q_proj_lora_a = { id(k): v for k, v in self.q_proj.lora_a_tensors.items() }
        q_proj_lora_b = { id(k): v for k, v in self.q_proj.lora_b_tensors.items() }
        k_proj_lora_a = { id(k): v for k, v in self.k_proj.lora_a_tensors.items() }
        k_proj_lora_b = { id(k): v for k, v in self.k_proj.lora_b_tensors.items() }
        v_proj_lora_a = { id(k): v for k, v in self.v_proj.lora_a_tensors.items() }
        v_proj_lora_b = { id(k): v for k, v in self.v_proj.lora_b_tensors.items() }
        o_proj_lora_a = { id(k): v for k, v in self.o_proj.lora_a_tensors.items() }
        o_proj_lora_b = { id(k): v for k, v in self.o_proj.lora_b_tensors.items() }

        temp_lora_size = ext_c.q_attn_set_loras(self.q_handle,
                                                q_proj_lora_a,
                                                q_proj_lora_b,
                                                k_proj_lora_a,
                                                k_proj_lora_b,
                                                v_proj_lora_a,
                                                v_proj_lora_b,
                                                o_proj_lora_a,
                                                o_proj_lora_b)

        self.temp_lora_size = temp_lora_size * self.model.config.max_batch_size * self.model.config.max_input_len


    def is_quant(self):
        return self.q_handle is not None

