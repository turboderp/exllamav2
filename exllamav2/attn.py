from __future__ import annotations
import torch
from torch import nn
from exllamav2.module import ExLlamaV2Module
from exllamav2.rmsnorm import ExLlamaV2RMSNorm
from exllamav2.layernorm import ExLlamaV2LayerNorm
from exllamav2.headnorm import ExLlamaV2HeadNorm
from exllamav2.linear import ExLlamaV2Linear
from exllamav2.cache import ExLlamaV2CacheBase
from exllamav2.ext import exllamav2_ext as ext_c, none_tensor
from exllamav2.compat import safe_move_tensor
from exllamav2.lora import ExLlamaV2Lora
import math
# import xformers.ops as xops
# from exllamav2.util import list_live_tensors, set_snapshot, diff_snapshot, print_vram_usage_peak
# import torch.nn.functional as F

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from exllamav2.model import ExLlamaV2

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

    name: str = "Attention"

    layer_idx: int
    input_layernorm: ExLlamaV2RMSNorm | ExLlamaV2LayerNorm | None
    q_proj: ExLlamaV2Linear | None
    k_proj: ExLlamaV2Linear | None
    v_proj: ExLlamaV2Linear | None
    o_proj: ExLlamaV2Linear | None
    q_norm: ExLlamaV2HeadNorm | None
    k_norm: ExLlamaV2HeadNorm | None

    q_handle: int | None

    temp_state: torch.tensor
    temp_q: torch.tensor
    temp_k: torch.tensor
    temp_v: torch.tensor
    temp_o: torch.tensor
    temp_dq: torch.tensor
    # temp_kv: torch.tensor

    temp_lora_size: int

    has_norm: bool
    has_residual: bool


    class Params:

        batch_size: int
        seq_len: int
        past_len: int | None
        past_lens: list[int] | None
        input_mask: torch.Tensor | None
        multi_cache: bool
        attn_mask: torch.Tensor | None
        attn_masks: torch.Tensor | None
        position_offsets: torch.Tensor | None
        past_lens_tensor: torch.Tensor | None

        def __init__(self,
                     batch_size: int,
                     seq_len: int,
                     past_len: int | list[int],
                     input_mask: torch.Tensor,
                     position_offsets: torch.Tensor):

            self.batch_size = batch_size
            self.seq_len = seq_len
            if isinstance(past_len, list):
                self.past_len = None
                self.past_lens = past_len
                self.multi_cache = True
            else:
                self.past_len = past_len
                self.past_lens = None
                self.multi_cache = False
            self.input_mask = input_mask

            self.attn_mask = None
            self.attn_masks = None

            self.position_offsets = position_offsets
            self.past_lens_tensor = None


        def is_causal(self) -> bool:

            return self.input_mask is None


        def get_position_offsets(self, device) -> torch.Tensor | None:

            assert self.position_offsets is not None
            if self.position_offsets.device != device:
                self.position_offsets = safe_move_tensor(self.position_offsets, device)
            return self.position_offsets


        def get_past_lens(self, device) -> torch.Tensor | None:

            assert self.past_lens is not None
            if self.past_lens_tensor is None:
                self.past_lens_tensor = torch.tensor(self.past_lens, dtype = torch.int, device = device)
            elif self.past_lens_tensor.device != device:
                self.past_lens_tensor = safe_move_tensor(self.past_lens_tensor, device)
            return self.past_lens_tensor


        def get_attn_mask(self, device) -> torch.Tensor | None:

            if self.attn_mask is None:
                self.attn_mask = self.build_attn_mask(device)
            elif self.attn_mask.device != device:
                self.attn_mask = safe_move_tensor(self.attn_mask, device)
            return self.attn_mask


        def get_attn_masks(self, device) -> torch.Tensor | None:

            if self.attn_masks is None:
                self.attn_masks = self.build_attn_masks(device)
            elif self.attn_masks[0] is not None and self.attn_masks[0].device != device:
                self.attn_masks = [(safe_move_tensor(m, device) if m is not None else None) for m in self.attn_masks]
            return self.attn_masks


        def build_single_attn_mask(self, batch_size, seq_len, past_len, device, input_mask):

            attn_mask = torch.zeros((batch_size, 1, seq_len, past_len + seq_len), dtype = torch.float16, device = device)
            attn_mask_triu = torch.triu(torch.full((seq_len - 1, seq_len - 1), -65504.0))
            attn_mask[:, :, : seq_len - 1, past_len + 1: past_len + seq_len] = attn_mask_triu

            if input_mask is not None:
                min_mask_width = min(input_mask.shape[-1], seq_len + past_len)
                input_mask_part = safe_move_tensor(input_mask[:, :min_mask_width], attn_mask.device)
                input_mask_part = input_mask_part.unsqueeze(1).unsqueeze(2)
                attn_mask[:, :, :, :min_mask_width] = torch.minimum(attn_mask[:, :, :, :min_mask_width], input_mask_part)

            return attn_mask


        def build_attn_mask(self, device) -> torch.Tensor | None:
            assert not self.multi_cache, "Building single mask for multiple caches"

            if self.input_mask is None and self.seq_len == 1: return None
            return self.build_single_attn_mask(self.batch_size, self.seq_len, self.past_len, device, self.input_mask)


        def build_attn_masks(self, device) -> torch.Tensor | None:
            assert self.multi_cache, "Building multiple masks for single cache"

            attn_masks = []
            for i, past_len in enumerate(self.past_lens):
                if self.input_mask is None and self.seq_len == 1:
                    attn_masks.append(None)
                else:
                    attn_masks.append(self.build_single_attn_mask(1, self.seq_len, past_len, device, self.input_mask[i]))
            return attn_masks


    def __init__(self,
                 model: ExLlamaV2,
                 key: str,
                 layer_idx: int,
                 has_norm: bool = True,
                 has_residual: bool = True):

        super().__init__(model, key)

        cfg = self.model.config

        self.layer_idx = layer_idx
        self.has_norm = has_norm
        self.has_residual = has_residual

        self.q_handle = None
        self.temp_lora_size = 0

        hidden_size = cfg.hidden_size

        if self.has_norm:
            if cfg.arch.norm == "layernorm":
                self.input_layernorm = ExLlamaV2LayerNorm(model, key + cfg.arch.norm_key_1)
            elif cfg.arch.norm == "rmsnorm":
                self.input_layernorm = ExLlamaV2RMSNorm(model, key + cfg.arch.norm_key_1)
        else:
            self.input_layernorm = None

        f_a = 0
        f_b = cfg.num_attention_heads * cfg.head_dim
        f_c = f_b + cfg.num_key_value_heads * cfg.head_dim
        f_d = f_c + cfg.num_key_value_heads * cfg.head_dim
        f_key = (key + ".self_attn." + cfg.arch.fused_qkv_key) if cfg.arch.fused_qkv_key else None

        self.q_proj = ExLlamaV2Linear(model, key + ".self_attn.q_proj", hidden_size, cfg.num_attention_heads * cfg.head_dim, cfg.arch.attention_bias_qkv, f_key = f_key, f_beg = f_a, f_end = f_b)
        self.k_proj = ExLlamaV2Linear(model, key + ".self_attn.k_proj", hidden_size, cfg.num_key_value_heads * cfg.head_dim, cfg.arch.attention_bias_qkv, f_key = f_key, f_beg = f_b, f_end = f_c)
        self.v_proj = ExLlamaV2Linear(model, key + ".self_attn.v_proj", hidden_size, cfg.num_key_value_heads * cfg.head_dim, cfg.arch.attention_bias_qkv, f_key = f_key, f_beg = f_c, f_end = f_d)
        self.o_proj = ExLlamaV2Linear(model, key + ".self_attn.o_proj", cfg.num_attention_heads * cfg.head_dim, hidden_size, cfg.arch.attention_bias_o)

        if cfg.use_qk_norm:
            self.q_norm = ExLlamaV2HeadNorm(model, key + ".self_attn.q_norm", cfg.num_attention_heads, cfg.head_dim)
            self.k_norm = ExLlamaV2HeadNorm(model, key + ".self_attn.k_norm", cfg.num_key_value_heads, cfg.head_dim)
        else:
            self.q_norm = None
            self.k_norm = None

        self.submodules = [self.q_proj,
                           self.k_proj,
                           self.v_proj,
                           self.o_proj]
        if self.has_norm:
            self.submodules += [self.input_layernorm]
        if cfg.use_qk_norm:
            self.submodules += [self.q_norm,
                                self.k_norm]


    def numel(self) -> int:

        numel = self.q_proj.numel() + \
                self.k_proj.numel() + \
                self.v_proj.numel() + \
                self.o_proj.numel()

        if self.input_layernorm is not None: numel += self.input_layernorm.numel()
        if self.q_norm is not None: numel += self.q_norm.numel()
        if self.k_norm is not None: numel += self.k_norm.numel()

        return numel


    def load(self):

        if self.input_layernorm is not None: self.input_layernorm.load()
        self.q_proj.load()
        self.k_proj.load()
        self.v_proj.load()
        self.o_proj.load()
        if self.q_norm is not None: self.q_norm.load()
        if self.k_norm is not None: self.k_norm.load()

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

            if self.has_norm:
                norm_weight = self.input_layernorm.weight if self.input_layernorm.weight is not None else none_tensor
                norm_bias = self.input_layernorm.bias if self.input_layernorm.bias is not None else none_tensor
                is_rms = isinstance(self.input_layernorm, ExLlamaV2RMSNorm)
                eps = self.input_layernorm.variance_epsilon
            else:
                norm_weight = none_tensor
                norm_bias = none_tensor
                is_rms = False
                eps = 0

            if self.q_norm is None:
                q_norm = none_tensor
            else:
                q_norm = self.q_norm.weight

            if self.k_norm is None:
                k_norm = none_tensor
            else:
                k_norm = self.k_norm.weight

            self.q_handle = ext_c.make_q_attn(norm_weight,
                                              norm_bias,
                                              is_rms,
                                              eps,
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
                                              self.model.config.max_seq_len,
                                              self.has_residual,
                                              self.model.config.arch.rope_neox_style,
                                              q_norm,
                                              k_norm)


    def unload(self):
        if self.q_handle is not None:
            ext_c.free_q_attn(self.q_handle)
            self.q_handle = None

        if self.input_layernorm is not None: self.input_layernorm.unload()
        if self.q_proj is not None: self.q_proj.unload()
        if self.k_proj is not None: self.k_proj.unload()
        if self.v_proj is not None: self.v_proj.unload()
        self.o_proj.unload()

        self.temp_state = None
        self.temp_dq = None

        if self.q_norm is not None: self.q_norm.unload()
        if self.k_norm is not None: self.k_norm.unload()


    def weight_footprint(self):

        fp = self.q_proj.weight_footprint() + \
             self.k_proj.weight_footprint() + \
             self.v_proj.weight_footprint() + \
             self.o_proj.weight_footprint()
        if self.input_layernorm is not None:
            fp += self.input_layernorm.weight_footprint()
        if self.q_norm is not None:
            fp += self.q_norm.weight_footprint()
        if self.k_norm is not None:
            fp += self.k_norm.weight_footprint()

        return fp


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

        return self.model.config.max_input_len * self.model.config.max_batch_size * self.model.config.num_attention_heads * self.model.config.head_dim * 2 + 128


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

        if self.input_layernorm is not None: self.input_layernorm.set_device_idx(idx)
        self.q_proj.set_device_idx(idx)
        self.k_proj.set_device_idx(idx)
        self.v_proj.set_device_idx(idx)
        self.o_proj.set_device_idx(idx)
        if self.q_norm is not None: self.q_norm.set_device_idx(idx)
        if self.k_norm is not None: self.k_norm.set_device_idx(idx)


    def repeat_kv(self, hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:

        if n_rep == 1: return hidden_states

        batch, num_key_value_heads, slen, head_dim = hidden_states.shape
        hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        hidden_states = hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
        return hidden_states


    def forward(self,
                hidden_states: torch.Tensor,
                cache: ExLlamaV2CacheBase | None = None,
                attn_params: ExLlamaV2Attention.Params | None = None,
                past_len: int | None = None,
                intermediates: bool = False,
                loras: list[ExLlamaV2Lora] | None = None,
                **kwargs) -> torch.Tensor | dict[str: torch.Tensor]:

        global has_flash_attn

        if self.q_handle is None or intermediates:
            return self.forward_torch(hidden_states,
                                      cache,
                                      attn_params,
                                      past_len,
                                      intermediates,
                                      loras = loras,
                                      **kwargs)

        batch_size = hidden_states.shape[0]
        q_len = hidden_states.shape[1]

        cfg = self.model.config

        direct = (batch_size == 1 and cache is not None and isinstance(cache, ExLlamaV2CacheBase))

        # past_len = 0
        # if cache is not None:
        #     if isinstance(cache, ExLlamaV2Cache):
        #         past_len = cache.current_seq_len
        #     if isinstance(cache, list):
        #         past_len = [c.current_seq_len for c in cache]

        num_attention_heads = cfg.num_attention_heads
        num_key_value_heads = cfg.num_key_value_heads
        num_key_value_groups = cfg.num_key_value_groups
        head_dim = cfg.head_dim
        hidden_size = cfg.hidden_size

        constants = self.model.get_device_tensors(self.device_idx)

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
            pass_lora_temp = none_tensor
        else:
            pass_loras = [id(x) for x in loras]
            pass_lora_temp = torch.empty((self.temp_lora_size,), dtype = torch.half, device = hidden_states.device)

        if attn_params.multi_cache:
            pass_past_len_1 = -1
            pass_past_len_2 = attn_params.get_past_lens(hidden_states.device)
        elif attn_params.position_offsets is not None:
            pass_past_len_1 = past_len
            pass_past_len_2 = attn_params.get_position_offsets(hidden_states.device)
        else:
            pass_past_len_1 = past_len
            pass_past_len_2 = none_tensor

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

                    k_states = batch_keys.narrow(0, 0, batch_size).narrow(1, 0, past_len + q_len)
                    v_states = batch_values.narrow(0, 0, batch_size).narrow(1, 0, past_len + q_len)

            # Torch matmul attention

            if cfg.no_flash_attn or not has_flash_attn or not attn_params.is_causal():

                q_states = q_states.transpose(1, 2)
                k_states = k_states.transpose(1, 2)
                v_states = v_states.transpose(1, 2)

                k_states = self.repeat_kv(k_states, num_key_value_groups)
                k_states = k_states.transpose(-1, -2)

                attn_weights = torch.matmul(q_states, k_states)
                k_states = None
                q_states = None

                attn_weights /= math.sqrt(head_dim)
                attn_mask = attn_params.get_attn_mask(hidden_states.device)
                if attn_mask is not None: attn_weights = attn_weights + attn_mask
                attn_weights = nn.functional.softmax(attn_weights, dim = -1, dtype = torch.float16)

                v_states = self.repeat_kv(v_states, num_key_value_groups)
                attn_output = torch.matmul(attn_weights, v_states)
                v_states = None

                attn_output = attn_output.transpose(1, 2)
                attn_output = attn_output.reshape((batch_size, q_len, cfg.num_attention_heads * cfg.head_dim))

            # Flash Attention 2

            else:

                # TODO: Enable flash-attn with input mask
                attn_output = flash_attn_func(q_states, k_states, v_states, causal = True)
                attn_output = attn_output.reshape((batch_size, q_len, cfg.num_attention_heads * cfg.head_dim))

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

            # Update 8-bit/Q4 cache

            if cache is not None:
                cache.store_kv_state(self.layer_idx, batch_size, past_len, q_len)

        # Multiple caches

        elif isinstance(cache, list):

            assert attn_params.multi_cache
            attn_masks = attn_params.get_attn_masks(hidden_states.device)

            attn_outputs = []
            for i in range(len(cache)):

                # TODO: Once nested tensors are finalized in Torch, this could all be batched, probably

                # Add keys and values to cache

                batch_keys, batch_values = cache[i].get_kv_state(self.layer_idx, 1, 0, past_len[i])
                new_keys = batch_keys.narrow(1, past_len[i], q_len)
                new_values = batch_values.narrow(1, past_len[i], q_len)
                new_keys.copy_(k_states.narrow(0, i, 1))
                new_values.copy_(v_states.narrow(0, i, 1))

                # Store updated cache values

                cache[i].store_kv_state(self.layer_idx, 1, past_len[i], q_len)

                # Key/value tensors with past

                k_states_b = batch_keys.narrow(1, 0, past_len[i] + q_len)
                v_states_b = batch_values.narrow(1, 0, past_len[i] + q_len)

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
                if attn_masks[i] is not None: attn_weights = attn_weights + attn_masks[i]
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


    def forward_torch(self,
                      hidden_states: torch.Tensor,
                      cache: ExLlamaV2CacheBase | None = None,
                      attn_params: ExLlamaV2Attention.Params | None = None,
                      past_len: int | None = None,
                      intermediates: bool = False,
                      loras: list[ExLlamaV2Lora] | None = None,
                      **kwargs) -> torch.Tensor | dict:

        cfg = self.model.config
        num_attention_heads = cfg.num_attention_heads
        num_key_value_heads = cfg.num_key_value_heads
        num_key_value_groups = cfg.num_key_value_groups
        head_dim = cfg.head_dim
        hidden_size = cfg.hidden_size

        batch_size, q_len, _ = hidden_states.size()

        past_len = 0 if cache is None else cache.current_seq_len

        # Project q, k, v

        residual = hidden_states
        post_norm = self.input_layernorm.forward(hidden_states) if self.has_norm else hidden_states

        query_states = self.q_proj.forward(post_norm, loras = loras)
        key_states = self.k_proj.forward(post_norm, loras = loras)
        value_states = self.v_proj.forward(post_norm, loras = loras)

        # Shape for attention

        query_states = query_states.view(batch_size, q_len, num_attention_heads, head_dim)
        key_states = key_states.view(batch_size, q_len, num_key_value_heads, head_dim)
        value_states = value_states.view(batch_size, q_len, num_key_value_heads, head_dim)

        # Apply Q/K norms

        if cfg.use_qk_norm:
            query_states = self.q_norm.forward(query_states)
            key_states = self.k_norm.forward(key_states)

        # Apply position embeddings

        constants = self.model.get_device_tensors(self.device_idx, scratch = False)

        if attn_params.position_offsets is not None:
            position_offsets = attn_params.get_position_offsets(hidden_states.device)
        else:
            position_offsets = none_tensor

        ext_c.rope_(query_states, constants.sin, constants.cos, past_len, num_attention_heads, head_dim, position_offsets, self.model.config.arch.rope_neox_style)
        ext_c.rope_(key_states, constants.sin, constants.cos, past_len, num_key_value_heads, head_dim, position_offsets, self.model.config.arch.rope_neox_style)

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

        if cfg.no_flash_attn or not has_flash_attn or not attn_params.is_causal():

            query_states = query_states.transpose(1, 2)
            key_states = key_states.transpose(1, 2)
            value_states = value_states.transpose(1, 2)

            key_states = self.repeat_kv(key_states, cfg.num_key_value_groups)
            key_states = key_states.transpose(-1, -2)

            attn_weights = torch.matmul(query_states, key_states)
            attn_weights /= math.sqrt(head_dim)
            attn_mask = attn_params.get_attn_mask(hidden_states.device)
            if attn_mask is not None: attn_weights = attn_weights + attn_mask
            attn_weights = nn.functional.softmax(attn_weights, dim = -1, dtype = torch.float16)

            value_states = self.repeat_kv(value_states, cfg.num_key_value_groups)
            attn_output = torch.matmul(attn_weights, value_states)

            attn_output = attn_output.transpose(1, 2)
            attn_output = attn_output.reshape((batch_size, q_len, cfg.num_attention_heads * cfg.head_dim))

        # Flash Attention 2

        else:

            attn_output = flash_attn_func(query_states, key_states, value_states, causal = True)
            attn_output = attn_output.reshape((batch_size, q_len, cfg.num_attention_heads * cfg.head_dim))

        # Update 8-bit/Q4 cache

        if cache is not None:
            cache.store_kv_state(self.layer_idx, batch_size, past_len, q_len)

        # Output projection

        attn_proj = self.o_proj.forward(attn_output, loras = loras)

        # Add residual connection

        hidden_states = (attn_proj + residual) if self.has_residual else attn_proj

        if intermediates:
            return {"post_norm": post_norm,
                    "attn_output": attn_output,
                    "hidden_states": hidden_states}
        else:
            return hidden_states


    def update_loras(self):

        if self.q_handle is None: return

        cfg = self.model.config

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

        self.temp_lora_size = temp_lora_size * cfg.max_batch_size * cfg.max_input_len


    def is_quant(self):
        return self.q_handle is not None

