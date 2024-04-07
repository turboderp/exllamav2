from __future__ import annotations
import os, sys
min_version = (3, 8)
if sys.version_info < min_version:
    print("")
    print(f" ## Warning: this project requires Python {min_version[0]}.{min_version[1]} or higher.")
    print("")

# Set CUDA context to lazy loading since we won't need 95% of the modules in Torch
os.environ["CUDA_MODULE_LOADING"] = "LAZY"

# # Set cudaMallocAsync allocator by default as it appears slightly more memory efficient, unless Torch is already
# # imported in which case changing the allocator would cause it to crash
# if not "PYTORCH_CUDA_ALLOC_CONF" in os.environ:
#     try:
#         x = torch.__version__
#         # TODO: Should maybe be a warning here?
#     except NameError:
#         os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "backend:cudaMallocAsync"

import torch
import math
from exllamav2.config import ExLlamaV2Config
from exllamav2.cache import ExLlamaV2CacheBase
from exllamav2.linear import ExLlamaV2Linear
from exllamav2.module import ExLlamaV2Module
from exllamav2.rmsnorm import ExLlamaV2RMSNorm
from exllamav2.layernorm import ExLlamaV2LayerNorm
from exllamav2.attn import ExLlamaV2Attention
from exllamav2.lora import ExLlamaV2Lora
from exllamav2.mlp import ExLlamaV2MLP
from exllamav2.moe_mlp import ExLlamaV2MoEMLP
from exllamav2.parallel_decoder import ExLlamaV2ParallelDecoder
from exllamav2.embedding import ExLlamaV2Embedding
from exllamav2.compat import safe_move_tensor
from exllamav2.fasttensors import cleanup_stfiles
# from exllamav2.util import list_live_tensors, print_vram_usage, set_snapshot, diff_snapshot, print_vram_usage_peak
import gc
import threading
from typing import Callable

def _torch_device(idx):
    if idx == -1: return "cpu"
    return f"cuda:{idx}"


class ExLlamaV2DeviceTensors:

    model: ExLlamaV2
    device_idx: int
    ready: bool

    scratch_bytes: int
    scratch_idx: int

    sin: torch.Tensor | None
    cos: torch.Tensor | None

    scratch: torch.Tensor | None


    def __init__(self,
                 model: ExLlamaV2,
                 device_idx: int,
                 scratch_bytes: int):

        self.model = model
        self.device_idx = device_idx
        self.ready = False
        self.scratch = None
        self.scratch_bytes = scratch_bytes
        self.scratch_idx = 0


    def prepare(self, scratch):

        self.prepare_sincos()

        if scratch:
            self.scratch = torch.empty((self.scratch_bytes // 2,), dtype = torch.half, device = _torch_device(self.device_idx))

        self.ready = True


    def drop(self):

        self.scratch = None
        self.sin = None
        self.cos = None
        self.ready = False


    def free(self):

        self.drop()
        self.scratch_bytes = 1


    def begin_scratch_alloc(self):

        self.scratch_idx = 0


    def get_scratch_slice(self, size_bytes):

        if self.scratch is None: self.prepare(True)

        size_bytes = ((size_bytes + 127) // 128) * 128
        size_half = size_bytes // 2
        scratch_slice = self.scratch.narrow(0, self.scratch_idx, size_half)
        self.scratch_idx += size_half
        return scratch_slice


    def prepare_sincos(self):

        base = self.model.config.rotary_embedding_base
        alpha = self.model.config.scale_alpha_value or 1.0
        scale = self.model.config.scale_pos_emb or 1.0
        head_dim = self.model.config.head_dim
        device = _torch_device(self.device_idx)

        if alpha != 1.0: base *= alpha ** (self.model.config.head_dim / (self.model.config.head_dim - 2))

        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, device = device).float() / head_dim))
        t = torch.arange(self.model.config.max_seq_len, device = device, dtype = torch.float32)

        if scale != 1.0: t /= scale

        freqs = torch.einsum("i,j->ij", t, inv_freq)
        if self.model.config.arch.rope_neox_style:
            emb = torch.cat((freqs, freqs), dim=-1)
        else:
            emb = torch.repeat_interleave(freqs, 2, dim=-1)

        self.sin = emb.sin()[None, None, :, :].half()
        self.cos = emb.cos()[None, None, :, :].half()


class ExLlamaV2:

    config: ExLlamaV2Config
    modules: list[ExLlamaV2Module]
    modules_dict: dict[str: ExLlamaV2Module]
    device_tensors: list[ExLlamaV2DeviceTensors]
    cache_map: dict[int: str]
    last_kv_layer_idx: int
    head_layer_idx: int
    loaded: bool

    def __init__(self, config: ExLlamaV2Config, lazy_load = False):

        self.config = config
        self.modules = []
        self.modules_dict = {}
        self.device_tensors = []
        self.cache_map = {}
        self.loaded = False

        # Build model

        emb = ExLlamaV2Embedding(self, "model.embed_tokens")
        self.modules += [emb]

        for layer_idx in range(self.config.num_hidden_layers):

            layer_key = f"model.layers.{layer_idx}"
            if self.config.arch.parallel_decoder_blocks:
                pd = ExLlamaV2ParallelDecoder(self, layer_key, layer_idx)
                self.modules += [pd]
            else:
                attn = ExLlamaV2Attention(self, layer_key, layer_idx)
                if self.config.arch.is_moe: mlp = ExLlamaV2MoEMLP(self, layer_key, layer_idx)
                else: mlp = ExLlamaV2MLP(self, layer_key, layer_idx)
                self.modules += [attn, mlp]

        if self.config.arch.norm == "layernorm": norm = ExLlamaV2LayerNorm(self, "model.norm")
        elif self.config.arch.norm == "rmsnorm": norm = ExLlamaV2RMSNorm(self, "model.norm")
        else: raise ValueError("unknown norm type")
        self.modules += [norm]

        self.head_layer_idx = len(self.modules)
        head = ExLlamaV2Linear(self, "lm_head",
                               self.config.hidden_size,
                               self.config.vocab_size,
                               False,
                               max_out_len = self.config.max_output_len,
                               prescale = self.config.logit_scale)
        if self.config.arch.lm_head_key != "lm_head":
            head.alt_key = self.config.arch.lm_head_key
        self.modules += [head]

        # Compile dictionary of modules

        for module in self.modules:
            if len(module.submodules) > 0:
                for m in module.submodules: self.modules_dict[m.key] = m
            else:
                self.modules_dict[module.key] = module

        # Find last layer that affects k/v cache

        layer_idx = len(self.modules)
        while True:
            layer_idx -= 1
            if isinstance(self.modules[layer_idx], ExLlamaV2Attention) or \
               isinstance(self.modules[layer_idx], ExLlamaV2ParallelDecoder):
                break

        self.last_kv_layer_idx = layer_idx


    def set_device_map(self,
                       allocation: list[float],
                       embed_cpu: bool = True) -> list[float]:

        self.cache_map = {}

        # Constant shared between layers

        sincos_size = self.config.head_dim * self.config.max_seq_len * 2
        constant_size = sincos_size * 2

        # Max size of hidden state

        state_size = self.config.hidden_size * self.config.max_input_len * self.config.max_batch_size * 2
        mask_size = self.config.max_input_len ** 2 * self.config.max_batch_size * 2

        # Bytes remaining per device

        allocation_bytes =  [a * 1024**3 - (constant_size + state_size + mask_size) for a in allocation]

        # Scratch space required per device

        reserve_bytes = [0 for a in allocation]
        reserve_bytes_attn = [0 for a in allocation]
        fixed_bytes = [0 for a in allocation]

        current_idx = 0
        for idx, module in enumerate(self.modules):

            # Special case for token embeddings on CPU

            if idx == 0 and embed_cpu:

                module.set_device_idx(-1)
                continue

            # Special case for attention

            attn_bytes_current = 0
            if isinstance(module, ExLlamaV2Attention): attn_bytes_current = module.temp_attn_size()

            # Advance current_idx until module fits in allocation

            footprint = module.weight_footprint()   # Footprint, in bytes
            scratch = module.scratch_space()        # Scratch space required by module

            while True:
                assert current_idx < len(allocation_bytes), "Insufficient space in device allocation"
                dev_scratch = max(scratch, reserve_bytes[current_idx])
                dev_scratch_attn = max(attn_bytes_current, reserve_bytes_attn[current_idx])
                if footprint + dev_scratch + dev_scratch_attn <= allocation_bytes[current_idx]: break
                current_idx += 1

            # Size for fixed tensors

            scratch_fixed = module.scratch_space_fixed()
            fixed_bytes[current_idx] = max(scratch_fixed, fixed_bytes[current_idx])

            # Subtract module size from allocation

            reserve_bytes[current_idx] = dev_scratch
            reserve_bytes_attn[current_idx] = dev_scratch_attn
            allocation_bytes[current_idx] -= footprint

            module.set_device_idx(current_idx)

        # Prepare to prepare device tensors

        self.device_tensors = []
        for idx, scratch_bytes in enumerate(fixed_bytes):
            self.device_tensors.append(ExLlamaV2DeviceTensors(self, idx, scratch_bytes))

        # Create map for cache

        self.set_cache_map()

        # Return unused space, in GB

        return [(ab - rb - rba) / 1024**3 for (ab, rb, rba) in zip(allocation_bytes, reserve_bytes, reserve_bytes_attn)]


    def load(self,
             gpu_split: list[float] | None = None,
             lazy: bool = False,
             stats: bool = False,
             callback: Callable[[int, int], None] | None = None,
             callback_gen: Callable[[int, int], None] | None = None):

        f = self.load_gen(gpu_split, lazy, stats, callback, callback_gen)
        for item in f: x = item


    def load_gen(self,
                 gpu_split: list[float] | None = None,
                 lazy: bool = False,
                 stats: bool = False,
                 callback: Callable[[int, int], None] | None = None,
                 callback_gen: Callable[[int, int], None] | None = None):

        with torch.inference_mode():

            stats_ = self.set_device_map(gpu_split or [99999])

            # Load module weights

            if not lazy:

                for idx, module in enumerate(self.modules):

                    if callback is not None: callback(idx, len(self.modules))
                    if callback_gen is not None: yield from callback_gen(idx, len(self.modules))

                    module.load()

                if callback is not None: callback(len(self.modules), len(self.modules))
                if callback_gen is not None: yield from callback_gen(len(self.modules), len(self.modules))

            # Cache map

            self.set_cache_map()

            self.loaded = True
            cleanup_stfiles()

            # if stats: yield gpu_split, stats_
            # else: yield gpu_split


    def load_autosplit(self,
                       cache: ExLlamaV2CacheBase,
                       reserve_vram: int | None = None,
                       last_id_only: bool = False,
                       callback: Callable[[int, int], None] | None = None,
                       callback_gen: Callable[[int, int], None] | None = None):

        f = self.load_autosplit_gen(cache, reserve_vram, last_id_only, callback, callback_gen)
        for item in f: x = item

    def load_autosplit_gen(self,
                           cache: ExLlamaV2CacheBase,
                           reserve_vram: int | None = None,
                           last_id_only: bool = False,
                           callback: Callable[[int, int], None] | None = None,
                           callback_gen: Callable[[int, int], None] | None = None):

        # Limit model's max_input_len to max_seq_len if necessary
        self.config.max_input_len = min(self.config.max_input_len, self.config.max_seq_len)

        minimum_reserve_vram = 256 * 1024**2
        last_touched_device = -1
        current_device = 0
        num_devices = torch.torch.cuda.device_count()
        loras = None  # TODO:

        with torch.inference_mode():

            self.device_tensors = []

            # Reserved space

            if reserve_vram is None:
                reserve_vram = [192 * 1024**2] + [64 * 1024**2] * (num_devices - 1)

            reserved_vram_tensors = []
            minimum_reserve_tensor = None

            # Largest hidden state to ever forward through model

            hidden_state = torch.zeros((1, self.config.max_input_len), dtype = torch.long)
            batch_size, seq_len = hidden_state.shape
            past_len = 0
            attn_params = ExLlamaV2Attention.Params(batch_size, seq_len, past_len, None, None)

            # Size of fixed scratch space

            scratch_fixed = max(module.scratch_space_fixed() for module in self.modules)

            # Load modules and create cache tensors sequentially

            self.cache_map = {}
            for idx, module in enumerate(self.modules):

                if callback is not None: callback(idx, len(self.modules))
                if callback_gen is not None: yield from callback_gen(idx, len(self.modules))

                # Embedding layer on CPU

                if idx == 0:

                    module.set_device_idx(-1)
                    module.load()
                    hidden_state = module.forward(hidden_state)
                    continue

                while True:

                    # If we've reached a new device, allocate fixed tensors

                    if current_device > last_touched_device:

                        self.device_tensors.append(ExLlamaV2DeviceTensors(self, current_device, scratch_fixed))
                        # if attn_mask is not None:
                        #     reserved_vram_tensors.append(attn_mask)
                        #     attn_mask = safe_move_tensor(attn_mask, _torch_device(current_device))
                        # else:
                        #     attn_mask = self.build_attn_mask(batch_size, seq_len, past_len, None, _torch_device(current_device))

                        b = reserve_vram[current_device]
                        reserved_vram_tensors.append(torch.empty((b,), dtype = torch.int8, device = _torch_device(current_device)))
                        minimum_reserve_tensor = torch.empty((minimum_reserve_vram,), dtype = torch.int8, device = _torch_device(current_device))

                        last_touched_device = current_device

                    # Attempt to load module and forward state

                    module.set_device_idx(current_device)

                    hidden_state_backup = safe_move_tensor(hidden_state, "cpu").clone()

                    try:
                        if isinstance(module, ExLlamaV2Attention) or \
                           isinstance(module, ExLlamaV2ParallelDecoder):
                            self.cache_map[module.layer_idx] = module.device()
                            cache.update_cache_tensors()

                        module.load()

                        if idx == self.head_layer_idx:
                            if last_id_only:
                                hidden_state = hidden_state.narrow(-2, -1, 1)
                            elif self.config.max_output_len is not None:
                                hidden_state = hidden_state.narrow(-2, -self.config.max_output_len, self.config.max_output_len)

                        hidden_state = safe_move_tensor(hidden_state, _torch_device(current_device))
                        hidden_state = module.forward(hidden_state, cache = cache, attn_params = attn_params, past_len = past_len, loras = loras)
                        fail = False

                    except Exception as e:

                        test = 0
                        if e.__class__.__name__ == "OutOfMemoryError" or \
                            "CUDA out of memory" in str(e) or \
                            "HIP out of memory" in str(e):
                            fail = True  # Exception object will hold references to tensors so we can't free them here
                        else:
                            raise

                    # If we failed, roll back and advance to next device

                    if fail:

                        module.unload()
                        hidden_state = None

                        if minimum_reserve_tensor is not None: del minimum_reserve_tensor
                        minimum_reserve_tensor = None

                        gc.collect()
                        torch.cuda.empty_cache()
                        hidden_state = hidden_state_backup.clone()

                        current_device += 1
                        if current_device >= num_devices:
                            raise RuntimeError("Insufficient VRAM for model and cache")

                        continue

                    break

            if callback is not None: callback(len(self.modules), len(self.modules))
            if callback_gen is not None: yield from callback_gen(len(self.modules), len(self.modules))

            hidden_state = None
            attn_params = None
            reserved_vram_tensors = None

            gc.collect()
            torch.cuda.empty_cache()
            self.loaded = True
            cleanup_stfiles()

        if 'yield' in locals():
            yield


    def unload(self):

        for module in self.modules:
            module.unload()

        self.modules = []
        self.modules_dict = {}
        self.device_tensors = []


    def set_cache_map(self):

        for module in self.modules:
            if isinstance(module, ExLlamaV2Attention) or \
               isinstance(module, ExLlamaV2ParallelDecoder):
                self.cache_map[module.layer_idx] = module.device()


    def get_cache_devices(self) -> list[str]:

        return list(set(self.cache_map.values()))


    def create_device_tensors(self, scratch_bytes):

        for idx, b in enumerate(scratch_bytes):

            tensors = ExLlamaV2DeviceTensors(self, idx, b)
            self.device_tensors.append(tensors)


    def drop_device_tensors(self):

        for dt in self.device_tensors:
            dt.drop()


    def free_device_tensors(self):

        for dt in self.device_tensors:
            dt.free()


    def get_device_tensors(self, device_idx, scratch = True):

        tensors = self.device_tensors[device_idx]
        if not tensors.ready: tensors.prepare(scratch)
        return tensors


    def get_modules(self) -> list[ExLlamaV2Module]:

        return [module for module in self.modules]  #?


    def update_loras(self):

        for module in self.modules:
            if isinstance(module, ExLlamaV2Attention): module.update_loras()
            if isinstance(module, ExLlamaV2MLP): module.update_loras()
            if isinstance(module, ExLlamaV2MoEMLP): module.update_loras()


    def is_quant(self) -> bool:

        for module in self.modules:
            if isinstance(module, ExLlamaV2Attention):
                if module.is_quant(): return True

        return False


    @torch.inference_mode()
    def forward(self,
                input_ids: torch.Tensor,
                cache: ExLlamaV2CacheBase | list[ExLlamaV2CacheBase] | None = None,
                input_mask: torch.Tensor | None = None,
                preprocess_only: bool = False,
                last_id_only: bool = False,
                loras: list[ExLlamaV2Lora] | None = None,
                return_last_state: bool = False,
                position_offsets: torch.Tensor | None = None,
                abort_event: threading.Event | None = None,
                **kwargs) \
        -> torch.Tensor | tuple[torch.Tensor, torch.Tensor] | None:
        """
        Runs a forward pass through the model. If a cache is used, also appends keys/values to the cache
        and advances it.

        :param input_ids:
            LongTensor of input token IDs, shape (batch_size, q_len)

        :param cache:
            Optional ExLlamaV2Cache. If not provided, q_len must be less than config.max_input_len

        :param input_mask:
            Additive attention bias, shape (batch_size, past_len + q_len, q_len)

        :param preprocess_only:
            Only forward up to the last layer that affects the K/V cache. Does not return logits. Used
            to prefill the cache.

        :param last_id_only:
            Process the entire input sequence but only pass the last token through the head layer and
            only return logits for the last token.

        :param loras:
            List of ExLlamaV2Lora objects to apply during the forward pass

        :param return_last_state:
            Also return the hidden state right before the head layer

        :param position_offsets:
            Tensor of position offsets, shape (batch_size, 1). Offset is applied to position IDs during
            RoPE application.

        :param abort_event:
            Optional event that, if set, will abort the forward pass. Function will return None if aborted.

        :return:
            FP16 logits tensor, shape (batch_size, q_len, vocab_size)
            (optional) state tensor, shape (batch_size, q_len, hidden_size)

        :indexed_embeddings:
            Tensor of embeddings, shape (batch_size, q_len, hidden_size), indexed by input token IDs >=
            ExLlamaV2.EMBEDDING_INDEX
        """

        bsz, q_len = input_ids.shape
        remaining_q_len = q_len

        # Attn and MLP layers have preallocated buffers for temp states, sized by the model config. Effective max input
        # length depends on the current batch size

        effective_max_input_len = self.config.max_input_len * self.config.max_batch_size // bsz

        # Without a cache we can't process the sequence in chunks, so forward the whole thing and assume the input length
        # is less than config.max_input_len

        if cache is None or not isinstance(cache, ExLlamaV2CacheBase):

            assert q_len <= effective_max_input_len, "Maximum input length exceeded in model.forward"

            result, last_state = self._forward(input_ids = input_ids,
                                               cache = cache,
                                               input_mask = input_mask,
                                               preprocess_only = preprocess_only,
                                               last_id_only = last_id_only,
                                               loras = loras,
                                               return_last_state = return_last_state,
                                               position_offsets = position_offsets,
                                               abort_event = abort_event,
                                               **kwargs)

            if abort_event and abort_event.is_set(): return

            if last_state is None:
                return result
            else:
                return result, last_state

        # Confirm that the input fits within the allocated cache space

        past_len = cache.current_seq_len
        assert past_len + q_len <= cache.max_seq_len, "Total sequence length exceeds cache size in model.forward"

        # Split sequence

        result = None
        last_state = None

        chunk_begin = 0
        while chunk_begin < q_len:

            # Limit chunk_size to max_input_len

            chunk_size = min(remaining_q_len, effective_max_input_len)

            # Limit chunk_size to keep size of attention operation <= max_attention_size

            past_len = cache.current_seq_len
            attn_size = (past_len + remaining_q_len) * remaining_q_len
            max_a = self.config.max_attention_size
            if attn_size > max_a:
                cs = (math.sqrt(past_len ** 2 + 4 * max_a) - past_len) / 2
                chunk_size = min(chunk_size, math.floor(cs))

            # Process chunk

            chunk_end = min(chunk_begin + chunk_size, q_len)

            # print(f"Forward chunk length: {chunk_end - chunk_begin}")

            _last_id_only = last_id_only
            _preprocess_only = preprocess_only or (chunk_end < q_len and last_id_only)

            r, ls = self._forward(input_ids = input_ids[:, chunk_begin : chunk_end],
                                  cache = cache,
                                  input_mask = input_mask,
                                  preprocess_only = _preprocess_only,
                                  last_id_only = _last_id_only,
                                  loras = loras,
                                  return_last_state = return_last_state and remaining_q_len <= chunk_size,
                                  position_offsets = position_offsets,
                                  abort_event = abort_event,
                                  **kwargs)

            if abort_event and abort_event.is_set(): return

            if not _preprocess_only:
                result = r if result is None else torch.cat((result, r), dim = 1)
                r = None

            chunk_begin = chunk_end
            remaining_q_len -= chunk_size
            last_state = ls

        if last_state is None:
            return result
        else:
            return result, last_state


    @torch.inference_mode()
    def _forward(self,
                 input_ids: torch.Tensor,
                 cache: ExLlamaV2CacheBase | list[ExLlamaV2CacheBase] | None = None,
                 input_mask: torch.Tensor | None = None,
                 preprocess_only: bool = False,
                 last_id_only: bool = False,
                 loras: list[ExLlamaV2Lora] | None = None,
                 return_last_state: bool = False,
                 position_offsets: torch.Tensor | None = None,
                 abort_event: threading.Event | None = None,
                 **kwargs) \
        -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:

        batch_size, seq_len = input_ids.shape
        past_len = 0
        if cache is not None:
            if isinstance(cache, ExLlamaV2CacheBase):
                past_len = cache.current_seq_len
            else:
                past_len = [c.current_seq_len for c in cache]

        assert self.config.max_output_len is None or \
            preprocess_only or \
            last_id_only or \
            seq_len <= self.config.max_output_len, \
            "seq_len exceeds max_output_len"

        # assert cache is None or isinstance(cache, list) or batch_size <= cache.batch_size

        x = input_ids
        attn_params = ExLlamaV2Attention.Params(batch_size, seq_len, past_len, input_mask, position_offsets)
        last_state = None
        last_module = None

        for idx, module in enumerate(self.modules):

            # Respect abort signal

            if abort_event and abort_event.is_set(): return None, None

            # Onward

            device = _torch_device(module.device_idx)

            if idx == self.head_layer_idx:
                if last_id_only and return_last_state:
                    x = x.narrow(-2, -1, 1)
                    last_state = x
                elif last_id_only:
                    x = x.narrow(-2, -1, 1)
                elif return_last_state:
                    last_state = x.narrow(-2, -1, 1)

            x = safe_move_tensor(x, device)
            x = module.forward(x, cache = cache, attn_params = attn_params, past_len = past_len, loras = loras, **kwargs)

            if preprocess_only and idx == self.last_kv_layer_idx:
                x = None
                break

        # Advance cache

        if cache is not None:
            if isinstance(cache, list):
                for c in cache: c.current_seq_len += seq_len
            else:
                cache.current_seq_len += seq_len

        # Apply logit scale

        # if x is not None and self.config.logit_scale != 1:
        #     x.mul_(self.config.logit_scale)

        # Set padding logits to -inf

        if x is not None:
            head_padding = self.modules[-1].padding
            if head_padding > 0:
                x[:, :, -head_padding:] = -65504.

        return x, last_state