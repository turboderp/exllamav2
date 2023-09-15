import sys
min_version = (3, 9)
if sys.version_info < min_version:
    print("")
    print(f" ## Warning: this project requires Python {min_version[0]}.{min_version[1]} or higher.")
    print("")

import torch
import math
from exllamav2.config import ExLlamaV2Config
from exllamav2.cache import ExLlamaV2Cache
from exllamav2.linear import ExLlamaV2Linear
from exllamav2.module import ExLlamaV2Module
from exllamav2.rmsnorm import ExLlamaV2RMSNorm
from exllamav2.attn import ExLlamaV2Attention
from exllamav2.mlp import ExLlamaV2MLP
from exllamav2.embedding import ExLlamaV2Embedding
# from exllamav2.util import list_live_tensors, print_vram_usage, set_snapshot, diff_snapshot, print_vram_usage_peak

def _torch_device(idx):
    if idx == -1: return "cpu"
    return f"cuda:{idx}"


class ExLlamaV2DeviceTensors:

    model = None
    device_idx: int
    ready: bool

    scratch_bytes: int
    scratch_idx: int

    sin: torch.tensor
    cos: torch.tensor

    scratch: torch.tensor = None


    def __init__(self, model, device_idx, scratch_bytes):

        self.model = model
        self.device_idx = device_idx
        self.ready = False
        self.scratch_bytes = scratch_bytes
        self.scratch_idx = 0


    def prepare(self, scratch):

        self.prepare_sincos()

        if scratch:
            self.scratch = torch.empty((self.scratch_bytes // 2,), dtype = torch.half, device = _torch_device(self.device_idx))

        self.ready = True


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
        alpha = self.model.config.scale_alpha_value
        scale = self.model.config.scale_pos_emb
        head_dim = self.model.config.head_dim
        device = _torch_device(self.device_idx)

        if alpha != 1.0: base *= alpha ** (self.model.config.head_dim / (self.model.config.head_dim - 2))

        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, device = device).float() / head_dim))
        t = torch.arange(self.model.config.max_seq_len, device = device, dtype = torch.float32)

        if scale != 1.0: t /= scale

        freqs = torch.einsum("i,j->ij", t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)

        self.sin = emb.sin()[None, None, :, :].half()
        self.cos = emb.cos()[None, None, :, :].half()


class ExLlamaV2:

    config: ExLlamaV2Config
    modules: list = []
    modules_dict: dict = {}
    device_tensors: list = []
    cache_map: dict
    last_kv_layer_idx: int


    def __init__(self, config: ExLlamaV2Config, lazy_load = False):

        self.config = config
        self.modules = []
        self.modules_dict = {}
        self.device_tensors = []
        self.cache_map = {}

        # Build model

        self.modules.append(ExLlamaV2Embedding(self, "model.embed_tokens"))
        self.modules_dict[self.modules[-1].key] = self.modules[-1]

        for layer_idx in range(self.config.num_hidden_layers):

            self.modules.append(ExLlamaV2Attention(self, f"model.layers.{layer_idx}", layer_idx))
            for m in self.modules[-1].submodules: self.modules_dict[m.key] = m
            self.modules.append(ExLlamaV2MLP(self, f"model.layers.{layer_idx}", layer_idx))
            for m in self.modules[-1].submodules: self.modules_dict[m.key] = m

        self.modules.append(ExLlamaV2RMSNorm(self, "model.norm"))
        self.modules_dict[self.modules[-1].key] = self.modules[-1]
        self.modules.append(ExLlamaV2Linear(self, "lm_head", self.config.hidden_size, self.config.vocab_size, False))
        self.modules_dict[self.modules[-1].key] = self.modules[-1]

        # Find last layer that affects k/v cache

        layer_idx = len(self.modules)
        while True:
            layer_idx -= 1
            if isinstance(self.modules[layer_idx], ExLlamaV2Attention):
                break

        self.last_kv_layer_idx = layer_idx


    def set_device_map(self, allocation, embed_cpu = True):

        self.cache_map = {}

        # Constant shared between layers

        sincos_size = self.config.head_dim * self.config.max_seq_len * 2
        constant_size = sincos_size * 2

        # Max size of hidden state
        # TODO: Option to reserve space for cache while loading model

        state_size = self.config.hidden_size * self.config.max_input_len * self.config.max_batch_size * 2
        mask_size = self.config.max_input_len ** 2 * 2

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


    def load(self, gpu_split = None, lazy = False, stats = False):

        with torch.inference_mode():

            stats = self.set_device_map(gpu_split or [99999])

            # Load module weights

            if not lazy:

                for module in self.modules: module.load()

            # Cache map

            self.set_cache_map()

            if stats: return gpu_split, stats
            else: return gpu_split


    def set_cache_map(self):

        for module in self.modules:
            if isinstance(module, ExLlamaV2Attention): self.cache_map[module.layer_idx] = module.device()


    def create_device_tensors(self, scratch_bytes):

        for idx, bytes in enumerate(scratch_bytes):

            tensors = ExLlamaV2DeviceTensors(self, idx, bytes)
            self.device_tensors.append(tensors)


    def get_device_tensors(self, device_idx, scratch = True):

        tensors = self.device_tensors[device_idx]
        if not tensors.ready: tensors.prepare(scratch)
        return tensors


    def get_modules(self):

        return [module for module in self.modules]


    def build_attn_mask(self, batch_size, seq_len, past_len, input_mask, device):

        if input_mask is None and seq_len == 1: return None

        if isinstance(past_len, tuple):

            attn_masks = []

            for i in range(len(past_len[1])):

                attn_mask = torch.zeros(1, 1, seq_len, past_len[1][i] + seq_len, dtype = torch.float16, device = device)
                attn_mask_triu = torch.triu(torch.full((seq_len - 1, seq_len - 1), -65504.))
                attn_mask[:, :, : seq_len - 1, past_len[1][i] + 1: past_len[1][i] + seq_len] = attn_mask_triu

                if input_mask is not None:
                    min_mask_width = min(input_mask[i].shape[-1], seq_len + past_len[1][i])
                    input_mask_part = input_mask[i][:, :min_mask_width].to(attn_mask.device)
                    input_mask_part = input_mask_part.unsqueeze(1).unsqueeze(2)
                    attn_mask[:, :, :, :min_mask_width] = torch.minimum(attn_mask[:, :, :, :min_mask_width], input_mask_part)

                attn_masks.append(attn_mask)

            return attn_masks

        else:

            attn_mask = torch.zeros(batch_size, 1, seq_len, past_len + seq_len, dtype = torch.float16, device = device)
            attn_mask_triu = torch.triu(torch.full((seq_len - 1, seq_len - 1), -65504.))
            attn_mask[:, :, : seq_len - 1, past_len + 1: past_len + seq_len] = attn_mask_triu

            if input_mask is not None:
                min_mask_width = min(input_mask.shape[-1], seq_len + past_len)
                input_mask_part = input_mask[:, :min_mask_width].to(attn_mask.device)
                input_mask_part = input_mask_part.unsqueeze(1).unsqueeze(2)
                attn_mask[:, :, :, :min_mask_width] = torch.minimum(attn_mask[:, :, :, :min_mask_width], input_mask_part)

            return attn_mask


    def forward(self, input_ids, cache = None, input_mask = None, preprocess_only = False):

        q_len = input_ids.shape[-1]
        remaining_q_len = q_len
        bsz = input_ids.shape[0]

        # Attn and MLP layers have preallocated buffers for temp states, sized by the model config. Effective max input
        # length depends on the current batch size

        effective_max_input_len = self.config.max_input_len * self.config.max_batch_size // bsz

        # Without a cache we can't process the sequence in chunks, so forward the whole thing and assume the input length
        # is less than config.max_input_len

        if cache is None or not isinstance(cache, ExLlamaV2Cache):

            assert q_len <= effective_max_input_len, "Maximum input length exceeded in model.forward"

            return self._forward(input_ids = input_ids,
                                 cache = cache,
                                 input_mask = input_mask,
                                 preprocess_only = preprocess_only)

        # Confirm that the input fits within the allocated cache space

        past_len = cache.current_seq_len
        assert past_len + q_len <= cache.max_seq_len, "Total sequence length exceeds cache size in model.forward"

        # Split sequence

        result = None

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

            r = self._forward(input_ids = input_ids[:, chunk_begin : chunk_end],
                              cache = cache,
                              input_mask = input_mask,
                              preprocess_only = preprocess_only)

            if not preprocess_only:
                result = r if result is None else torch.cat((result, r), dim = 1)
                r = None

            chunk_begin = chunk_end
            remaining_q_len -= chunk_size

        return result


    def _forward(self, input_ids, cache = None, input_mask = None, preprocess_only = False):

        batch_size, seq_len = input_ids.shape
        past_len = 0
        if cache is not None:
            if isinstance(cache, ExLlamaV2Cache):
                past_len = cache.current_seq_len
            else:
                pl = [c.current_seq_len for c in cache]
                past_len = torch.tensor(pl, dtype = torch.int)
                past_len = (past_len, past_len)

        # assert cache is None or isinstance(cache, list) or batch_size <= cache.batch_size

        x = input_ids
        prev_device = None
        attn_mask = None

        for idx, module in enumerate(self.modules):

            device = _torch_device(module.device_idx)

            # Build attention mask

            if device != prev_device and device != "cpu":

                prev_device = device
                attn_mask = self.build_attn_mask(batch_size, seq_len, past_len, input_mask, device)
                if isinstance(past_len, tuple): past_len = (past_len[0].to(device), past_len[1])

            # Onward

            x = x.to(device)
            x = module.forward(x, cache = cache, attn_mask = attn_mask, past_len = past_len)

            if preprocess_only and idx == self.last_kv_layer_idx:
                x = None
                break

            # print(module.key, module.name, x[0, 0])
            # print("max", torch.max(x).item(), "min",torch.min(x).item())

        # Advance cache

        if cache is not None:
            if isinstance(cache, list):
                for c in cache: c.current_seq_len += seq_len
            else:
                cache.current_seq_len += seq_len

        # Set padding logits to -inf

        if x is not None:
            head_padding = self.modules[-1].padding
            if head_padding > 0:
                x[:, :, -head_padding:] = -65504.

        return x