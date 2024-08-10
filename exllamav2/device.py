from __future__ import annotations

def _torch_device(idx):
    if idx == -1: return "cpu"
    return f"cuda:{idx}"

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from exllamav2.model import ExLlamaV2

from exllamav2.architecture import RopeStyle

import torch
import math


global_streams = {}


class ExLlamaV2DeviceContext:

    model: ExLlamaV2
    device_idx: int
    ready: bool

    scratch_bytes: int
    scratch_idx: int

    sin: torch.Tensor | None
    cos: torch.Tensor | None

    scratch: torch.Tensor | None

    stream: torch.cuda.Stream

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

        # Create streams (only one per device)

        if device_idx not in global_streams:
            global_streams[device_idx] = torch.cuda.Stream(torch.device(device_idx), -100)

        self.stream = global_streams[device_idx]

        xx = 0


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

        size_bytes = ((size_bytes + 63) // 64) * 64
        size_half = size_bytes // 2
        scratch_slice = self.scratch.narrow(0, self.scratch_idx, size_half)
        self.scratch_idx += size_half
        return scratch_slice


    @staticmethod
    def _apply_scaling(
        freqs: torch.Tensor,
        scale_factor: float = 8,
        low_freq_factor: float = 1,
        high_freq_factor: float = 4,
        old_context_len: int = 8192,  # original llama3 length
    ):
        low_freq_wavelen = old_context_len / low_freq_factor
        high_freq_wavelen = old_context_len / high_freq_factor
        new_freqs = []

        for freq in freqs:
            wavelen = 2 * math.pi / freq
            if wavelen < high_freq_wavelen:
                new_freqs.append(freq)
            elif wavelen > low_freq_wavelen:
                new_freqs.append(freq / scale_factor)
            else:
                assert low_freq_wavelen != high_freq_wavelen
                smooth = (old_context_len / wavelen - low_freq_factor) / (high_freq_factor - low_freq_factor)
                new_freqs.append((1 - smooth) * freq / scale_factor + smooth * freq)
        return torch.tensor(new_freqs, dtype = freqs.dtype, device = freqs.device)


    def prepare_sincos(self):

        device = _torch_device(self.device_idx)

        cfg = self.model.config
        if cfg.arch.rope_style == RopeStyle.NONE:
            self.sin = torch.zeros((1,), device = device, dtype = torch.half)
            self.cos = self.sin
            return

        base = cfg.rotary_embedding_base
        alpha = cfg.scale_alpha_value or 1.0
        scale = cfg.scale_pos_emb or 1.0
        head_dim = cfg.head_dim
        scaling_factor = 1.0

        # Alpha scaling for any rope_scaling type

        if alpha != 1.0: base *= alpha ** (cfg.head_dim / (cfg.head_dim - 2))

        # "su"

        if cfg.alt_rope_method == "su":

            a = cfg.max_seq_len
            b = cfg.original_max_seq_len
            if a > b:
                ext_factors = torch.tensor(cfg.scale_long_factor, dtype = torch.float32, device = device)
                scaling_factor = math.sqrt(1 + math.log(a / b) / math.log(b))
            else:
                ext_factors = torch.tensor(cfg.scale_short_factor, dtype = torch.float32, device = device)

            inv_freq = 1.0 / (ext_factors * base ** (torch.arange(0, head_dim, 2, device = device).float() / head_dim))

        # Llama 3.1

        elif cfg.alt_rope_method == "llama3":

            inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, device = device).float() / head_dim))
            inv_freq = self._apply_scaling(
                inv_freq,
                cfg.l3_rope_factor,
                cfg.l3_rope_low_freq_factor,
                cfg.l3_rope_high_freq_factor,
                cfg.l3_rope_original_max_position_embeddings,
            )

        # Regular

        else:

            inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, device = device).float() / head_dim))

        # Common

        t = torch.arange(cfg.max_seq_len, device = device, dtype = torch.float32)
        if scale != 1.0: t /= scale

        freqs = torch.einsum("i,j->ij", t, inv_freq)
        if cfg.arch.rope_style == RopeStyle.NEOX:
            emb = torch.cat((freqs, freqs), dim=-1)
        elif cfg.arch.rope_style == RopeStyle.GPTJ:
            emb = torch.repeat_interleave(freqs, 2, dim=-1)
        else:
            raise ValueError()

        self.sin = emb.sin()[None, None, :, :]
        self.cos = emb.cos()[None, None, :, :]
        if scaling_factor != 1.0:
            self.sin *= scaling_factor
            self.cos *= scaling_factor
        self.sin = self.sin.half()
        self.cos = self.cos.half()
