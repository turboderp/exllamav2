from __future__ import annotations

import torch
import math

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from exllamav2 import ExLlamaV2Config

# "su"

def get_rope_params_su(
    device: torch.Device,
    cfg: ExLlamaV2Config,
):
    head_dim = cfg.head_dim
    base = cfg.rotary_embedding_base

    a = cfg.max_seq_len
    b = cfg.original_max_seq_len
    if a > b:
        ext_factors = torch.tensor(cfg.scale_long_factor, dtype = torch.float32, device = device)
        scaling_factor = math.sqrt(1 + math.log(a / b) / math.log(b))
    else:
        ext_factors = torch.tensor(cfg.scale_short_factor, dtype = torch.float32, device = device)
        scaling_factor = 1.0

    inv_freq = 1.0 / (ext_factors * base ** (torch.arange(0, head_dim, 2, device = device).float() / head_dim))
    return inv_freq, scaling_factor


# Llama 3.1

def get_rope_params_llama3(
    device: torch.Device,
    cfg: ExLlamaV2Config,
):
    head_dim = cfg.head_dim
    base = cfg.rotary_embedding_base

    def apply_scaling(
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

    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    inv_freq = apply_scaling(
        inv_freq,
        cfg.l3_rope_factor,
        cfg.l3_rope_low_freq_factor,
        cfg.l3_rope_high_freq_factor,
        cfg.l3_rope_original_max_position_embeddings,
    )
    return inv_freq, 1.0

# YaRN
# Adapted from transformers: https://github.com/huggingface/transformers/blob/2e24ee4dfa39cc0bc264b89edbccc373c8337086/src/transformers/modeling_rope_utils.py#L163

def get_rope_params_yarn(
    device: torch.Device,
    cfg: ExLlamaV2Config,
):
    head_dim = cfg.head_dim
    base = cfg.rotary_embedding_base
    yarn_max_position_embeddings = cfg.max_seq_len

    # Only activate if longer than original ctx
    if cfg.max_seq_len > cfg.yarn_rope_original_max_position_embeddings:

        partial_rotary_factor = 1.0  # Placeholder, assume no partial_rotary_factor in config.
        dim = int(head_dim * partial_rotary_factor)

        factor = cfg.yarn_rope_factor

        # Sets the attention factor as suggested in the paper
        # See: https://github.com/huggingface/transformers/blob/main/examples/modular-transformers/modeling_super.py#L190-L191
        scaling_factor = 0.1 * math.log(factor) + 1.0

        # Optional config options
        # beta_fast/beta_slow: as suggested in the paper, default to 32/1 (correspondingly)
        beta_fast = 32
        beta_slow = 1

        # Compute the inverse frequencies
        def find_correction_dim(num_rotations, dim, base, yarn_max_position_embeddings):
            """Inverse dimension formula to find the dimension based on the number of rotations"""
            return (dim * math.log(yarn_max_position_embeddings / (num_rotations * 2 * math.pi))) / (2 * math.log(base))

        def find_correction_range(low_rot, high_rot, dim, base, yarn_max_position_embeddings):
            """Find dimension range bounds based on rotations"""
            low = math.floor(find_correction_dim(low_rot, dim, base, yarn_max_position_embeddings))
            high = math.ceil(find_correction_dim(high_rot, dim, base, yarn_max_position_embeddings))
            return max(low, 0), min(high, dim - 1)

        def linear_ramp_factor(min, max, dim):
            if min == max:
                max += 0.001  # Prevent singularity

            linear_func = (torch.arange(dim, dtype = torch.float32) - min) / (max - min)
            ramp_func = torch.clamp(linear_func, 0, 1)
            return ramp_func

        # Note on variable naming: "interpolation" comes from the original technique, where we interpolate the position IDs
        # to expand the possible context length. In other words, interpolation = apply scaling factor.
        pos_freqs = base ** (torch.arange(0, dim, 2).float().to(device) / dim)
        inv_freq_extrapolation = 1.0 / pos_freqs
        inv_freq_interpolation = 1.0 / (factor * pos_freqs)

        low, high = find_correction_range(beta_fast, beta_slow, dim, base, yarn_max_position_embeddings)

        # Get n-dimensional rotational scaling corrected for extrapolation
        inv_freq_extrapolation_factor = 1 - linear_ramp_factor(low, high, dim // 2).float().to(device)
        inv_freq = (
                inv_freq_interpolation * (1 - inv_freq_extrapolation_factor)
                + inv_freq_extrapolation * inv_freq_extrapolation_factor
        )
    else:
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
        scaling_factor = 1.0

    return inv_freq, scaling_factor

# Default

def get_rope_params_default(
    device: torch.Device,
    cfg: ExLlamaV2Config,
):
    head_dim = cfg.head_dim
    base = cfg.rotary_embedding_base

    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, device = device).float() / head_dim))
    return inv_freq, 1.0


def get_rope_params(
    device: torch.Device,
    cfg: ExLlamaV2Config,
):
    if cfg.alt_rope_method == "su":
        inv_freq, scaling_factor = get_rope_params_su(device, cfg)
    elif cfg.alt_rope_method == "llama3":
        inv_freq, scaling_factor = get_rope_params_llama3(device, cfg)
    elif cfg.alt_rope_method == "yarn":
        inv_freq, scaling_factor = get_rope_params_yarn(device, cfg)
    else:
        inv_freq, scaling_factor = get_rope_params_default(device, cfg)

    if cfg.arch.lm.rope_freq_half:
        inv_freq = inv_freq.half()
    return inv_freq, scaling_factor
