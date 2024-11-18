from __future__ import annotations

import torch
from exllamav2.ext import exllamav2_ext as ext_c, none_tensor
from exllamav2 import rope

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from exllamav2.generator import ExLlamaV2MMEmbedding
    from exllamav2 import ExLlamaV2Config

def gen_mrope_embed(
    config: ExLlamaV2Config,
    input_ids: torch.Tensor,
    embeddings: list[ExLlamaV2MMEmbedding],
    max_length: int,
) -> tuple:
    """
    Generate RoPE embeddings (sin/cos table) for a multimodal context

    :param config:
        Model config

    :param input_ids:
        Tokenized context including MM token IDs

    :param embeddings:
        List of embedded MM objects

    :param max_length:
        Total length of context (including max_new_tokens)

    :return:
        tuple of sin and cos Tensors
    """

    # Create 3D position IDs

    ids = input_ids.squeeze(0)
    mrope_pos_ids = torch.zeros((3, max_length), dtype = torch.long).contiguous()
    merge_size = 1 if not embeddings else embeddings[0].model.config.vision_spatial_merge_size
    spans = []
    grids = []

    for embedding in embeddings:
        tmin, tmax = embedding.get_vision_token_range()
        thw = embedding.thw_grid
        spans.append((tmin, tmax))
        grids.append(thw)

    offset = ext_c.gen_mrope_pos_ids(mrope_pos_ids, ids, merge_size, spans, grids)

    # Get RoPE params

    inv_freq, scaling_factor = rope.get_rope_params("cpu", config)

    # Create embeddings

    inv_freq_expanded = inv_freq[None, None, :, None].float().expand(3, 1, -1, 1)
    position_ids_expanded = mrope_pos_ids[:, None, None, :].float()

    freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(2, 3)
    emb = torch.cat((freqs, freqs), dim = -1)
    cos = emb.cos()
    sin = emb.sin()

    cos = (cos * scaling_factor).half()
    sin = (sin * scaling_factor).half()

    mrope_section = config.mrope_section * 2
    unsqueeze_dim = 1

    cos = torch.cat([m[i % 3] for i, m in enumerate(cos.split(mrope_section, dim = -1))], dim = -1).unsqueeze(unsqueeze_dim)
    sin = torch.cat([m[i % 3] for i, m in enumerate(sin.split(mrope_section, dim = -1))], dim = -1).unsqueeze(unsqueeze_dim)

    return (sin, cos), offset