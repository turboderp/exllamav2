from __future__ import annotations
import torch
from torch import nn
from exllamav2.module import ExLlamaV2Module
from exllamav2.attn import ExLlamaV2Attention
from exllamav2.compat import safe_move_tensor

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from exllamav2.model import ExLlamaV2

class ExLlamaV2PosEmbedding(ExLlamaV2Module):

    name: str = "Pos. Embedding"

    embedding: nn.Embedding | None
    native_ctx_size: int | None


    def __init__(self,
                 model: ExLlamaV2,
                 key: str):
        super().__init__(model, key)

        self.native_ctx_size = model.config.max_seq_len
        self.embedding = None


    @torch.inference_mode
    def load(self):

        w = self.load_weight()
        assert isinstance(w, nn.Parameter)
        self.native_ctx_size = w.shape[0]
        assert self.model.config.max_seq_len <= self.native_ctx_size, \
            f"Learned positional embeddings cannot be extended past native size of {self.native_ctx_size}."

        self.embedding = nn.Embedding(self.native_ctx_size, self.model.config.hidden_size, device = "meta")
        self.embedding.weight = w


    def unload(self):

        del self.embedding
        self.embedding = None


    def get_weight(self) -> torch.Tensor:

        return self.embedding.weight.data


    def weight_footprint(self) -> int:

        return self.native_ctx_size * self.model.config.hidden_size * 2


    def scratch_space_fixed(self) -> int:

        return 0


    def scratch_space(self) -> int:

        return 0


    def forward(self,
                hidden_states: torch.Tensor,
                cache = None,
                attn_params: ExLlamaV2Attention.Params = None,
                past_len = None,
                intermediates: bool = False,
                loras = None,
                **kwargs) -> torch.Tensor | dict[str: torch.Tensor]:

        if attn_params is None:

            pos_start = 0
            pos_end = hidden_states.shape[1]
            emb_slice = self.embedding.weight.data[pos_start:pos_end]
            hidden_states[:] += emb_slice

        else:

            bsz, q_len, dim = hidden_states.shape
            for b in range(bsz):

                if isinstance(attn_params, ExLlamaV2Attention.PagedParams):
                    past_len = attn_params.cache_seqlens[b]
                    offset = 0
                else:
                    if attn_params.past_len is not None:
                        past_len = attn_params.past_len
                    else:
                        assert attn_params.past_lens is not None
                        past_len = attn_params.past_lens[b]
                    if attn_params.position_offsets is not None:
                        offset = attn_params.position_offsets[b].item()
                    else:
                        offset = 0

                slice_a = past_len + offset
                slice_b = past_len + q_len + offset
                assert slice_b > 0
                target_a = 0
                target_b = q_len

                if slice_a < 0:
                    target_a -= slice_a
                    slice_a = 0

                emb_slice = self.embedding.weight.data[slice_a : slice_b]
                hidden_states[b, target_a:target_b] += emb_slice

        if intermediates:
            return {"hidden_states": hidden_states}
        else:
            return hidden_states
