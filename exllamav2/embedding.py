from __future__ import annotations
import torch
from torch import nn
from exllamav2.module import ExLlamaV2Module

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from exllamav2.model import ExLlamaV2

class ExLlamaV2Embedding(ExLlamaV2Module):

    name: str = "Embedding"

    embedding: nn.Embedding | None
    native_vocab_size: int | None


    def __init__(self,
                 model: ExLlamaV2,
                 key: str):
        super().__init__(model, key)

        self.native_vocab_size = None
        self.embedding = None


    def load(self):

        vocab_size = self.model.config.vocab_size
        hidden_size = self.model.config.hidden_size
        pad_token_id = self.model.config.pad_token_id

        w = self.load_weight()
        assert isinstance(w, nn.Parameter)
        self.native_vocab_size = w.shape[0]

        self.embedding = nn.Embedding(vocab_size, hidden_size, pad_token_id, device = "meta")
        self.embedding.weight = w


    def unload(self):

        del self.embedding
        self.embedding = None


    def get_weight(self) -> torch.Tensor:

        return self.embedding.weight.data


    def weight_footprint(self) -> int:

        vocab_size = self.model.config.vocab_size
        hidden_size = self.model.config.hidden_size
        # kv_size = self.model.config.num_key_value_heads * self.model.config.head_dim

        return vocab_size * hidden_size * 2


    def scratch_space_fixed(self) -> int:

        return 0


    def scratch_space(self) -> int:

        return 0


    def forward(self,
                hidden_states: torch.Tensor,
                cache = None,
                attn_params = None,
                past_len = None,
                intermediates: bool = False,
                loras = None) -> torch.Tensor | dict[str: torch.Tensor]:

        hidden_states = self.embedding.forward(hidden_states)

        # Normalize the input embeddings for Gemma
        if self.model.config.arch.normalize_embeddings:
            hidden_states = hidden_states * (self.model.config.hidden_size ** 0.5)

        if intermediates:
            return {"hidden_states": hidden_states}
        else:
            return hidden_states

