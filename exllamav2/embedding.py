from __future__ import annotations
import torch
from torch import nn
from exllamav2.module import ExLlamaV2Module
from exllamav2.attn import ExLlamaV2Attention
from exllamav2.compat import safe_move_tensor

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
                attn_params: ExLlamaV2Attention.Params = None,
                past_len = None,
                intermediates: bool = False,
                loras = None,
                **kwargs) -> torch.Tensor | dict[str: torch.Tensor]:

        # Apply indexed embeddings

        indexed_embeddings = kwargs.get("indexed_embeddings")
        if indexed_embeddings:

            # Create combined tensor on the target device

            offset = ExLlamaV2.EMBEDDING_INDEX
            input_ids = hidden_states
            batch_size, seq_len = input_ids.shape
            hidden_size = self.model.config.hidden_size
            combined_embeddings = torch.empty(batch_size, seq_len, hidden_size,
                                              device = indexed_embeddings.device,
                                              dtype = indexed_embeddings.dtype)

            # Extract standard embeddings, copy to target device and insert in-place

            standard_mask = hidden_states < offset
            attn_params.rope_mask = standard_mask
            if standard_mask.any():
                standard_ids = input_ids[standard_mask]
                standard_embeddings = self.embedding(standard_ids)
                standard_embeddings = safe_move_tensor(standard_embeddings, indexed_embeddings.device)

                if self.model.config.arch.normalize_embeddings:
                    standard_embeddings *= self.model.config.hidden_size ** 0.5

                combined_embeddings[standard_mask] = standard_embeddings

            # Extract indexed embeddings and insert in-place

            indexed_mask = input_ids >= offset
            if indexed_mask.any():
                indexed_ids = input_ids[indexed_mask] - offset
                combined_embeddings[indexed_mask] = indexed_embeddings[indexed_ids]

            hidden_states = combined_embeddings

        # Call embedding module if no indexed embeddings

        else:
            hidden_states = self.embedding.forward(hidden_states)

            if self.model.config.arch.normalize_embeddings:
                hidden_states *= self.model.config.hidden_size ** 0.5

        if intermediates:
            return {"hidden_states": hidden_states}
        else:
            return hidden_states
