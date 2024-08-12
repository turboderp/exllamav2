from __future__ import annotations
import torch
from torch import nn
from exllamav2.module import ExLlamaV2Module
from exllamav2.attn import ExLlamaV2Attention
from exllamav2.compat import safe_move_tensor

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from exllamav2.model import ExLlamaV2

EMBEDDING_INDEX: int = 1000000

class ExLlamaV2Embedding(ExLlamaV2Module):

    name: str = "Embedding"

    embedding: nn.Embedding | None
    native_vocab_size: int | None

    is_tp: bool

    def __init__(self,
                 model: ExLlamaV2,
                 key: str):
        super().__init__(model, key)

        self.is_tp = False

        self.native_vocab_size = None
        self.embedding = None


    def tp_split(self):

        self.is_tp = True


    @torch.inference_mode
    def load(self, device_context: bool = True):

        vocab_size = self.model.config.vocab_size
        hidden_size = self.model.config.hidden_size
        pad_token_id = self.model.config.pad_token_id

        w = self.load_weight()
        assert isinstance(w, nn.Parameter)
        # TODO: Figure out why pinning this tensor allocates GPU memory??
        # w.pin_memory()
        self.native_vocab_size = w.shape[0]

        self.embedding = nn.Embedding(vocab_size, hidden_size, pad_token_id, device = "meta")
        if self.model.config.scale_emb != 1:
            w *= self.model.config.scale_emb
        self.embedding.weight = w


    def unload(self):

        del self.embedding
        self.embedding = None


    def get_weight(self) -> torch.Tensor:

        if self.model.config.scale_emb != 1:
            return self.embedding.weight.data / self.model.config.scale_emb
        else:
            return self.embedding.weight.data


    def weight_footprint(self) -> int:

        vocab_size = self.model.config.vocab_size
        hidden_size = self.model.config.hidden_size
        # kv_size = self.model.config.num_key_value_heads * self.model.config.head_dim

        return vocab_size * hidden_size * 2


    def scratch_space_fixed(self) -> int:

        return 0


    def scratch_space_tp(self) -> list[int]:

        return [0] * self.model.tp_context.num_devices


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

        cfg = self.model.config

        # If input IDs contain negative values, assume they are padding tokens from a model with not pad_token_id
        # defined

        hidden_states = hidden_states.clamp(min = 0)

        # Apply indexed embeddings

        indexed_embeddings = kwargs.get("indexed_embeddings")
        if indexed_embeddings is not None:

            # Split prompt

            offset = EMBEDDING_INDEX
            input_ids = hidden_states
            standard_mask = input_ids < offset
            indexed_mask = input_ids >= offset

        if indexed_embeddings is not None and indexed_mask.any():

            # Create combined tensor on the target device

            batch_size, seq_len = input_ids.shape
            hidden_size = cfg.hidden_size
            combined_embeddings = torch.empty(batch_size, seq_len, hidden_size,
                                              device = indexed_embeddings.device,
                                              dtype = indexed_embeddings.dtype)

            # Extract standard embeddings, copy to target device and insert in-place

            attn_params.rope_mask = standard_mask
            if standard_mask.any():
                for i in range(batch_size):
                    standard_mask_ = standard_mask[i]
                    input_ids_ = input_ids[i]
                    standard_ids_ = input_ids_[standard_mask_]
                    if loras is not None and loras[0].embed_tokens is not None:
                        standard_embeddings_ = loras[0].embed_tokens(standard_ids_)
                    else:
                        standard_embeddings_ = self.embedding(standard_ids_)
                    standard_embeddings_ = safe_move_tensor(standard_embeddings_, indexed_embeddings.device)
                    combined_embeddings[i][standard_mask_] = standard_embeddings_

            # Normalization

            if cfg.arch.residual_stream_fp32:
                combined_embeddings = combined_embeddings.float()
            if cfg.arch.normalize_embeddings:
                combined_embeddings *= cfg.hidden_size ** 0.5

            # Extract indexed embeddings and insert in-place

            for i in range(batch_size):
                indexed_ids_ = input_ids[i][indexed_mask[i]] - offset
                combined_embeddings[i][indexed_mask[i]] = indexed_embeddings[i][indexed_ids_]

            hidden_states = combined_embeddings

        # Call embedding module if no indexed embeddings

        else:
            if loras is not None and loras[0].embed_tokens is not None:
                hidden_states = loras[0].embed_tokens(hidden_states)
            else:
                hidden_states = self.embedding(hidden_states)

            if cfg.arch.residual_stream_fp32:
                hidden_states = hidden_states.float()
            if cfg.arch.normalize_embeddings:
                hidden_states *= cfg.hidden_size ** 0.5

        # Move to pinned temp buffer for TP

        if self.is_tp:
            ctx = self.model.tp_context
            hidden_states = ctx.copy_pinned(0, hidden_states)

        if intermediates:
            return {"hidden_states": hidden_states}
        else:
            return hidden_states
