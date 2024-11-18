from __future__ import annotations

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from exllamav2.model import ExLlamaV2

import torch

# Assume no model will have more than one billion regular text tokens, and assign dynamic token IDs starting from
# that index. Token indices are 64 bits so a global instance of this counter should suffice

class ExLlamaV2MMAllocator:

    next_token_index: int

    def __init__(self):
        self.next_token_index = 1000000000

    def allocate(self, num_tokens):
        idx = self.next_token_index
        self.next_token_index += num_tokens
        return idx

global_allocator = ExLlamaV2MMAllocator()


class ExLlamaV2MMEmbedding:
    """
    Container for one embedding (image etc.) and associated metadata
    """

    model: ExLlamaV2
    text_alias: str
    embeddings: torch.Tensor
    first_index: int
    length: int
    thw_grid: tuple[int, int, int]
    pre_tokens: int
    post_tokens: int

    metadata: dict

    def __init__(
        self,
        model: ExLlamaV2,
        embeddings: torch.Tensor,
        text_alias: str | None = None,
        thw_grid: tuple[int, int, int] | None = None,
        pre_tokens: int = 0,
        post_tokens: int = 0,
    ):
        """
        :param model:
            Model instance

        :param embeddings:
            Embeddings, shape (num_tokens, input_dim)

        :param text_alias:
            Text string to represent this embedding for tokenizing
        """

        global global_allocator

        self.model = model
        self.embeddings = embeddings
        self.text_alias = text_alias
        self.thw_grid = thw_grid
        self.pre_tokens = pre_tokens
        self.post_tokens = post_tokens

        self.metadata = {}

        self.length = embeddings.shape[0]
        dim = embeddings.shape[1]
        assert dim == model.config.hidden_size, \
            "Embedding dimension doesn't match model hidden dimension"

        self.first_index = global_allocator.allocate(self.length)

        # Auto-assign text alias

        if not self.text_alias:
            self.text_alias = f"<$EMB_{self.first_index}$>"


    def get_ids_tensor(self):
        return torch.arange(
            start = self.first_index,
            end = self.first_index + self.length,
            dtype = torch.long
        ).unsqueeze(0)

    def get_ids(self):
        return list(range(
            self.first_index,
            self.first_index + self.length
        ))

    def get_vision_token_range(self):
        return self.first_index + self.pre_tokens, self.first_index + self.length - self.post_tokens
