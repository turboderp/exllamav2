from exllamav2 import (
    ExLlamaV2,
    ExLlamaV2Cache,
    ExLlamaV2Tokenizer
)
from exllamav2.generator import (
    ExLlamaV2Sampler
)
import torch
import random

import torch.nn.functional as F

class ExLlamaV2BaseGenerator:

    # Internal state

    model: ExLlamaV2
    cache: ExLlamaV2Cache
    tokenizer: ExLlamaV2Tokenizer

    sequence_ids: torch.tensor = None

    def __init__(self, model, cache, tokenizer):

        self.model = model
        self.cache = cache
        self.tokenizer = tokenizer


    # For testing purposes, run a forward pass to make sure CUDA is fully initialized

    def warmup(self):

        input_ids = torch.zeros((1, 4), dtype = torch.long)
        self.gen_begin_base(input_ids)


    def full(self):

        return self.sequence_ids.shape[-1] >= self.model.config.max_seq_len


    def generate_simple(self, prompt: str, gen_settings: ExLlamaV2Sampler.Settings, num_tokens: int, seed = None):

        if seed is not None: random.seed(seed)

        ids = self.tokenizer.encode(prompt)

        overflow = ids.shape[-1] + num_tokens - self.model.config.max_seq_len
        if overflow > 0: ids = ids[:, -overflow:]

        self.gen_begin_base(ids)

        for i in range(num_tokens):

            logits = self.model.forward(self.sequence_ids[:, -1:], self.cache).float().cpu()
            token, _ = ExLlamaV2Sampler.sample(logits, gen_settings, self.sequence_ids, random.random())
            self.sequence_ids = torch.cat([self.sequence_ids, token], dim = 1)

        text = self.tokenizer.decode(self.sequence_ids[0])
        return text


    def gen_begin_base(self, input_ids):

        self.cache.current_seq_len = 0
        self.model.forward(input_ids[:, :-1], self.cache, preprocess_only = True)

        self.sequence_ids = input_ids.clone()
        self.sequence_ids = input_ids

