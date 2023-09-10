from exllamav2 import (
    ExLlamaV2,
    ExLlamaV2Cache,
    ExLlamaV2Tokenizer
)
from exllamav2.generator import (
    ExLlamaV2Sampler,
    ExLlamaV2BaseGenerator
)
import torch
import random

class ExLlamaV2SpeculativeGenerator(ExLlamaV2BaseGenerator):

    draft_model: ExLlamaV2
    draft_cache: ExLlamaV2Cache

    default_predict_len: int
    prob_threshold: float = 0.15
    default_predict_len: int = 5

    attempts: list
    hits: list

    def __init__(self, model, cache, draft_model, draft_cache, tokenizer):
        super().__init__(model, cache, tokenizer)

        self.draft_model = draft_model
        self.draft_cache = draft_cache

        assert model.config.max_seq_len == draft_model.config.max_seq_len


    def generate_simple(self, prompt: str, gen_settings: ExLlamaV2Sampler.Settings, num_tokens: int, seed = None):

        if seed is not None: random.seed(seed)
        self.attempts = []
        self.hits = []

        # Create draft sampling settings

        draft_settings = gen_settings.clone()
        draft_settings.top_k = 1
        draft_settings.top_p = 0.0

        # Tokenize prompt

        input_ids = self.tokenizer.encode(prompt)

        overflow = input_ids.shape[-1] + num_tokens - self.model.config.max_seq_len
        if overflow > 0: input_ids = input_ids[:, -overflow:]

        # Feed prompt to model and draft model

        self.cache.current_seq_len = 0
        self.model.forward(input_ids[:, :-1], self.cache, preprocess_only = True)

        self.draft_cache.current_seq_len = 0
        self.draft_model.forward(input_ids[:, :-1], self.draft_cache, preprocess_only = True)

        self.sequence_ids = input_ids.clone()
        self.sequence_ids = input_ids

        # Generate

        predict_len = self.default_predict_len

        while num_tokens > 0:

            # Predict some tokens with the draft model

            predict_len = min(predict_len, num_tokens)
            predict_ids = self.sequence_ids[:, -1:]

            randoms = [random.random() for i in range(predict_len + 1)]

            used_predict_len = 0
            for i in range(predict_len):

                logits = self.draft_model.forward(predict_ids[:, -1:], self.draft_cache).float().cpu()
                past = torch.cat((self.sequence_ids[:, :-1], predict_ids), dim = 1)
                token, prob = ExLlamaV2Sampler.sample(logits, draft_settings, past, randoms[i])
                predict_ids = torch.cat([predict_ids, token], dim = 1)
                used_predict_len += 1
                if prob < self.prob_threshold: break

            # Forward (predict_len) tokens through full model

            logits = self.model.forward(predict_ids[:, :], self.cache).float().cpu()

            tokens = 0
            while True:

                token, _ = ExLlamaV2Sampler.sample(logits[:, tokens : tokens + 1, :], gen_settings, self.sequence_ids, randoms[tokens])
                self.sequence_ids = torch.cat([self.sequence_ids, token], dim = 1)
                tokens += 1
                if tokens == used_predict_len or token != predict_ids[:, tokens]: break

            # If all predictions matched, the full model has one more valid set of logits. Sample from it and feed the
            # result back through the draft model

            if tokens == used_predict_len and token == predict_ids[:, tokens]:

                token, _ = ExLlamaV2Sampler.sample(logits[:, tokens : tokens + 1, :], gen_settings, self.sequence_ids, randoms[tokens])
                self.sequence_ids = torch.cat([self.sequence_ids, token], dim = 1)
                tokens += 1
                self.draft_model.forward(self.sequence_ids[:, -1:], self.draft_cache, preprocess_only = True)
                used_predict_len += 1

            else:

                self.cache.current_seq_len -= 1

            # Record stats

            while len(self.attempts) < used_predict_len: self.attempts.append(0)
            self.attempts[used_predict_len - 1] += 1
            while len(self.hits) < tokens: self.hits.append(0)
            self.hits[tokens - 1] += 1

            # Roll back mismatched tokens

            mismatches = used_predict_len - tokens
            self.draft_cache.current_seq_len -= mismatches

            self.cache.current_seq_len -= mismatches

            # Dynamically adjust prediction length

            # if tokens == used_predict_len: predict_len += 1
            # if tokens < used_predict_len - 1: predict_len -= 1

            num_tokens -= tokens

        text = self.tokenizer.decode(self.sequence_ids[0])
        return text
