import torch
from exllamav2 import ext
from exllamav2.ext import exllamav2_ext as ext_c, none_tensor


class ExLlamaV2Sampler:

    class Settings:

        temperature = 0.9
        top_k = 40
        top_p = 0.9

        token_repetition_penalty = 1.15
        token_repetition_range = -1
        token_repetition_decay = 0

        token_bias = None


        def clone(self):

            c = ExLlamaV2Sampler.Settings()
            c.temperature = self.temperature
            c.top_k = self.top_k
            c.top_p = self.top_p
            c.token_repetition_penalty = self.token_repetition_penalty
            c.token_repetition_range = self.token_repetition_range
            c.token_repetition_decay = self.token_repetition_decay
            c.token_bias = c.token_bias

            return c


        def disallow_tokens(self, tokenizer, tokens):

            if self.token_bias is None:
                self.token_bias = torch.zeros((tokenizer.config.vocab_size,), dtype = torch.float)

            self.token_bias[tokens] = float("-inf")


    @staticmethod
    def sample(logits: torch.tensor, settings: Settings, sequence_ids: torch.tensor, random: float):

        assert logits.shape[1] == 1, "Logits tensor is incorrect shape, must be (bsz, 1, vocab_size)"

        logits = logits.clone().squeeze(1)

        # Repetition penalty

        if settings.token_repetition_penalty != 1.0:

            ext_c.apply_rep_penalty(sequence_ids,
                                    settings.token_repetition_penalty,
                                    settings.token_repetition_range,
                                    settings.token_repetition_decay,
                                    logits)

        # Token bias

        if settings.token_bias is not None: logits += settings.token_bias

        # Sampling

        batch_size = logits.shape[0]

        output_tokens = torch.empty((batch_size, 1), device = "cpu", dtype = torch.long)
        output_probs = torch.empty((batch_size, 1), device = "cpu", dtype = torch.float)
        ext_c.sample_basic(logits,
                           settings.temperature,
                           settings.top_k,
                           settings.top_p,
                           random,
                           output_tokens,
                           output_probs)

        return output_tokens, output_probs









