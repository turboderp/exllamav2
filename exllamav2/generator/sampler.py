import torch
from exllamav2 import ExLlamaV2Tokenizer
from exllamav2.ext import exllamav2_ext as ext_c, none_tensor


class ExLlamaV2Sampler:

    class Settings:

        token_repetition_penalty = 1.15
        token_repetition_range = -1
        token_repetition_decay = 0

        temperature = 0.9
        top_k = 40
        top_p = 0.9
        min_p = 0
        tfs = 0
        typical = 0

        temperature_last = False

        mirostat = False
        mirostat_tau = 1.5
        mirostat_eta = 0.1
        mirostat_mu = None  # (re)initialized from mirostat_tau on first sample

        token_bias = None

        filters = []


        def clone(self):

            c = ExLlamaV2Sampler.Settings()

            c.token_repetition_penalty = self.token_repetition_penalty
            c.token_repetition_range = self.token_repetition_range
            c.token_repetition_decay = self.token_repetition_decay

            c.temperature = self.temperature
            c.top_k = self.top_k
            c.top_p = self.top_p
            c.min_p = self.min_p
            c.tfs = self.tfs
            c.typical = self.typical

            c.mirostat = self.mirostat
            c.mirostat_tau = self.mirostat_tau
            c.mirostat_eta = self.mirostat_eta
            c.mirostat_mu = None if self.mirostat_mu is None else self.mirostat_mu.copy()

            c.token_bias = self.token_bias
            c.filters = [f.clone() for f in self.filters]

            return c


        def greedy_clone(self):

            c = ExLlamaV2Sampler.Settings()
            c.top_k = 1
            c.top_p = 0
            c.token_repetition_penalty = self.token_repetition_penalty
            c.token_repetition_range = self.token_repetition_range
            c.token_repetition_decay = self.token_repetition_decay
            c.token_bias = self.token_bias
            c.filters = []
            return c


        def disallow_tokens(self, tokenizer, tokens):

            if self.token_bias is None:
                padding = -tokenizer.config.vocab_size % 32
                self.token_bias = torch.zeros((tokenizer.config.vocab_size + padding,), dtype = torch.float)

            self.token_bias[tokens] = float("-inf")


        def begin_filters(self, prefix_str = ""):

            for f in self.filters: f.begin(prefix_str)


        def feed_filters(self, feed_token):

            for f in self.filters: f.feed(feed_token)


    @staticmethod
    def sample(logits: torch.tensor, settings: Settings, sequence_ids: torch.tensor, random: float, tokenizer: ExLlamaV2Tokenizer, prefix_token = None):

        batch_size, _, vocab_size = logits.shape

        assert logits.shape[1] == 1, "Logits tensor is incorrect shape, must be (bsz, 1, vocab_size)"
        assert prefix_token is None or prefix_token.shape == (batch_size, 1), "Prefix token list doesn't match batch shape"
        assert batch_size == 1 or len(settings.filters) == 0, "Filters not implemented for batch size > 1"

        logits = logits.clone().squeeze(1)
        logit_filter = torch.ones((batch_size, vocab_size), dtype = torch.bool)

        # Repetition penalty

        if settings.token_repetition_penalty != 1.0:

            ext_c.apply_rep_penalty(sequence_ids,
                                    settings.token_repetition_penalty,
                                    settings.token_repetition_range,
                                    settings.token_repetition_decay,
                                    logits)

        # Token bias

        if settings.token_bias is not None: logits += settings.token_bias

        # Evaluate filters

        if len(settings.filters) > 0:

            pass_tokens = None
            end_tokens = None
            for f in settings.filters:

                pt, et = f.next()
                pass_tokens = pt if pass_tokens is None else pass_tokens & pt
                end_tokens = et if end_tokens is None else end_tokens | et

            assert pass_tokens, "Filter excluded all tokens"
            ext_c.logit_filter_exclusive(logit_filter, [sorted(list(pass_tokens))])

        # Healing

        if prefix_token is not None:

            prefix_id_to_ids = tokenizer.get_prefix_id_to_ids_dict()

            valid_token_lists = []
            for i in range(batch_size):
                valid_token_lists.append(prefix_id_to_ids[prefix_token[i, 0].item()])

            ext_c.logit_filter_exclusive(logit_filter, valid_token_lists)

        # for i in range(logit_filter.shape[-1]):
        #     if logit_filter[0, i].item():
        #         print(i)

        # Begin Mirostat

        if settings.mirostat:
            if settings.mirostat_mu is None:
                settings.mirostat_mu = [0.0] * batch_size

        # Sampling

        batch_size = logits.shape[0]

        output_tokens = torch.empty((batch_size, 1), device = "cpu", dtype = torch.long)
        output_probs = torch.empty((batch_size, 1), device = "cpu", dtype = torch.float)

        m = ext_c.sample_basic(logits,
                               1.0 if settings.temperature_last else settings.temperature,
                               settings.top_k,
                               settings.top_p,
                               settings.min_p,
                               settings.tfs,
                               settings.typical,
                               random,
                               output_tokens,
                               output_probs,
                               logit_filter,
                               settings.mirostat,
                               settings.mirostat_mu if settings.mirostat else [],
                               settings.mirostat_tau,
                               settings.mirostat_eta,
                               settings.temperature if settings.temperature_last else 1.0)

        if settings.mirostat: settings.mirostat_mu = m

        # Stop condition from filters

        end_filter = False
        if len(settings.filters) > 0 and output_tokens[0].item() in end_tokens: end_filter = True

        return output_tokens, output_probs, end_filter
