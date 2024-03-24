from __future__ import annotations
from dataclasses import dataclass, field
import torch
import torch.nn.functional as F
from exllamav2 import ExLlamaV2Tokenizer
from exllamav2.generator.filters import ExLlamaV2Filter
from exllamav2.generator.hooks import ExLlamaV2PostSamplingHook
from exllamav2.ext import exllamav2_ext as ext_c, none_tensor
from copy import copy

class ExLlamaV2Sampler:

    @dataclass
    class Settings:

        token_repetition_penalty: float = 1.05
        token_repetition_range: int = -1
        token_repetition_decay: int  = 0

        token_frequency_penalty: float = 0.0
        token_presence_penalty: float = 0.0

        temperature: float = 0.8
        smoothing_factor: float = 0.0
        min_temp: float = 0
        max_temp: float = 0.0
        temp_exponent: float = 1.0
        top_k: int = 50
        top_p: float = 0.8
        top_a: float = 0.0
        min_p: float = 0
        tfs: float = 0
        typical: float = 0
        skew: float = 0

        temperature_last: bool = False

        mirostat: bool = False
        mirostat_tau: float = 1.5
        mirostat_eta: float = 0.1
        mirostat_mu: float | None = None  # (re)initialized from mirostat_tau on first sample

        token_bias: torch.Tensor | None = None
        cfg_scale: float | None = None

        filters: list[ExLlamaV2Filter] = field(default_factory = list)
        post_sampling_hooks: list[ExLlamaV2PostSamplingHook] = field(default_factory = list)
        filter_prefer_eos = False


        def clone(self):
            c = copy(self)
            c.filters = [f.clone() for f in self.filters]
            return c


        def greedy_clone(self):
            c = ExLlamaV2Sampler.Settings()
            c.top_k = 1
            c.top_p = 0
            c.token_repetition_penalty = self.token_repetition_penalty
            c.token_repetition_range = self.token_repetition_range
            c.token_repetition_decay = self.token_repetition_decay
            c.token_frequency_penalty = self.token_frequency_penalty
            c.token_presence_penalty = self.token_presence_penalty
            c.token_bias = None
            c.filters = []
            return c


        def disallow_tokens(self,
                            tokenizer: ExLlamaV2Tokenizer,
                            tokens: list[int]):

            if self.token_bias is None:
                padding = -tokenizer.config.vocab_size % 32
                self.token_bias = torch.zeros((tokenizer.config.vocab_size + padding,), dtype = torch.float)

            self.token_bias[tokens] = float("-inf")


        def begin_filters(self, prefix_str = ""):

            for f in self.filters: f.begin(prefix_str)


        def feed_filters(self, feed_token):

            for f in self.filters: f.feed(feed_token)


    @staticmethod
    def sample(logits: torch.tensor,
               settings: Settings,
               sequence_ids: torch.tensor,
               random: float,
               tokenizer: ExLlamaV2Tokenizer,
               prefix_token: torch.Tensor | None = None,
               return_top_tokens: int = 0):
        """
        Sample tokens from (batched) logits tensor

        :param logits:
            Input logits, float tensor of shape (batch_size, 1, vocab_size)

        :param settings:
            ExLlamaV2Sampler.Settings

        :param sequence_ids:
            Past token IDs to consider for repetition penalty etc., shape (batch_size, seq_len)

        :param random:
            Float between 0 and 1, determining sampling point in the final normalized distribution.

        :param tokenizer:
            ExLlamaV2Tokenizer

        :param prefix_token:
            Tensor of shape (batch_size, 1). If provided, sampling will be restricted to token pieces that begin with
            this token. Used for token healing.

        :param return_top_tokens:
            Number of top tokens to return

        :return:
            Tuple of:
            - Sampled tokens, tensor of shape (batch_size, 1)
            - Top candidates per token (batch_size, 1, return_top_tokens), or meta tensor if return_top_tokens = 0
            - Top probs per token (batch_size, 1, return_top_tokens), or meta tensor if return_top_tokens = 0
            - Probabilities per token, shape (batch_size, 1)
            - True if the current filter has reached a stop condition
        """

        batch_size, _, vocab_size = logits.shape

        assert logits.shape[1] == 1, "Logits tensor is incorrect shape, must be (bsz, 1, vocab_size)"
        assert prefix_token is None or prefix_token.shape == (batch_size, 1), "Prefix token list doesn't match batch shape"
        if settings.cfg_scale is not None: assert batch_size == 2, "CFG requires logits to be bsz 2"
        else: assert batch_size == 1 or len(settings.filters) == 0, "Filters not implemented for batch size > 1"

        logits = logits.squeeze(1)

        # CFG

        if settings.cfg_scale is not None:

            logits = F.log_softmax(logits, dim = -1)
            logits = settings.cfg_scale * logits[0] + (1 - settings.cfg_scale) * logits[1]
            logits = logits.unsqueeze(0)
            batch_size = 1

        # Prepare filter

        logit_filter = torch.empty((batch_size, vocab_size), dtype = torch.bool)
        ext_c.fast_fill_cpu_ones_bool(logit_filter)

        # Repetition penalty

        if settings.token_repetition_penalty != 1.0 or \
            settings.token_frequency_penalty != 0.0 or \
            settings.token_presence_penalty != 0.0:

            ext_c.apply_rep_penalty(sequence_ids[:, :],
                                    settings.token_repetition_penalty,
                                    settings.token_repetition_range,
                                    settings.token_repetition_decay,
                                    settings.token_frequency_penalty,
                                    settings.token_presence_penalty,
                                    logits)

        # Token bias

        if settings.token_bias is not None:
            # logits = logits + settings.token_bias
            ext_c.fast_fadd_cpu(logits, settings.token_bias)

        # Evaluate filters

        if len(settings.filters) > 0:

            pass_tokens = None
            end_tokens = None
            for f in settings.filters:

                pt, et = f.next()
                if pt is not None: pass_tokens = pt if pass_tokens is None else pass_tokens & pt
                if et is not None: end_tokens = et if end_tokens is None else end_tokens | et

            if pass_tokens is not None:
                assert pass_tokens, "Filter excluded all tokens"
                if settings.filter_prefer_eos and tokenizer.eos_token_id in pass_tokens:
                    pass_tokens = { tokenizer.eos_token_id }
                ext_c.logit_filter_exclusive(logit_filter, [sorted(list(pass_tokens))])

        # Healing

        if prefix_token is not None:

            prefix_id_to_ids = tokenizer.get_prefix_id_to_ids_dict()

            valid_token_lists = []
            for i in range(batch_size):
                valid_token_lists.append(prefix_id_to_ids[prefix_token[i, 0].item()])

            ext_c.logit_filter_exclusive(logit_filter, valid_token_lists)

        # Begin Mirostat

        if settings.mirostat:
            if settings.mirostat_mu is None:
                settings.mirostat_mu = [0.0] * batch_size

        # Mask off logits if tokenizer's vocabulary is smaller than head layer

        vs = tokenizer.get_vocab_size()
        if vs < logits.shape[-1]:
            logits[:, vs:] = float("-inf")

        # Sampling

        batch_size = logits.shape[0]

        output_tokens = torch.empty((batch_size, 1), device = "cpu", dtype = torch.long)
        output_probs = torch.empty((batch_size, 1), device = "cpu", dtype = torch.float)
        if return_top_tokens == 0:
            output_ktokens = none_tensor
            output_kprobs = none_tensor
        else:
            output_ktokens = torch.empty((batch_size, 1, return_top_tokens), device = "cpu", dtype = torch.long)
            output_kprobs = torch.empty((batch_size, 1, return_top_tokens), device = "cpu", dtype = torch.float)

        m = ext_c.sample_basic(logits,
                               1.0 if settings.temperature_last else settings.temperature,
                               settings.top_k,
                               settings.top_p,
                               settings.top_a,
                               settings.min_p,
                               settings.tfs,
                               settings.typical,
                               random,
                               output_tokens,
                               output_probs,
                               output_kprobs,
                               output_ktokens,
                               logit_filter,
                               settings.mirostat,
                               settings.mirostat_mu if settings.mirostat else [],
                               settings.mirostat_tau,
                               settings.mirostat_eta,
                               settings.temperature if settings.temperature_last else 1.0,
                               settings.min_temp,
                               settings.max_temp,
                               settings.temp_exponent,
                               settings.smoothing_factor,
                               settings.skew)

        if settings.mirostat: settings.mirostat_mu = m

        # Stop condition from filters

        end_filter = False
        if len(settings.filters) > 0 and output_tokens[0].item() in end_tokens: end_filter = True

        return output_tokens, output_ktokens, output_kprobs, output_probs, end_filter
