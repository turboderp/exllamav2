from ast import Tuple
from exllamav2 import (
    ExLlamaV2,
    ExLlamaV2Cache,
    ExLlamaV2Tokenizer,
    ExLlamaV2Lora
)
from exllamav2.generator import (
    ExLlamaV2Sampler,
    ExLlamaV2BaseGenerator
)

import torch
import random

class ExLlamaV2StreamingGenerator(ExLlamaV2BaseGenerator):

    tail_decode_tokens: int = 2
    
    remaining_tokens: int = 0
    held_text: str = ""
    held_utf8_tokens: torch.tensor = None
    expect_utf8: int = 0
    held_tokens: torch.Tensor or None = None
    settings: ExLlamaV2Sampler.Settings = None
    stop_strings: list = []
    stop_tokens: list = []
    no_tokens: torch.Tensor = None

    first_token = False
    heal_next_token = False

    draft_model: ExLlamaV2 or None = None
    draft_cache: ExLlamaV2Cache or None = None

    future_logits: torch.tensor or None = None
    future_tokens: torch.tensor or None = None
    num_speculative_tokens: int
    speculative_prob_threshold: float = 0.25
    total_draft_tokens: int = 0
    total_tokens: int = 0
    accepted_draft_tokens: int = 0

    active_loras = []

    def __init__(self, model, cache, tokenizer, draft_model = None, draft_cache = None, num_speculative_tokens = 5):
        super().__init__(model, cache, tokenizer)

        self.stop_strings = []
        self.stop_tokens = [tokenizer.eos_token_id]

        self.no_tokens = torch.empty((1, 0), dtype = torch.long)

        if draft_model:
            self.draft_model = draft_model
            self.num_speculative_tokens = num_speculative_tokens
            if draft_cache:
                self.draft_cache = draft_cache
            else:
                self.draft_cache = ExLlamaV2Cache(draft_model,
                                                  batch_size = cache.batch_size,
                                                  max_seq_len = cache.max_seq_len)


    def set_stop_conditions(self, stop_conditions):

        assert isinstance(stop_conditions, list)

        self.stop_strings = []
        self.stop_tokens = []
        for t in stop_conditions:
            if isinstance(t, int): self.stop_tokens += [t]
            elif isinstance(t, str): self.stop_strings += [t]
            else: raise ValueError("Unsupported type in stop_conditions")
    
    
    def begin_stream(self, input_ids: torch.Tensor, gen_settings: ExLlamaV2Sampler.Settings, token_healing = False, loras = None):

        # Accept LoRA or list of LoRAs
        if loras is not None and isinstance(loras, ExLlamaV2Lora): loras = [loras]
        self.active_loras = loras

        self.held_text = ""
        self.held_utf8_tokens = self.no_tokens
        self.expect_utf8 = 0
        self.held_tokens = self.no_tokens
        self.settings = gen_settings
        self._gen_begin_reuse(input_ids, gen_settings)

        self.heal_next_token = (token_healing and self.sequence_ids.shape[-1] >= 2)


    # Get the next chunk of text in the stream. Returns eos if stop condition has been met but does not count tokens

    def stream(self) -> (str, bool, torch.Tensor):

        # Token healing

        if self.heal_next_token:

            # Pop the last toke

            old_tail = self.tokenizer.decode(self.sequence_ids[:, -self.tail_decode_tokens:])[0]
            last_token = self.sequence_ids[:, -1:]
            self.sequence_ids = self.sequence_ids[:, :-1]
            self.cache.current_seq_len -= 1

            # Start filters

            if self.first_token:

                self.settings.begin_filters(self.tokenizer.get_id_to_piece_list()[last_token])
                self.first_token = False

            # Regenerate the last token again, with prefix

            healed_token, eos = self._gen_single_token(self.settings, prefix_token = last_token)
            new_tail = self.tokenizer.decode(self.sequence_ids[:, -self.tail_decode_tokens:])[0]
            self.held_text += new_tail[len(old_tail):]

            self.heal_next_token = False

            # In case we only needed the healed token

            if eos: return self.held_text, True, self.no_tokens

        # Start filters when not healing

        else:

            if self.first_token:

                self.settings.begin_filters()
                self.first_token = False


        # Decode the current tail end of the sequence

        old_tail = self.tokenizer.decode(self.sequence_ids[:, -self.tail_decode_tokens:])[0]

        # Generate a single token and append to the sequence

        next_token, eos = self._gen_single_token(self.settings)

        # End immediately if it was a stop token

        if next_token in self.stop_tokens:
            return self.held_text, True, self.no_tokens

        # Decode the tail end of the sequence with the added token to get (actual) characters added

        new_tail = self.tokenizer.decode(self.sequence_ids[:, -(self.tail_decode_tokens + 1):])[0]
        new_text = new_tail[len(old_tail):]

        next_token, new_text = self._catch_utf8(next_token, new_text)

        self.held_text += new_text
        self.held_tokens = torch.cat([self.held_tokens, next_token], dim = -1)

        # Return now if newly added token ends a filter

        if eos: return self.held_text, True, self.held_tokens

        # Hold text as long as it contains part of a stop string

        partial_ss = False
        for ss in self.stop_strings:

            # Check if held_text fully contains stop string

            position = self.held_text.find(ss)
            if position != -1:
                return self.held_text[:position], True, self.no_tokens

            # Check for overlap between end of held_text and start of stop string

            overlap = 0
            for j in range(1, min(len(self.held_text), len(ss)) + 1):
                if self.held_text[-j:] == ss[:j]: overlap = j
            if overlap > 0: partial_ss = True

        # If holding text because of a partial stop condition, return nothing but also EOS = False

        if partial_ss:
            return "", False, self.no_tokens

        # No stop condition, so return whatever is being held

        stream_text = self.held_text
        stream_tokens = self.held_tokens
        self.held_text = ""
        self.held_tokens = self.no_tokens
        return stream_text, False, stream_tokens
    

    def _decode_utf8(self):

        if self.held_utf8_tokens.shape[-1] == 0: return self.no_tokens, ""

        try:
            id_to_ord = self.tokenizer.get_id_to_ord_list()
            b = [id_to_ord[x] for x in self.held_utf8_tokens[0].tolist()]
            c = bytes(b).decode('utf-8')
        except UnicodeDecodeError:
            c = "�"

        pre_t = self.held_utf8_tokens
        self.held_utf_tokens = self.no_tokens
        return pre_t, c


    def _catch_utf8(self, next_token, new_text):

        if self.expect_utf8 == 0:

            if new_text != "�": return next_token, new_text

            id_to_ord = self.tokenizer.get_id_to_ord_list()
            t = next_token[0, 0].item()
            b = id_to_ord[t]

            if 0 < b < 256:
                if b & 0b1100000 == 0b1000000: self.expect_utf8 = 2
                if b & 0b1110000 == 0b1100000: self.expect_utf8 = 3
                if b & 0b1111000 == 0b1110000: self.expect_utf8 = 4
                if b & 0b1111100 == 0b1111000: self.expect_utf8 = 5
            self.held_utf8_tokens = self.no_tokens
            if self.expect_utf8 == 0: return next_token, new_text
            new_text = ""

        if self.expect_utf8:

            if len(new_text) > 1:

                pre_t, pre_c = self._decode_utf8()
                next_token = torch.cat((pre_t, next_token), dim = -1)
                new_text = pre_c + new_text
                return next_token, new_text

            self.held_utf8_tokens = torch.cat((self.held_utf8_tokens, next_token), dim = -1)
            self.expect_utf8 -= 1
            if self.expect_utf8 == 0: return self._decode_utf8()
            return self.no_tokens, ""


    def _gen_begin(self, in_tokens, gen_settings):

        self.sequence_ids = in_tokens.clone()
        self.cache.current_seq_len = 0
        self.model.forward(self.sequence_ids[:, :-1], self.cache, preprocess_only = True, loras = self.active_loras)

        if self.draft_model is not None:
            self.draft_cache.current_seq_len = 0
            self.draft_model.forward(self.sequence_ids[:, :-1], self.draft_cache, preprocess_only = True)
            self.future_logits = None
            self.future_tokens = None

        self.first_token = True


    def _gen_begin_reuse(self, in_tokens, gen_settings):

        if self.sequence_ids is None or self.cache.current_seq_len == 0:
            self._gen_begin(in_tokens, gen_settings)
            return

        reuse = 0
        while reuse < self.sequence_ids.shape[-1] and reuse < in_tokens.shape[-1] and self.sequence_ids[0, reuse] == in_tokens[0, reuse]:
            reuse += 1

        if reuse < 2:
            self._gen_begin(in_tokens, gen_settings)
            return

        self.cache.current_seq_len = reuse - 1
        if self.draft_model is not None:
            self.draft_cache.current_seq_len = reuse - 1
        self.sequence_ids = in_tokens[:, :reuse]

        if reuse < in_tokens.shape[-1]: self._gen_feed_tokens(in_tokens[:, reuse:], gen_settings)

        if self.draft_model is not None:
            self.future_logits = None
            self.future_tokens = None


    def _gen_feed_tokens(self, in_tokens, gen_settings):

        if self.sequence_ids is None:
            self._gen_begin(in_tokens, gen_settings)
            return

        start = self.cache.current_seq_len
        self.sequence_ids = torch.cat((self.sequence_ids, in_tokens), dim = 1)

        self.model.forward(self.sequence_ids[:, start : -1], self.cache, preprocess_only = True, loras = self.active_loras)

        if self.draft_model is not None:
            self.draft_model.forward(self.sequence_ids[:, start: -1], self.draft_cache, preprocess_only = True)
            self.future_logits = None
            self.future_tokens = None


    def _gen_single_token(self, gen_settings, prefix_token = None):

        if self.draft_model is None:

            logits = self.model.forward(self.sequence_ids[:, -1:], self.cache, loras = self.active_loras).float().cpu()
            token, _, eos = ExLlamaV2Sampler.sample(logits, gen_settings, self.sequence_ids, random.random(), self.tokenizer, prefix_token)

        else:

            token, eos = self._gen_single_token_speculative(gen_settings, prefix_token)

        self.sequence_ids = torch.cat([self.sequence_ids, token], dim = 1)
        gen_settings.feed_filters(token)
        return token, eos


    def _gen_single_token_speculative(self, gen_settings, prefix_token = None):

        if self.future_tokens is None:

            # Generate draft

            draft_gen_settings = gen_settings.greedy_clone()
            draft_sequence_ids = self.sequence_ids.clone()
            num_drafted_tokens = 0

            for k in range(self.num_speculative_tokens):

                logits = self.draft_model.forward(draft_sequence_ids[:, -1:], self.draft_cache).float().cpu()
                token, prob, _ = ExLlamaV2Sampler.sample(logits, draft_gen_settings, draft_sequence_ids, random.random(), self.tokenizer, prefix_token if k == 0 else None)

                if prob < self.speculative_prob_threshold:
                    self.draft_cache.current_seq_len -= 1
                    break

                draft_sequence_ids = torch.cat((draft_sequence_ids, token), dim = 1)
                num_drafted_tokens += 1

            self.total_draft_tokens += num_drafted_tokens

            # Rewind draft cache

            self.draft_cache.current_seq_len -= num_drafted_tokens

            # Forward last sampled token plus draft through model

            self.future_tokens = draft_sequence_ids[:, -1 - num_drafted_tokens:]
            self.future_logits = self.model.forward(self.future_tokens, self.cache, loras = self.active_loras).float().cpu()

            # Rewind model cache

            self.cache.current_seq_len -= num_drafted_tokens + 1

        # Sample the first future logits

        token, _, eos = ExLlamaV2Sampler.sample(self.future_logits[:, :1, :], gen_settings, self.sequence_ids, random.random(), self.tokenizer, prefix_token)
        self.future_logits = self.future_logits[:, 1:, :]
        self.future_tokens = self.future_tokens[:, 1:]
        self.cache.current_seq_len += 1
        self.draft_cache.current_seq_len += 1

        # If sampled token doesn't match future token or no more future tokens

        if self.future_tokens.shape[-1] == 0 or self.future_tokens[0, 0] != token[0, 0]:
            self.future_tokens = None
            self.future_logits = None
        else:
            self.accepted_draft_tokens += 1
        self.total_tokens += 1

        return token, eos


    def reset_sd_stats(self):

        self.total_tokens = 0
        self.total_draft_tokens = 0
        self.accepted_draft_tokens = 0


    def get_sd_stats(self):

        efficiency = self.accepted_draft_tokens / self.total_tokens
        accuracy = self.accepted_draft_tokens / self.total_draft_tokens
        return efficiency, accuracy, self.total_tokens, self.total_draft_tokens, self.accepted_draft_tokens