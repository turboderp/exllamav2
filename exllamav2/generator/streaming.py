from ast import Tuple
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

class ExLlamaV2StreamingGenerator(ExLlamaV2BaseGenerator):

    tail_decode_tokens: int = 2
    
    remaining_tokens: int = 0
    held_text: str = ""
    held_tokens: torch.Tensor = None
    settings: ExLlamaV2Sampler.Settings = None
    stop_strings: list = []
    stop_tokens: list = []
    no_tokens: torch.Tensor = None

    first_token = False
    heal_next_token = False

    def __init__(self, model, cache, tokenizer):
        super().__init__(model, cache, tokenizer)

        self.stop_strings = []
        self.stop_tokens = [tokenizer.eos_token_id]

        self.no_tokens = torch.empty((1, 0), dtype = torch.long)


    def set_stop_conditions(self, stop_conditions):

        assert isinstance(stop_conditions, list)

        self.stop_strings = []
        self.stop_tokens = []
        for t in stop_conditions:
            if isinstance(t, int): self.stop_tokens += [t]
            elif isinstance(t, str): self.stop_strings += [t]
            else: raise ValueError("Unsupported type in stop_conditions")
    
    
    def begin_stream(self, input_ids: torch.Tensor, gen_settings: ExLlamaV2Sampler.Settings, token_healing = False):

        self.held_text = ""
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
        self.held_text += new_tail[len(old_tail):]
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
    

    def _gen_begin(self, in_tokens, gen_settings):

        self.sequence_ids = in_tokens.clone()
        self.cache.current_seq_len = 0
        self.model.forward(self.sequence_ids[:, :-1], self.cache, preprocess_only = True)

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
        self.sequence_ids = in_tokens[:, :reuse]

        if reuse < in_tokens.shape[-1]: self._gen_feed_tokens(in_tokens[:, reuse:], gen_settings)
    

    def _gen_feed_tokens(self, in_tokens, gen_settings):

        if self.sequence_ids is None:
            self._gen_begin(in_tokens, gen_settings)
            return

        start = self.cache.current_seq_len
        self.sequence_ids = torch.cat((self.sequence_ids, in_tokens), dim = 1)

        self.model.forward(self.sequence_ids[:, start : -1], self.cache, preprocess_only = True)


    def _gen_single_token(self, gen_settings, prefix_token = None):

        logits = self.model.forward(self.sequence_ids[:, -1:], self.cache).float().cpu()
        token, _, eos = ExLlamaV2Sampler.sample(logits, gen_settings, self.sequence_ids, random.random(), self.tokenizer, prefix_token)
        self.sequence_ids = torch.cat([self.sequence_ids, token], dim = 1)
        gen_settings.feed_filters(token)
        return token, eos
