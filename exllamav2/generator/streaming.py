from ast import Tuple
from typing import Union, Tuple
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
from exllamav2.generator.ngram import NgramCache
import torch
import random
import threading

class ExLlamaV2StreamingGenerator(ExLlamaV2BaseGenerator):

    tail_decode_tokens: int = 2

    settings: ExLlamaV2Sampler.Settings = None
    stop_strings: set = set()
    stop_tokens: set = set()
    draft_model: ExLlamaV2 or None = None
    draft_cache: ExLlamaV2Cache or None = None

    remaining_tokens: int = 0
    held_utf8_tokens: torch.tensor = None
    held_fallback_tokens: torch.tensor = None
    max_fallback_tokens: int = 4
    expect_utf8: int = 0

    no_tokens: torch.Tensor = None
    no_ptokens: torch.Tensor = None
    no_probs: torch.Tensor = None
    no_pprobs: torch.Tensor = None
    no_logits: torch.Tensor = None

    held_text: str = ""
    held_tokens: torch.Tensor or None = None
    held_ptokens: torch.Tensor or None = None
    held_probs: torch.Tensor or None = None
    held_pprobs: torch.Tensor or None = None
    held_logits: torch.Tensor or None = None

    first_token = False
    heal_next_token = False

    future_logits: torch.tensor or None = None
    future_tokens: torch.tensor or None = None
    num_speculative_tokens: int
    speculative_prob_threshold: float = 0.25
    total_draft_tokens: int = 0
    total_tokens: int = 0
    accepted_draft_tokens: int = 0

    return_probabilities: bool = False
    return_top_tokens: int = 0
    return_logits: bool = False

    active_loras = []
    position_offsets = None
    input_mask = None

    speculative_ngram = False
    speculative_ngram_min = 1
    speculative_ngram_max = 5
    speculative_ngram_max_inf = 3
    speculative_ngram_threshold = 1
    ngram = None
    ngram_preloaded = None


    def __init__(self, model, cache, tokenizer, draft_model = None, draft_cache = None, num_speculative_tokens = 5):
        super().__init__(model, cache, tokenizer)

        self.stop_strings = set()
        self.stop_tokens = {tokenizer.eos_token_id,}

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

        assert isinstance(stop_conditions, (list, tuple, set))

        self.stop_strings = set()
        self.stop_tokens = set()
        for t in stop_conditions:
            if isinstance(t, int): self.stop_tokens.add(t)
            elif isinstance(t, str): self.stop_strings.add(t)
            else: raise ValueError("Unsupported type in stop_conditions")


    # Begin stream

    def begin_stream_ex(self,
                       input_ids: torch.Tensor,
                       gen_settings: ExLlamaV2Sampler.Settings,
                       token_healing = False,
                       loras = None,
                       input_mask = None,
                       position_offsets = None,
                       return_probabilities = False,
                       return_top_tokens = 0,
                       return_logits = False,
                       abort_event: threading.Event = None):

        self.return_probabilities = return_probabilities
        self.return_top_tokens = return_top_tokens
        self.return_logits = return_logits

        self.begin_stream(input_ids,
                          gen_settings,
                          token_healing,
                          loras,
                          input_mask,
                          position_offsets,
                          abort_event)


    # Legacy function

    def begin_stream(self,
                     input_ids: torch.Tensor,
                     gen_settings: ExLlamaV2Sampler.Settings,
                     token_healing = False,
                     loras = None,
                     input_mask = None,
                     position_offsets = None,
                     abort_event: threading.Event = None):

        self.abort_event = abort_event

        assert input_ids.shape[0] <= 2, "Streaming generator does not support batch size > 1"
        if input_ids.shape[0] == 2:
            assert gen_settings.cfg_scale is not None, "No CFG scale set"

        self.position_offsets = position_offsets
        self.input_mask = input_mask

        # Accept LoRA or list of LoRAs
        if loras is not None and isinstance(loras, ExLlamaV2Lora): loras = [loras]
        self.active_loras = loras

        self.no_logits = torch.empty((0, ((self.model.config.vocab_size + 31) // 32) * 32), dtype=torch.float)
        self.no_tokens = torch.empty((1, 0), dtype=torch.long)
        self.no_probs = torch.empty((1, 0), dtype=torch.float)
        self.no_ptokens = torch.empty((1, 0, self.return_top_tokens), dtype=torch.long)
        self.no_pprobs = torch.empty((1, 0, self.return_top_tokens), dtype=torch.float)

        self.held_text = ""
        self.held_utf8_tokens = self.no_tokens
        self.held_fallback_tokens = self.no_tokens
        self.expect_utf8 = 0
        self.held_tokens = self.no_tokens
        self.held_ptokens = self.no_ptokens
        self.held_probs = self.no_probs
        self.held_pprobs = self.no_pprobs
        self.held_logits = self.no_logits
        self.settings = gen_settings
        self._gen_begin_reuse(input_ids, gen_settings)

        self.heal_next_token = (token_healing and self.sequence_ids.shape[-1] >= 2)

        if self.speculative_ngram:
            self.ngram = NgramCache(self.speculative_ngram_min, self.speculative_ngram_max, self.ngram_preloaded)
            self.ngram.update(self.sequence_ids[0].tolist())


    # Get the next chunk of text in the stream

    def stream_ex(self):

        chunk, eos, chunk_token_ids, probs, ptokens, pprobs, logits = self._stream()

        ret = { "chunk": chunk,                         # output text
                "eos": eos,                             # True if stop condition met
                "chunk_token_ids": chunk_token_ids }    # token(s) corresponding to output text: [1, n] (torch.long)

        if self.return_probabilities:
            ret["probs"] = probs                        # Probability of selected token(s): [1, n] (torch.float)

        if self.return_top_tokens > 0:
            ret["top_probs"] = pprobs                   # Top k probs: [1, n, k] (torch.float)
            ret["top_tokens"] = ptokens                 # Top k tokens: [1, n, k] (torch.long)

        if self.return_logits:
            ret["logits"] = logits.unsqueeze(0)         # Raw output logits: [1, n, vocab_size] (torch.float)

        return ret


    # Legacy function

    def stream(self) -> Union[Tuple[str, bool, torch.Tensor],
                              Tuple[str, bool, torch.Tensor, torch.Tensor],
                              Tuple[str, bool, torch.Tensor, torch.Tensor, torch.Tensor]]:

        assert self.return_top_tokens == 0, "Use stream_ex() to return top K probs"

        chunk, eos, chunk_token_ids, probs, _, _, logits = self._stream()
        ret = [chunk, eos, chunk_token_ids]

        if self.return_probabilities:
            ret.append(probs)

        if self.return_logits:
            ret.append(logits)
        
        return tuple(ret)


    # @profile
    def _stream(self) -> (str, bool, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor):

        # Token healing

        if self.heal_next_token:

            # Pop the last token

            old_tail = self.tokenizer.decode(self.sequence_ids[:, -self.tail_decode_tokens:])[0]
            last_token = self.sequence_ids[:, -1:]
            self.sequence_ids = self.sequence_ids[:, :-1]
            self.cache.current_seq_len -= 1

            # Start filters

            if self.first_token:

                self.settings.begin_filters(self.tokenizer.get_id_to_piece_list()[last_token])
                self.first_token = False

            # Regenerate the last token again, with prefix

            healed_token, _, _, _, eos, logits = self._gen_single_token(self.settings, prefix_token = last_token)
            new_tail = self.tokenizer.decode(self.sequence_ids[:, -self.tail_decode_tokens:])[0]
            self.held_text += new_tail[len(old_tail):]

            self.heal_next_token = False

            # In case we only needed the healed token

            if eos: return self.held_text, True, self.no_tokens, self.no_probs, self.no_ptokens, self.no_pprobs, self.no_logits

        # Start filters when not healing

        else:

            if self.first_token:

                self.settings.begin_filters()
                self.first_token = False

        # Generate a single token and append to the sequence

        next_token, next_ptokens, next_pprobs, next_prob, eos, next_logits = self._gen_single_token(self.settings)

        # End immediately if it was a stop token

        if next_token.item() in self.stop_tokens:
            return self.held_text, True, self.no_tokens, self.no_probs, self.no_ptokens, self.no_pprobs, self.no_logits

        id_to_piece = self.tokenizer.get_id_to_piece_list()
        new_text = id_to_piece[next_token]

        next_token, new_text = self._catch_utf8(next_token, new_text)
        next_token, new_text = self._catch_fallback(next_token, new_text)

        self.held_text += new_text
        self.held_tokens = torch.cat([self.held_tokens, next_token], dim = -1)
        if self.return_probabilities:
            self.held_probs = torch.cat([self.held_probs, next_prob], dim = 1)
        if self.return_top_tokens > 0:
            self.held_ptokens = torch.cat([self.held_ptokens, next_ptokens], dim = 1)
            self.held_pprobs = torch.cat([self.held_pprobs, next_pprobs], dim = 1)
        if self.return_logits:
            self.held_logits = torch.cat([self.held_logits, next_logits], dim = 0)

        # Return now if newly added token ends a filter

        if eos: return self.held_text, True, self.held_tokens, self.held_probs, self.held_ptokens, self.held_pprobs, self.held_logits

        # Hold text as long as it contains part of a stop string

        partial_ss = False
        for ss in self.stop_strings:

            # Check if held_text fully contains stop string

            position = self.held_text.find(ss)
            if position != -1:
                return self.held_text[:position], True, self.no_tokens, self.no_probs, self.no_ptokens, self.no_pprobs, self.no_logits

            # Check for overlap between end of held_text and start of stop string

            overlap = 0
            for j in range(1, min(len(self.held_text), len(ss)) + 1):
                if self.held_text[-j:] == ss[:j]: overlap = j
            if overlap > 0: partial_ss = True

        # If holding text because of a partial stop condition, return nothing but also EOS = False

        if partial_ss:
            return "", False, self.no_tokens, self.no_probs, self.no_ptokens, self.no_pprobs, self.no_logits

        # No stop condition, so return whatever is being held

        stream_text = self.held_text
        stream_tokens = self.held_tokens
        stream_probs = self.held_probs
        stream_ptokens = self.held_ptokens
        stream_pprobs = self.held_pprobs
        stream_logits = self.held_logits
        self.held_text = ""
        self.held_tokens = self.no_tokens
        self.held_probs = self.no_probs
        self.held_ptokens = self.no_ptokens
        self.held_pprobs = self.no_pprobs
        self.held_logits = self.no_logits
        return stream_text, False, stream_tokens, stream_probs, stream_ptokens, stream_pprobs, stream_logits


    def _decode_utf8(self):

        if self.held_utf8_tokens.shape[-1] == 0: return self.no_tokens, ""

        try:
            id_to_ord = self.tokenizer.get_id_to_ord_list()
            b = [id_to_ord[x] for x in self.held_utf8_tokens[0].tolist()]
            c = bytes(b).decode('utf-8')
        except ValueError or UnicodeDecodeError:
            id_to_piece = self.tokenizer.get_id_to_piece_list()
            c = "".join(id_to_piece[x] for x in self.held_utf8_tokens[0].tolist())
        # except UnicodeDecodeError:
        #     c = "�"

        pre_t = self.held_utf8_tokens
        self.held_utf8_tokens = self.no_tokens
        return pre_t, c


    def _catch_fallback(self, next_token, new_text):

        if self.held_fallback_tokens.shape[-1] == 0:
            if "�" not in new_text: return next_token, new_text

        self.held_fallback_tokens = torch.cat((self.held_fallback_tokens, next_token), dim = -1)
        new_decode = self.tokenizer.decode(self.held_fallback_tokens)[0]

        if "�" not in new_decode or self.held_fallback_tokens.shape[-1] >= self.max_fallback_tokens:
            r_tokens = self.held_fallback_tokens
            self.held_fallback_tokens = self.no_tokens
            return r_tokens, new_decode

        return self.no_tokens, ""


    def _catch_utf8(self, next_token, new_text):

        if self.held_fallback_tokens.shape[-1] > 0:
            return next_token, new_text

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


    def _truncate_seq_to_cache(self):
        cachelen = self.cache.current_seq_len
        if self.draft_cache: cachelen = min(cachelen, self.draft_cache.current_seq_len)
        self.sequence_ids = self.sequence_ids[:, :cachelen + 1]


    def _gen_begin(self, in_tokens, gen_settings):

        self.sequence_ids = in_tokens.clone()
        self.cache.current_seq_len = 0
        self.model.forward(self.sequence_ids[:, :-1],
                           self.cache,
                           preprocess_only = True,
                           loras = self.active_loras,
                           input_mask = self.input_mask,
                           position_offsets = self.position_offsets,
                           abort_event = self.abort_event)
        if self.abort_event and self.abort_event.is_set():
            self._truncate_seq_to_cache()
            return

        if self.draft_model is not None:
            self.draft_cache.current_seq_len = 0
            self.draft_model.forward(self.sequence_ids[:1, :-1],
                                     self.draft_cache,
                                     input_mask = self.input_mask,
                                     position_offsets = self.position_offsets,
                                     preprocess_only = True,
                                     abort_event = self.abort_event)
            if self.abort_event and self.abort_event.is_set():
                self._truncate_seq_to_cache()
                return
            self.future_logits = None
            self.future_tokens = None

        if self.speculative_ngram:
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

        if self.speculative_ngram or self.draft_model is not None:
            self.future_logits = None
            self.future_tokens = None

        self.first_token = True


    def _gen_feed_tokens(self, in_tokens, gen_settings):

        if self.sequence_ids is None:
            self._gen_begin(in_tokens, gen_settings)
            return

        start = self.cache.current_seq_len
        self.sequence_ids = torch.cat((self.sequence_ids, in_tokens), dim = 1)

        self.model.forward(self.sequence_ids[:, start : -1],
                           self.cache,
                           preprocess_only = True,
                           loras = self.active_loras,
                           input_mask = self.input_mask,
                           position_offsets = self.position_offsets,
                           abort_event = self.abort_event)
        if self.abort_event and self.abort_event.is_set():
            self._truncate_seq_to_cache()
            return

        if self.draft_model is not None:
            self.draft_model.forward(self.sequence_ids[:, start: -1],
                                     self.draft_cache,
                                     preprocess_only = True,
                                     input_mask = self.input_mask,
                                     position_offsets = self.position_offsets,
                                     abort_event = self.abort_event)
            if self.abort_event and self.abort_event.is_set():
                self._truncate_seq_to_cache()
                return
            self.future_logits = None
            self.future_tokens = None


    def _gen_single_token(self, gen_settings, prefix_token = None):

        if self.speculative_ngram:

            token, ptokens, pprobs, prob, eos, logits = self._gen_single_token_ngram(gen_settings, prefix_token)

        elif self.draft_model is None:

            logits = self.model.forward(self.sequence_ids[:, -1:],
                                        self.cache,
                                        loras = self.active_loras,
                                        input_mask = self.input_mask,
                                        position_offsets = self.position_offsets).float().cpu()

            token, ptokens, pprobs, prob, eos = \
                ExLlamaV2Sampler.sample(logits,
                                        gen_settings,
                                        self.sequence_ids[:1, :],
                                        random.random(),
                                        self.tokenizer,
                                        prefix_token,
                                        self.return_top_tokens)

        else:

            token, ptokens, pprobs, prob, eos, logits = \
                self._gen_single_token_speculative(gen_settings, prefix_token)

        if self.sequence_ids.shape[0] > 1 and token.shape[0] == 1:
            self.sequence_ids = torch.cat([self.sequence_ids, token.repeat(self.sequence_ids.shape[0], 1)], dim = 1)
        else:
            self.sequence_ids = torch.cat([self.sequence_ids, token], dim = 1)

        gen_settings.feed_filters(token)
        return token, ptokens, pprobs, prob, eos, logits.flatten(1)


    def _gen_single_token_speculative(self, gen_settings, prefix_token = None):

        if self.future_tokens is None:

            # Generate draft

            draft_gen_settings = gen_settings.greedy_clone()
            draft_sequence_ids = self.sequence_ids[:1, :]
            num_drafted_tokens = 0

            for k in range(self.num_speculative_tokens):

                logits = self.draft_model.forward(draft_sequence_ids[:, -1:],
                                                  self.draft_cache,
                                                  input_mask = self.input_mask,
                                                  position_offsets = self.position_offsets).float().cpu()
                token, _, _, prob, _ = ExLlamaV2Sampler.sample(logits, draft_gen_settings, draft_sequence_ids, random.random(), self.tokenizer, prefix_token if k == 0 else None)

                if prob < self.speculative_prob_threshold:
                    self.draft_cache.current_seq_len -= 1
                    break

                draft_sequence_ids = torch.cat((draft_sequence_ids, token), dim = 1)
                num_drafted_tokens += 1

            self.total_draft_tokens += num_drafted_tokens

            # Rewind draft cache

            self.draft_cache.current_seq_len -= num_drafted_tokens

            # Forward last sampled token plus draft through model

            if self.sequence_ids.shape[0] > 1:
                self.future_tokens = draft_sequence_ids[:, -1 - num_drafted_tokens:].repeat(self.sequence_ids.shape[0], 1)
            else:
                self.future_tokens = draft_sequence_ids[:, -1 - num_drafted_tokens:]
            self.future_logits = self.model.forward(self.future_tokens,
                                                    self.cache,
                                                    loras = self.active_loras,
                                                    input_mask = self.input_mask,
                                                    position_offsets = self.position_offsets).float().cpu()

            # Rewind model cache

            self.cache.current_seq_len -= num_drafted_tokens + 1

        # Sample the first future logits

        logits = self.future_logits[:, :1, :]
        token, ptokens, pprobs, prob, eos = ExLlamaV2Sampler.sample(logits, gen_settings, self.sequence_ids[:1, :], random.random(), self.tokenizer, prefix_token, self.return_top_tokens)
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

        return token, ptokens, pprobs, prob, eos, logits


    def _gen_single_token_ngram(self, gen_settings, prefix_token = None):

        if self.future_tokens is None:

            inf_ids = self.sequence_ids[0, -1:].tolist()
            pred_ids = self.sequence_ids[0, -self.speculative_ngram_max:].tolist()

            threshold = self.speculative_ngram_threshold
            while len(inf_ids) < self.speculative_ngram_max_inf:
                t = self.ngram.predict_next(pred_ids, threshold, self.ngram_preloaded)
                if t is None: break
                pred_ids = pred_ids[1:] + [t]
                inf_ids += [t]
                threshold += 1

            if len(inf_ids) + self.cache.current_seq_len > self.cache.max_seq_len:
                inf_ids = inf_ids[:-(len(inf_ids) + self.cache.current_seq_len - self.cache.max_seq_len)]

            self.future_tokens = torch.tensor([inf_ids], dtype = torch.long)
            self.future_logits = self.model.forward(self.future_tokens,
                                                    self.cache,
                                                    loras = self.active_loras,
                                                    input_mask = self.input_mask,
                                                    position_offsets = self.position_offsets)
            self.future_logits = self.future_logits.float().cpu()

            self.cache.current_seq_len -= len(inf_ids)
            self.total_draft_tokens += len(inf_ids) - 1

        # Sample the first future logits

        logits = self.future_logits[:, :1, :]
        token, ptokens, pprobs, prob, eos = ExLlamaV2Sampler.sample(logits, gen_settings, self.sequence_ids[:1, :], random.random(), self.tokenizer, prefix_token, self.return_top_tokens)
        self.future_logits = self.future_logits[:, 1:, :]
        self.future_tokens = self.future_tokens[:, 1:]
        self.cache.current_seq_len += 1

        # Update predictor

        tail_ids = self.sequence_ids[0, -(self.speculative_ngram_max - 1):].tolist() + [token.item()]
        self.ngram.update_single(tail_ids)

        # If sampled token doesn't match future token or no more future tokens

        if self.future_tokens.shape[-1] == 0 or self.future_tokens[0, 0] != token[0, 0]:
            self.future_tokens = None
            self.future_logits = None
        else:
            self.accepted_draft_tokens += 1
        self.total_tokens += 1

        return token, ptokens, pprobs, prob, eos, logits


    def reset_sd_stats(self):

        self.total_tokens = 0
        self.total_draft_tokens = 0
        self.accepted_draft_tokens = 0


    def get_sd_stats(self):

        efficiency = self.accepted_draft_tokens / self.total_tokens if self.total_tokens else 0
        accuracy = self.accepted_draft_tokens / self.total_draft_tokens if self.total_draft_tokens else 0
        return efficiency, accuracy, self.total_tokens, self.total_draft_tokens, self.accepted_draft_tokens


    def ngram_preload(self, input_ids):

        if isinstance(input_ids, torch.Tensor):
            input_ids = input_ids[0].tolist()

        self.ngram_preloaded = NgramCache(self.speculative_ngram_min, self.speculative_ngram_max, None)
        self.ngram_preloaded.update(input_ids)


