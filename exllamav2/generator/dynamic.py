from __future__ import annotations

from collections import deque
import hashlib
from dataclasses import dataclass
from exllamav2 import ExLlamaV2, ExLlamaV2Tokenizer, SeqTensor
from exllamav2.generator import ExLlamaV2Sampler
from exllamav2.generator.filters import ExLlamaV2Filter
from exllamav2.cache import ExLlamaV2Cache
from exllamav2.attn import ExLlamaV2Attention, assert_paged_attn
from exllamav2.ext import exllamav2_ext as ext_c, none_tensor
import torch
import random
import numpy as np
import time
from line_profiler import profile
# from line_profiler import profile

# TODO:
#  - Reuse partially evaluated pages
#  - Interface for CFG + test CFG
#  - Banned strings
#  - LoRA support
#  - "generate_simple" interface
#  - Unpaged mode to support matmul attn
#  - PGO, C++ functions where needed
#  - ExLlamaV2StreamingGenerator wrapper
#  - Q4 cache

PAGE_SIZE = 256
PARTIAL_PAGE_SIZE = 16

def _tensor_blake2b_checksum(tensor: torch.Tensor, prev_hash: bytes | None) -> bytes:
    hasher = hashlib.blake2b(digest_size = 16)
    if prev_hash is not None:
        hasher.update(prev_hash)
    hasher.update(tensor.numpy().tobytes())
    return hasher.digest()

def _randomhash():
    return np.random.bytes(16)

@dataclass
class CachePage:
    # Page index in the actual cache
    page_index: int
    # Hash of this page if prefill_complete == True, else random hash. Also used to index (un)referenced_pages
    phash: bytes
    # Number of active jobs referencing page
    ref_count: int = 0
    # Last time this page was assigned to a job
    access_serial: int = 0
    prefill_complete: bool = False

    def __repr__(self):
        return f"CachePage: idx = {self.page_index}, ref_count = {self.ref_count}, phash: ..{str(self.phash)[8:24]}.."


class NGramTrie:

    token: int
    count: int
    children: dict[int, NGramTrie]
    winningest_child: NGramTrie | None

    def __init__(self, token: int = -1):
        self.token = token
        self.count = 0
        self.children = {}
        self.winningest_child = None


class ExLlamaV2DynamicGenerator:

    model: ExLlamaV2
    cache: ExLlamaV2Cache
    draft_model: ExLlamaV2
    draft_cache: ExLlamaV2Cache
    tokenizer: ExLlamaV2Tokenizer

    max_batch_size: int
    max_total_tokens: int
    max_seq_len: int
    max_chunk_size: int
    padded_vocab_size: int

    num_draft_tokens: int
    max_ngram: int
    use_ngram_draft: bool

    # Page table

    max_pages: int
    referenced_pages: dict[bytes: CachePage]
    unreferenced_pages: dict[bytes: CachePage]
    all_pages: list[CachePage]
    partial_pages: dict[bytes: CachePage]
    access_serial: int
    job_serial: int

    # Job queue

    pending_jobs: list[ExLlamaV2DynamicJob]
    active_jobs: list[ExLlamaV2DynamicJob]

    # Pinned buffer for receiving logits

    logits_pinned: torch.Tensor
    draft_input_ids_pinned: torch.Tensor
    draft_ids_pinned: torch.Tensor


    # @profile
    def __init__(
        self,
        model: ExLlamaV2,
        cache: ExLlamaV2Cache,
        tokenizer: ExLlamaV2Tokenizer,
        max_batch_size: int = 16,
        max_seq_len: int | None = None,
        max_chunk_size: int | None = None,
        max_q_size: int = 16,
        draft_model: ExLlamaV2 | None = None,
        draft_cache: ExLlamaV2Cache | None = None,
        num_draft_tokens: int = 2,
        use_ngram_draft: bool = False,
        max_ngram: int = 4,
        **kwargs
    ):
        """
        Initialize generator

        :param model:
            The model (loaded)

        :param cache:
            ExLlamaV2Cache allocated with batch size 1. The max_seq_len of the cache defines the total
            number of tokens that the generator can assign to a batch of jobs.

        :param tokenizer:
            ExLlamaV2Tokenizer

        :param max_batch_size:
            The maximum number of sequences to process in parallel. The generator will also limit this
            dynamically considering the available cache space.

        :param max_seq_len:
            Maximum length of each individual sequence. Defaults to the model's max_seq_len.

        :param max_chunk_size:
            Maximum number of tokens to process in parallel during prefill (prompt ingestion). Should not
            exceed the model's max_input_len but can be lowered to trade off prompt speed for a shorter
            interruption to ongoing jobs when a new job is started.

        :param max_q_size:
            Maximum number of tokens to evaluate per sequence during generation. Leave this at the default
            (16) unless there's a good reason to increase it.

        :param draft_model:
            Draft model. Enables speculative decoding with draft, and must be specified along with
            draft_cache. Note that speculative decoding with many parallel jobs is likely not advantageous.

        :param draft_cache:
            ExLlamaV2Cache allocated for draft model. Must have batch_size 1 and same max_seq_len as the
            main cmodel cache.

        :param num_draft_tokens:
            Number of future tokens to draft.

        :param use_ngram_draft:
            Use n-gram speculative decoding. Uses a simple n-gram model created from the input sequence
            to predict future tokens.

        :param max_ngram:
            Longest n-gram to consider.

        :param kwargs:
        """

        self.model = model
        self.cache = cache
        self.tokenizer = tokenizer
        cfg = self.model.config
        self.padded_vocab_size = ((cfg.vocab_size + 31) // 32) * 32

        self.draft_model = draft_model
        self.draft_cache = draft_cache
        self.num_draft_tokens = num_draft_tokens if (draft_model or use_ngram_draft) else 0

        if draft_model:
            assert draft_cache is not None, \
                "Must supply cache for draft model"
            assert draft_cache.max_seq_len == cache.max_seq_len and \
                draft_cache.batch_size == cache.batch_size, \
                "Cache and draft cache must be same dimensions"
            assert draft_model.config.max_seq_len >= model.config.max_seq_len, \
                "Draft model seq len must be >= model seq len"

        assert_paged_attn()

        model_max_q = cfg.max_batch_size * cfg.max_input_len
        req_max_q = max_q_size * max_batch_size
        assert req_max_q <= model_max_q, \
            f"Model has max_batch_size * max_input_len = {cfg.max_batch_size} * {cfg.max_input_len} tokens, " + \
            f"generator requires max_batch_size * max_q_size = {max_batch_size} * {max_q_size} tokens."
        self.max_batch_size = max_batch_size

        if max_seq_len is not None:
            assert max_seq_len <= model.config.max_seq_len, \
                f"Model initialized with max seq len {model.config.max_seq_len}, " + \
                f"requested seq len is {max_seq_len}"
            self.max_seq_len = max_seq_len
        else:
            self.max_seq_len = model.config.max_seq_len

        # Initialize cache/page table

        assert cache.batch_size == 1, \
            f"DynamicGenerator requires cache to have batch_size = 1"

        assert self.cache.max_seq_len % PAGE_SIZE == 0, \
            f"cache.max_seq_len must be multiple of {PAGE_SIZE}, received {cache.max_seq_len}"
        self.max_pages = cache.max_seq_len // PAGE_SIZE
        self.max_total_tokens = cache.max_seq_len

        self.referenced_pages = {}
        self.unreferenced_pages = {}
        self.all_pages = []
        for idx in range(self.max_pages):
            h = _randomhash()
            cp = CachePage(page_index = idx, phash = h)
            self.all_pages.append(cp)
            self.unreferenced_pages[h] = cp
        self.partial_pages = {}

        # Chunking

        if max_chunk_size is not None:
            assert max_chunk_size <= model.config.max_input_len, \
                f"max_chunk_size must be less than model max_input_len ({model.config.max_input_len}), " + \
                f"received {max_chunk_size}"
            self.max_chunk_size = max_chunk_size
        else:
            self.max_chunk_size = model.config.max_input_len
        assert self.max_chunk_size % PAGE_SIZE == 0, \
            f"max_chunk_size must be multiple of {PAGE_SIZE}, received {max_chunk_size}"

        self.access_serial = 0

        # Jobs

        self.job_serial = 0
        self.pending_jobs = []
        self.active_jobs = []

        # Buffers

        self.logits_pinned = torch.empty(
            (max_batch_size, max_q_size, self.padded_vocab_size),
            dtype = torch.float,
            pin_memory = True
        )

        if draft_model:
            self.draft_input_ids_pinned = torch.empty(
                (max_batch_size, 1),
                dtype = torch.long,
                pin_memory = True
            )
            self.draft_ids_pinned = torch.empty(
                (max_batch_size, num_draft_tokens),
                dtype = torch.long,
                pin_memory = True
            )

        # Ngram

        if use_ngram_draft:
            assert draft_model is None, \
                "Cannot use both draft model and ngram draft"
        self.max_ngram = max_ngram
        self.use_ngram_draft = use_ngram_draft


    def print_page_list(self, short: bool = True):
        for cp in self.all_pages:
            if cp.phash in self.referenced_pages:
                assert cp.ref_count > 0
                ref = str(cp.ref_count) if cp.ref_count < 10 else "+"
            elif cp.phash in self.unreferenced_pages:
                assert cp.ref_count == 0
                ref = "."
            else:
                ref = "#"
            if short: print(ref, end = "")
            else: print(str(cp) + f", ref {ref}")
        print()


    def validate_cache(self):
        try:
            assert len(self.referenced_pages) + len(self.unreferenced_pages) == self.max_pages, "sum"
            ref_counts = [0] * self.max_pages
            for job in self.active_jobs:
                for seq in job.sequences:
                    for page in seq.allocated_pages:
                        ref_counts[page.page_index] += 1
            for (h, page) in self.referenced_pages.items():
                assert page.phash == h, "r hash " + str(page)
                assert page.ref_count == ref_counts[page.page_index], "r refc " + str(page)
                assert h not in self.unreferenced_pages, "r2u " + str(page)
            for (h, page) in self.unreferenced_pages.items():
                assert page.phash == h, "u hash " + str(page)
                assert page.ref_count == ref_counts[page.page_index], "u refc " + str(page)
                assert h not in self.referenced_pages, "u2r " + str(page)
        except Exception as ex:
            print(ex)
            raise ex


    # @profile
    def update_partial_pages(self, page: CachePage, lhash: bytes, ):
        pass


    # @profile
    def num_remaining_jobs(self):
        return len(self.pending_jobs) + len(self.active_jobs)


    # @profile
    def enqueue(
        self,
        job: ExLlamaV2DynamicJob | list[ExLlamaV2DynamicJob]
    ):
        """
        Adds a job or list of jobs to the queue.
        """

        if isinstance(job, list):
            for j in job: self.enqueue(j)
            return

        job.prepare_for_queue(self, self.job_serial)
        self.job_serial += 1
        self.pending_jobs.append(job)
        job.time_enqueue = time.time()


    # @profile
    def iterate(self) -> list[dict]:
        """
        Performs inference on available jobs.

        :return:
            List of dicts:

            # Job has started
            {
                "job": ExLlamaV2DynamicJob  - reference to job
                "stage": "started"
                "identifier":  - optional identifier
            }

            # Prefill is underway
            {
                "job": ExLlamaV2DynamicJob  - reference to job
                "stage": "prefill"
                "curr_progress": int  - prompt tokens ingested so far
                "max_progress": int  - total prompt tokens to ingest
                "identifier":  - optional identifier
            }

            # Generation is underway
            {
                "job": ExLlamaV2DynamicJob  - reference to job
                "stage": "streaming"
                "identifier":  - optional identifier
                "eos": bool  - True if stop condition has been met

                optional, if eos:
                    "eos_reason":  - one of:
                        "stop_token"
                        "stop_string"
                        "max_new_tokens"
                        "end_filter"
                    "full_completion": str  - full text completion
                    "new_tokens": int  - number of tokens generated
                    "time_enqueued": float  - time from job was enqueued until it started, in seconds
                    "time_prefill": float  - time to first token, in seconds
                    "time_generate": float  - time to last token, in seconds
                    optional, if SD enabled:
                        "accepted_draft_tokens": int
                        "rejected_draft_tokens": int

                "text": str  - streamed text output. Does not include prefix from healed token, or stop string
                "token_ids": torch.Tensor  - output tokens, shape (1, n)
                "token_probs": torch.Tensor  - last sampling probability of output tokens, shape (1, n)
                "top_k_tokens": torch.Tensor  - shape (1, n, k)
                "top_k_probs": torch.Tensor  - shape (1, n, k)
                "logits": torch.Tensor  - shape (1, n, vocab_size)
            }
        """

        results = []
        self.iterate_start_jobs(results)

        # Perform one round of prefill

        for job in self.active_jobs:
            job.prefill(results)

        # Generation with draft model

        if self.draft_model:
            draft_tokens = self.iterate_draftmodel_gen(results)
            self.iterate_gen(results, draft_tokens)

        # Generation with ngram draft

        elif self.use_ngram_draft:
            draft_tokens = self.iterate_ngram_gen(results)
            self.iterate_gen(results, draft_tokens)

        # Regular generation

        else:
            self.iterate_gen(results)

        # Finished iteration

        return results


    # @profile
    def iterate_ngram_gen(self, results: list):

        draft_ids_list = []

        for job in self.active_jobs:
            if not job.is_prefill_done(): continue

            # Update trie

            sequence = job.sequences[0].sequence_ids
            pos_a = job.ngram_position
            pos_b = len(sequence) - self.max_ngram
            if pos_b > pos_a:
                subs = sequence.torch()[0, pos_a : pos_b + self.max_ngram].tolist()
                job.ngram_position = pos_b

                for i in range(len(subs) - self.max_ngram):
                    node = job.ngrams
                    for j in range(i, i + self.max_ngram):
                        token = subs[j]
                        if token not in node.children:
                            node.children[token] = NGramTrie(token)
                        child = node.children[token]
                        child.count += 1
                        if not node.winningest_child or child.count > node.winningest_child.count:
                            node.winningest_child = child
                        node = child

            # Predict

            subs = sequence.torch()[0, -(self.max_ngram - 1):].tolist()
            ids = []

            for i in range(self.num_draft_tokens):
                w = subs[-1]
                for j in range(self.max_ngram - 1):
                    node = job.ngrams
                    for k in range(j, self.max_ngram - 1):
                        token = subs[k]
                        node = node.children.get(token)
                        if not node: break
                    if node:
                        w = node.winningest_child.token
                        break
                ids.append(w)
                subs = subs[1:] + [w]

            draft_ids_list.append(torch.tensor([ids], dtype = torch.long))

        return torch.cat(draft_ids_list, dim = 0) if len(draft_ids_list) > 0 else None


    # @profile
    def iterate_draftmodel_gen(self, results: list):

        batch_size = 0
        max_seq_len = 0
        for job in self.active_jobs:
            if not job.is_prefill_done(): continue
            max_seq_len = max(max_seq_len, job.get_max_seq_len() + self.num_draft_tokens)
            batch_size += 1

        if batch_size == 0:
            return None

        max_pages_batch = (max_seq_len + PAGE_SIZE - 1) // PAGE_SIZE
        block_index = torch.zeros((batch_size, max_pages_batch), dtype = torch.int32)
        cache_seqlens = torch.zeros((batch_size,), dtype = torch.int32)

        batch = 0
        for job in self.active_jobs:
            if not job.is_prefill_done(): continue
            for seq in job.sequences:
                seq_block_index = job.get_block_index(seq, len(seq.sequence_ids))
                block_index[batch, :seq_block_index.shape[-1]].copy_(seq_block_index)
                cache_seqlens[batch] = seq.kv_position
                batch += 1

        # Collect input IDs

        input_ids_list = []
        for job in self.active_jobs:
            if not job.is_prefill_done(): continue
            if job.time_first_token is None:
                torch.cuda.synchronize()
                job.time_first_token = time.time()
            job_ids = job.get_input_ids_list()
            input_ids_list += job_ids

        batch_ids = self.draft_input_ids_pinned[:batch_size, :]
        batch_ids.copy_(torch.cat(input_ids_list, dim = 0), non_blocking = True)

        # Greedy sample draft IDs

        for idx in range(self.num_draft_tokens):

            attn_params = ExLlamaV2Attention.PagedParams(
                batch_size = batch_size,
                block_index = block_index,
                cache_seqlens = cache_seqlens
            )

            device_logits, _ = self.draft_model.forward_chunk(
                input_ids = batch_ids,
                attn_params = attn_params,
                cache = self.draft_cache
            )

            new_ids = torch.argmax(device_logits, dim = -1)
            self.draft_ids_pinned[:batch_size, idx:idx+1].copy_(new_ids, non_blocking = True)
            batch_ids.copy_(new_ids, non_blocking = True)
            cache_seqlens += 1

        torch.cuda.synchronize()
        return self.draft_ids_pinned


    # @profile
    def iterate_gen(self, results: list, draft_tokens: torch.Tensor | None = None):

        batch_size = 0
        max_seq_len = 0
        for job in self.active_jobs:
            if not job.is_prefill_done(): continue
            max_seq_len = max(max_seq_len, job.get_max_seq_len() + self.num_draft_tokens)
            batch_size += len(job.sequences)

        if batch_size == 0:
            return # Nothing more to do this iteration

        max_pages_batch = (max_seq_len + PAGE_SIZE - 1) // PAGE_SIZE
        block_index = torch.zeros((batch_size, max_pages_batch), dtype = torch.int32)
        cache_seqlens = torch.zeros((batch_size,), dtype = torch.int32)

        batch = 0
        for job in self.active_jobs:
            if not job.is_prefill_done(): continue
            for seq in job.sequences:
                seq_block_index = job.get_block_index(seq, len(seq.sequence_ids))
                block_index[batch, :seq_block_index.shape[-1]].copy_(seq_block_index)
                cache_seqlens[batch] = seq.kv_position
                batch += 1

        # Collect input IDs

        input_ids_list = []
        logit_mapping = []
        for job in self.active_jobs:
            logit_mapping.append(len(input_ids_list))
            if not job.is_prefill_done(): continue
            if job.time_first_token is None:
                torch.cuda.synchronize()
                job.time_first_token = time.time()
            if draft_tokens is None:
                job_ids = job.get_input_ids_list()
            else:
                job_ids = job.get_input_ids_list(draft_tokens, len(input_ids_list))
            input_ids_list += job_ids
        logit_mapping.append(len(input_ids_list))

        batch_ids = torch.cat(input_ids_list, dim = 0)

        # Get logit batch from model

        attn_params = ExLlamaV2Attention.PagedParams(
            batch_size = batch_size,
            block_index = block_index,
            cache_seqlens = cache_seqlens
        )

        device_logits, _ = self.model.forward_chunk(
            input_ids = batch_ids,
            attn_params = attn_params,
            cache = self.cache
        )

        # Pass logits to jobs for sampling

        batch_logits = self.logits_pinned[:device_logits.shape[0], :device_logits.shape[1], :]
        batch_logits.copy_(device_logits, non_blocking = True)
        torch.cuda.synchronize()

        completed_jobs = []
        j = 0
        for job, a, b in zip(self.active_jobs, logit_mapping[:-1], logit_mapping[1:]):
            if a == b: continue
            for i in range(batch_logits.shape[1]):
                job_logits = batch_logits[a:b, i:i+1, :]
                eos, sampled_token = job.receive_logits(job_logits, results)
                if eos:
                    completed_jobs.append(job)
                    break
                if draft_tokens is not None and i < batch_logits.shape[1] - 1:
                    if draft_tokens[j, i].item() != sampled_token.item():
                        job.rejected_draft_tokens += batch_logits.shape[1] - 1 - i
                        break
                    else:
                        job.accepted_draft_tokens += 1
            j += 1

        # Release pages for completed jobs

        for job in completed_jobs:
            job.deallocate_pages()
            self.active_jobs.remove(job)


    # @profile
    def iterate_start_jobs(self, results: list):

        # Get current max batch

        current_max_batch = 0
        for job in self.active_jobs:
            current_max_batch += len(job.sequences)

        # Start new jobs if possible

        if (len(self.unreferenced_pages) and
            len(self.pending_jobs) and
            current_max_batch < self.max_batch_size):

            skipped_jobs = []
            for job in self.pending_jobs.copy():

                if (len(job.sequences) + current_max_batch > self.max_batch_size or
                        job.current_new_pages_required() > len(self.unreferenced_pages)):
                    skipped_jobs.append(job)
                    continue

                # Make sure the job we're about to add doesn't skip a job that's been skipped too many times

                for j in skipped_jobs:
                    if j.skips >= j.max_skips:
                        return
                for j in skipped_jobs:
                    j.skips += 1

                # Add job to active list

                self.pending_jobs.remove(job)
                self.active_jobs.append(job)

                # Allocate pages for job

                job.allocate_pages()
                current_max_batch += len(job.sequences)

                r = {
                    "job": job,
                    "stage": "started",
                }
                if job.identifier is not None:
                    r.update({ "identifier": job.identifier })
                results.append(r)


# Convert list of strings to UTF32 format to pass by reference to partial matching function

# @profile
def _strings_to_utf32(strings: list[str]) -> (np.array, list[int]):

    if not strings: return bytearray(), None

    encoded_strings = [s.encode("utf-32-le") for s in strings]
    encoded_lengths = [len(s) for s in encoded_strings]
    offsets = [0] + encoded_lengths
    for i in range(1, len(offsets)):
        offsets[i] += offsets[i - 1]
    total_length = offsets[-1]
    concat_strings = bytearray(total_length)
    for s, offset in zip(encoded_strings, offsets[:-1]):
        concat_strings[offset:offset + len(s)] = s

    concat_strings = np.frombuffer(concat_strings, dtype = np.uint8)
    offsets = np.frombuffer(np.array(offsets, dtype = np.int32), dtype = np.uint8)
    return concat_strings, offsets


class ExLlamaV2DynamicJob:

    generator: ExLlamaV2DynamicGenerator | None
    serial_number: int | None

    class Sequence:
        input_ids: SeqTensor
        sequence_ids: SeqTensor
        kv_position: int
        page_hashes: list[bytes]
        new_unique_pages: int
        allocated_pages: list[CachePage] | None
        block_index_tensor: torch.Tensor | None
        live: bool

    sequences: list[Sequence]
    all_unique_hashes: list[bytes]

    max_new_tokens: int
    new_tokens: int
    gen_settings: ExLlamaV2Sampler.Settings
    rng: random.Random

    decode_special_tokens: bool
    return_top_tokens: int
    prefix_token: torch.Tensor | None

    skips: int
    max_skips: int | None

    # Measurement

    time_enqueue: float | None
    time_first_prefill: float | None
    time_first_token: float | None
    time_last_token: float | None
    accepted_draft_tokens: int
    rejected_draft_tokens: int

    # Output buffers

    held_text: str
    held_tokens: SeqTensor
    held_k_tokens: SeqTensor
    held_k_probs: SeqTensor
    held_probs: SeqTensor
    held_logits: SeqTensor

    full_completion: str

    # Ngrams

    ngrams: NGramTrie
    ngram_position: int

    # Filters

    filters: list[ExLlamaV2Filter] | None
    filter_prefer_eos: bool


    # @profile
    def __init__(
        self,
        input_ids: torch.Tensor | list[torch.Tensor],
        max_new_tokens: int,
        max_skips: int | None = 4,
        gen_settings: ExLlamaV2Sampler.Settings = ExLlamaV2Sampler.Settings(),
        seed: int = None,
        stop_conditions: list | tuple | set = None,
        decode_special_tokens: bool = False,
        return_top_tokens: int = 0,
        return_logits: bool = False,
        return_probs: bool = False,
        filters: list[ExLlamaV2Filter] | None = None,
        filter_prefer_eos: bool = False,
        token_healing: bool = False,
        identifier: object | None = None,
        **kwargs
    ):
        """
        Create new job.

        :param input_ids:
            Tokenized IDs of the input prompt, shape (1, n)

        :param max_new_tokens:
            Max no. output tokens to allow

        :param max_skips:
            In the event that the job is too large to fit in the cache at any given moment but there are
            smaller jobs pending that would fit, those smaller jobs are started instead. This number
            specifies the maximum number of times a job can be skipped in favor of a smaller job before it
            stalls the queue. After this, the job is guaranteed to be the next job started.

        :param gen_settings:
            ExLlamaV2Sampler.Settings containing sampling parameters

        :param seed:
             RNG seed (determinism is not guaranteed)

        :param stop_conditions:
            List of strings and/or token IDs that will trigger the EOS condition. If a stop condition is
            encountered it is not emitted as output. If the beginning of a stop string is sampled, stream
            output will be held until the stop condition can be resolved.

        :param decode_special_tokens:
            If True, special tokens like <|im_start|> etc. will be decoded and included in the text output.
            If False, special tokens will still be respected as stop conditions.

        :param return_top_tokens:
            Number of top tokens to return, along with their final sampling probabilities. There is some
            performance penalty for enabling this.

        :param return_logits:
            Return pre-sampling logits along with output tokens.

        :param return_probs:
            Return final sampling probability for each chosen token.

        :param filters:
            List of ExLlamaV2Filters to apply during generation.

        :param filter_prefer_eos:
            If True, the sampler will prefer whatever token the filter presents as an EOS condition, e.g.
            the outer closing bracket in a JSON grammar, as soon as that (sub)token is legal under the
            grammar.

        :param token_healing:
            Resample the last token of the input with a prefix constraint. E.g. if the last token is
            "_Hel", it is removed from the input and the first token of the output will be constrained to
            one of "_Hello", "_Help", "_Helium", etc. Only the added part of the healed token is emitted as
            text, i.e. "lo", "p", "ium" etc.

        :param identifier:
            Object to return with every stream event relating to this job

        :param kwargs:
        """

        assert all(ids.device.type == "cpu" for ids in input_ids), \
                "input_ids must reside in system memory"

        self.generator = None
        self.serial_number = None
        self.identifier = identifier

        self.max_skips = max_skips
        self.allocated_pages = None

        # Prepare sequences

        if not isinstance(input_ids, list):
            input_ids = [input_ids]

        if token_healing and all(ids.shape[-1] > 1 for ids in input_ids):
            input_seq_ids = [ids[:, :-1] for ids in input_ids]
            self.prefix_token = torch.cat([ids[:, -1:] for ids in input_ids], dim = 0)
        else:
            input_seq_ids = input_ids
            self.prefix_token = None

        self.sequences = []
        for ids, seq_ids in zip(input_ids, input_seq_ids):
            assert ids.shape[-1] > 0, \
                "Input IDs cannot be empty."
            assert ids.shape[0] == 1, \
                "input_ids must be [1, seq_len] tensor or list of [1, seq_len] tensors"
            seq = ExLlamaV2DynamicJob.Sequence()
            seq.input_ids = SeqTensor.from_tensor(ids, seq_dim = -1)
            seq.sequence_ids = SeqTensor.from_tensor(seq_ids, seq_dim = -1)
            seq.kv_position = 0
            seq.page_hashes = None
            seq.new_unique_pages = 0
            seq.allocated_pages = None
            seq.block_index_tensor = None
            seq.live = True
            self.sequences.append(seq)

        # Generation parameters

        self.max_new_tokens = max_new_tokens
        self.new_tokens = 0 if self.prefix_token is None else -1
        self.gen_settings = gen_settings
        self.rng = random.Random() if seed is None else random.Random(seed)

        # Output options

        self.decode_special_tokens = decode_special_tokens
        self.return_top_tokens = return_top_tokens
        self.return_logits = return_logits
        self.return_probs = return_probs

        # Stop conditions

        self.stop_strings = set()
        self.stop_tokens = set()
        if stop_conditions is not None:
            for t in stop_conditions:
                if isinstance(t, int): self.stop_tokens.add(t)
                elif isinstance(t, str): self.stop_strings.add(t)
                else: raise ValueError("Unsupported type in stop_conditions")
            self.stop_strings_utf32_buffer, self.stop_strings_utf32_offsets = \
                _strings_to_utf32(list(self.stop_strings))
        else:
            self.stop_strings_utf32_offsets, self.stop_strings_utf32_buffer = None, None

        # Measurement

        self.time_enqueue = None
        self.time_first_prefill = None
        self.time_first_token = None
        self.time_last_token = None
        self.accepted_draft_tokens = 0
        self.rejected_draft_tokens = 0

        # Ngram

        self.ngrams = NGramTrie()
        self.ngram_position = 0

        # Filters

        self.filters = filters if filters is not None else []
        self.filter_prefer_eos = filter_prefer_eos


    def __repr__(self):
        if self.serial_number is None:
            return "ExLlamaV2DynamicJob (new)"
        else:
            return f"ExLlamaV2DynamicJob #{self.serial_number}"


    # @profile
    def is_prefill_done(self):
        return all(seq.kv_position == len(seq.sequence_ids) - 1 for seq in self.sequences)


    # @profile
    def get_max_seq_len(self):
        if not self.is_prefill_done():
            return 0
        max_seq_len = 0
        for seq in self.sequences:
            if seq.kv_position == len(seq.sequence_ids) - 1:
                max_seq_len = max(max_seq_len, len(seq.sequence_ids))
        return max_seq_len


    # @profile
    def get_input_ids_list(self, draft_tokens: torch.Tensor | None = None, idx: int = 0):
        input_ids_list = []
        for seq in self.sequences:
            ids = seq.sequence_ids.torch_slice(seq.kv_position, None)
            if draft_tokens is not None:
                ids = torch.cat((ids, draft_tokens[idx:idx+1, :]), dim = -1)
            input_ids_list.append(ids)
        return input_ids_list


    # @profile
    def receive_logits(
        self,
        logits: torch.Tensor,
        results: list
    ):
        # Support single seq and CFG for now

        assert logits.shape[0] == len(self.sequences) == (2 if self.gen_settings.cfg_scale is not None else 1)
        assert self.is_prefill_done()
        assert all(seq.live for seq in self.sequences)

        # Start filters

        if self.new_tokens == 0:
            for f in self.filters: f.begin("")

        # Sample

        next_token, next_k_tokens, next_k_probs, next_prob, filter_eos = \
        ExLlamaV2Sampler.sample(
            logits,
            self.gen_settings,
            self.sequences[0].sequence_ids.torch(),
            self.rng.random(),
            self.generator.tokenizer,
            self.prefix_token if self.new_tokens == -1 else None,
            self.return_top_tokens,
            blocked_tokens = None,  # self.current_blocked_tokens
            filters = self.filters if self.new_tokens >= 0 else None,
            filter_prefer_eos = self.filter_prefer_eos
        )

        # Feed filters

        if self.new_tokens >= 0:
            for f in self.filters: f.feed(next_token)

        # Accept token

        self.new_tokens += 1

        for seq in self.sequences:

            seq.sequence_ids.append(next_token)

            # Accept cached K/V

            page_before = seq.kv_position // PAGE_SIZE
            seq.kv_position += 1
            page_after = seq.kv_position // PAGE_SIZE

            # Hash completed page

            if page_before != page_after:
                page = seq.allocated_pages[page_before]
                if page_before > 0:
                    last_page = seq.allocated_pages[page_before - 1]
                    last_hash = last_page.phash
                else:
                    last_hash = None
                page_ids = seq.sequence_ids.torch_slice(page_before * PAGE_SIZE, page_after * PAGE_SIZE)
                old_hash = page.phash
                new_hash = _tensor_blake2b_checksum(page_ids, last_hash)

                del self.generator.referenced_pages[old_hash]

                if new_hash in self.generator.referenced_pages:
                    assert page.ref_count == 1
                    page.ref_count = 0
                    self.generator.unreferenced_pages[old_hash] = page
                    page = self.generator.referenced_pages[new_hash]
                    seq.allocated_pages[page_before] = page
                    assert page.prefill_complete
                    page.ref_count += 1
                else:
                    if new_hash in self.generator.unreferenced_pages:
                        assert page.ref_count == 1
                        page.ref_count = 0
                        self.generator.unreferenced_pages[old_hash] = page
                        page = self.generator.unreferenced_pages[new_hash]
                        assert page.ref_count == 0
                        assert page.prefill_complete
                        seq.allocated_pages[page_before] = page
                        del self.generator.unreferenced_pages[new_hash]
                        page.ref_count += 1
                        self.generator.referenced_pages[new_hash] = page
                    else:
                        page.phash = new_hash
                        page.prefill_complete = True
                        self.generator.referenced_pages[new_hash] = page

        # Stream output

        def emit(res: list, emit_eos: bool = False, eos_reason: str = None, emit_held = False):
            r = {
                "job": self,
                "stage": "streaming",
                "eos": emit_eos,
            }

            if eos_reason is not None:
                r.update({ "eos_reason": eos_reason })

            if emit_held:
                if self.held_text != "":
                    self.full_completion += self.held_text
                    r.update({ "text": self.held_text })
                    self.held_text = ""
                if self.held_tokens:
                    r.update({ "token_ids": self.held_tokens.torch() })
                    self.held_tokens.clear()
                if self.held_probs:
                    r.update({ "token_probs": self.held_probs.torch() })
                    self.held_probs.clear()
                if self.held_k_tokens:
                    r.update({ "top_k_tokens": self.held_k_tokens.torch() })
                    r.update({ "top_k_probs": self.held_k_probs.torch() })
                    self.held_k_tokens.clear()
                    self.held_k_probs.clear()
                if self.held_logits:
                    r.update({ "logits": self.held_logits.torch() })
                    self.held_logits.clear()

            if emit_eos:
                self.time_last_token = time.time()
                r.update({
                    "full_completion": self.full_completion,
                    "new_tokens": self.new_tokens,
                    "time_enqueued": self.time_first_prefill - self.time_enqueue,
                    "time_prefill": self.time_first_token - self.time_first_prefill,
                    "time_generate": self.time_last_token - self.time_first_token,
                })
                if self.generator.draft_model or self.generator.use_ngram_draft:
                    r.update({
                        "accepted_draft_tokens": self.accepted_draft_tokens,
                        "rejected_draft_tokens": self.rejected_draft_tokens
                    })

            if self.identifier is not None:
                r.update({ "identifier": self.identifier })

            results.append(r)
            return emit_eos, next_token

        # End on stop tokens

        if next_token.item() in self.stop_tokens:
            return emit(results, emit_eos = True, eos_reason = "stop_token")

        # Decode and buffer output

        id_to_piece = self.generator.tokenizer.get_id_to_piece_list(self.decode_special_tokens)
        new_text = id_to_piece[next_token.item()]

        if self.new_tokens == 0:
            unhealed = id_to_piece[self.prefix_token[0].item()]
            new_text = new_text[len(unhealed):]

        self.held_text += new_text
        self.held_tokens.append(next_token)
        if self.return_probs:
            self.held_probs.append(next_prob)
        if self.return_top_tokens > 0:
            self.held_k_tokens.append(next_k_tokens)
            self.held_k_probs.append(next_k_probs)
        if self.return_logits:
            self.held_logits.append(logits)

        # Stop if we reach max_new_tokens

        if self.new_tokens >= self.max_new_tokens - self.generator.num_draft_tokens:
            return emit(results, emit_eos = True, emit_held = True, eos_reason = "max_new_tokens")

        # End now if newly added token ends a filter

        if filter_eos:
            return emit(results, emit_eos = True, emit_held = True, eos_reason = "end_filter")

        # Hold text if it contains an incomplete character

        if self.held_text.endswith("�") and not self.held_text.endswith("�����"):
            test_decode = self.generator.tokenizer.decode(
                self.held_tokens.torch(),
                decode_special_tokens = self.decode_special_tokens
            )[0]
            if not test_decode.endswith("�"):
                self.held_text = test_decode
            else:
                return emit(results)

        # End on stop strings

        if self.stop_strings_utf32_offsets is not None:
            match = ext_c.partial_strings_match(
                np.frombuffer(self.held_text.encode("utf-32-le"), dtype = np.uint8),
                self.stop_strings_utf32_offsets,
                self.stop_strings_utf32_buffer
            )
            if match >= 0:
                self.held_text = self.held_text[:match]
                return emit(results, emit_eos = True, emit_held = True, eos_reason = "stop_string")
            if match == -2:
                return emit(results)

        # Stream output

        return emit(results, emit_held = True)


    # @profile
    def prepare_for_queue(self, generator, serial_number: int):

        self.serial_number = serial_number
        self.generator = generator
        self.skips = 0

        # Hash full pages of input IDs

        all_unique_hashes = set()
        all_unique_pages = 0

        for seq in self.sequences:

            seq.page_hashes = []

            max_len = len(seq.sequence_ids) + self.max_new_tokens
            context_pages = len(seq.sequence_ids) // PAGE_SIZE
            total_pages = (max_len + 255) // PAGE_SIZE

            r_hash = None
            for i in range(context_pages):
                page_ids = seq.sequence_ids.torch_slice(i * PAGE_SIZE, (i + 1) * PAGE_SIZE)
                r_hash = _tensor_blake2b_checksum(page_ids, r_hash)
                seq.page_hashes.append(r_hash)
                all_unique_hashes.add(r_hash)

            seq.new_unique_pages = total_pages - context_pages
            all_unique_pages += seq.new_unique_pages

        self.all_unique_hashes = list(all_unique_hashes)

        # Make sure the request can potentially fit

        total_pages = len(self.all_unique_hashes) + seq.new_unique_pages
        assert total_pages <= self.generator.max_pages, \
            f"Job requires {total_pages} pages, only {self.generator.max_pages} available and cannot " + \
            f"be enqueued. Total cache allocated is is {self.generator.max_pages} * {PAGE_SIZE} = " + \
            f"{self.generator.max_total_tokens} tokens"

        # Initial conditions

        self.held_text = ""
        self.held_tokens = SeqTensor((1, 0), dtype = torch.long, seq_dim = -1)
        self.held_k_tokens = SeqTensor((1, 0, self.return_top_tokens), dtype = torch.long, seq_dim = 1)
        self.held_k_probs = SeqTensor((1, 0, self.return_top_tokens), dtype = torch.float, seq_dim = 1)
        self.held_probs = SeqTensor((1, 0), dtype = torch.float, seq_dim = -1)
        self.held_logits = SeqTensor((0, self.generator.padded_vocab_size), dtype = torch.float, seq_dim = 0)

        self.full_completion = ""


    # @profile
    def current_new_pages_required(self):
        new_pages = 0
        for h in self.all_unique_hashes:
            if h not in self.generator.referenced_pages:
                new_pages += 1
        for s in self.sequences:
            new_pages += s.new_unique_pages
        return new_pages


    # @profile
    def prefill(self, results: list):

        if self.time_first_prefill is None:
            self.time_first_prefill = time.time()

        progress = 0
        for seq in self.sequences:

            prefill_start = seq.kv_position
            prefill_end = seq.kv_position + self.generator.max_chunk_size
            prefill_end = (prefill_end // PAGE_SIZE) * PAGE_SIZE
            prefill_end = min(prefill_end, len(seq.sequence_ids) - 1)

            p0 = prefill_start // PAGE_SIZE
            p1 = (prefill_end + PAGE_SIZE - 1) // PAGE_SIZE
            for local_idx in range(p0, p1):
                page = seq.allocated_pages[local_idx]
                if page.prefill_complete:
                    prefill_start = (local_idx + 1) * PAGE_SIZE
                    seq.kv_position = prefill_start
                else:
                    break

            p0 = prefill_start // PAGE_SIZE
            for local_idx in range(p0, p1):
                page = seq.allocated_pages[local_idx]
                if page.prefill_complete:
                    prefill_end = local_idx * PAGE_SIZE
                    break

            if prefill_end <= prefill_start:
                continue

            prefill_ids = seq.sequence_ids.torch_slice(prefill_start, prefill_end)

            attn_params = ExLlamaV2Attention.PagedParams(
                batch_size = 1,
                block_index = self.get_block_index(seq, prefill_end).unsqueeze(0),
                cache_seqlens = torch.tensor([prefill_start], dtype = torch.int32)
            )

            if self.generator.draft_model:
                self.generator.draft_model.forward_chunk(
                    input_ids = prefill_ids,
                    preprocess_only = True,
                    attn_params = attn_params,
                    cache = self.generator.draft_cache
                )

            self.generator.model.forward_chunk(
                input_ids = prefill_ids,
                preprocess_only = True,
                attn_params = attn_params,
                cache = self.generator.cache
            )

            seq.kv_position = prefill_end
            p0 = prefill_start // PAGE_SIZE
            p1 = prefill_end // PAGE_SIZE
            for local_idx in range(p0, p1):
                seq.allocated_pages[local_idx].prefill_complete = True

            progress += prefill_end - prefill_start

        if progress:
            r = {
                "job": self,
                "stage": "prefill",
                "curr_progress": sum(seq.kv_position for seq in self.sequences),
                "max_progress": sum(len(seq.sequence_ids) - 1 for seq in self.sequences),
            }
            if self.identifier is not None:
                r.update({"identifier": self.identifier})
            results.append(r)


    # @profile
    def get_block_index(self, seq: Sequence, max_len) -> torch.Tensor:

        if seq.block_index_tensor is None:
            block_index = [page.page_index for page in seq.allocated_pages]
            seq.block_index_tensor = torch.tensor(block_index, dtype = torch.int32)

        num_blocks = (max_len + PAGE_SIZE - 1) // PAGE_SIZE
        return seq.block_index_tensor[:num_blocks]


    # @profile
    def allocate_pages(self):

        for seq in self.sequences:

            seq.allocated_pages = []
            available_pages = None

            self.generator.access_serial += 1
            new_serial = self.generator.access_serial

            # Allocate whole pages

            for h in seq.page_hashes:

                # Find matching referenced page

                rp = self.generator.referenced_pages.get(h)
                if rp:
                    assert rp.ref_count > 0
                    rp.ref_count += 1
                    rp.access_serial = new_serial
                    seq.allocated_pages.append(rp)

                # If possible, reuse an unreferenced page with matching hash

                else:
                    up = self.generator.unreferenced_pages.get(h)
                    if up:
                        assert up.ref_count == 0
                        up.ref_count = 1
                        up.access_serial = new_serial
                        del self.generator.unreferenced_pages[h]
                        self.generator.referenced_pages[h] = up
                        seq.allocated_pages.append(up)

                    # No matching pages

                    else:

                        # Get list of unreferenced pages in order of oldest to newest

                        if available_pages is None:
                            available_pages = list(self.generator.unreferenced_pages.values())
                            available_pages.sort(key = lambda x: x.access_serial)
                            available_pages = deque(available_pages)

                        # Allocate oldest unreferenced page

                        np = available_pages.popleft()
                        del self.generator.unreferenced_pages[np.phash]
                        self.generator.referenced_pages[h] = np
                        np.phash = h
                        assert np.ref_count == 0
                        np.ref_count = 1
                        np.prefill_complete = False
                        np.access_serial = new_serial
                        seq.allocated_pages.append(np)

            # Allocate unique pages

            for npi in range(seq.new_unique_pages):

                # Get list of unreferenced pages in order of oldest to newest

                if available_pages is None:
                    available_pages = list(self.generator.unreferenced_pages.values())
                    available_pages.sort(key = lambda x: x.access_serial)
                    available_pages = deque(available_pages)

                np = available_pages.popleft()
                del self.generator.unreferenced_pages[np.phash]
                hr = _randomhash()
                self.generator.referenced_pages[hr] = np
                np.phash = hr
                assert np.ref_count == 0
                np.ref_count = 1
                np.prefill_complete = False
                np.access_serial = new_serial
                seq.allocated_pages.append(np)

            # Advance cache over prefilled pages

            for page in seq.allocated_pages:
                if page.prefill_complete:
                    seq.kv_position += PAGE_SIZE
                else:
                    break


    # @profile
    def deallocate_pages(self):

        for seq in self.sequences:

            for page in seq.allocated_pages:

                assert page.ref_count > 0
                page.ref_count -= 1
                if page.ref_count == 0:
                    del self.generator.referenced_pages[page.phash]
                    self.generator.unreferenced_pages[page.phash] = page
