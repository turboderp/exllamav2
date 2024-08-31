from __future__ import annotations
from threading import Lock
from concurrent.futures import ThreadPoolExecutor, Future
from exllamav2 import ExLlamaV2, ExLlamaV2Tokenizer
from exllamav2.ext import exllamav2_ext as ext_c, none_tensor
import torch

class ExLlamaV2Filter:

    # Internal state

    model: ExLlamaV2
    tokenizer: ExLlamaV2Tokenizer
    sequence_str: str

    background_result: Future | None = None

    # For compatibility
    allow_return_type_list: bool = True

    def __init__(self,
                 model: ExLlamaV2,
                 tokenizer: ExLlamaV2Tokenizer):

        self.model = model
        self.tokenizer = tokenizer
        self.sequence_str = ""


    def clone(self, c = None):
        if c is None:
            c = ExLlamaV2Filter.__new__(ExLlamaV2Filter)
        c.model = self.model
        c.tokenizer = self.tokenizer
        c.sequence_str = self.sequence_str
        return c


    def begin(self, prefix_str):
        raise NotImplementedError


    def feed(self, token):
        raise NotImplementedError


    def next(self):
        raise NotImplementedError


    def use_background_worker(self) -> bool:
        """
        To indicate whether filter can/should run as a background thread. If True, next() will be called
        asynchronously after the CUDA workload has been scheduled for the following forward pass, instead of right
        before sampling. Should be True for any CPU-intensive filter such as a grammar constraint.
        """
        return False


    def background_next(self, pool: ThreadPoolExecutor):
        """
        Schedule next() via the provided thread pool executor
        """
        assert self.background_result is None
        self.background_result = pool.submit(self.next)


    def background_drop(self):
        """
        Clear the result of an asynchronous filter pass. Used when a complex filter reaches an end state and forces
        the selection of eos_token_id. next() could still be scheduled after this selection, leaving a pending result
        that would break subsequent generations with the same filter.
        """
        if self.background_result is not None:
            self.background_result.result()
            self.background_result = None


    def get_next(self) -> tuple:
        """
        Return either next() or the result of any scheduled call to next()
        """
        if self.background_result is None:
            return self.next()
        r = self.background_result.result()
        self.background_result = None
        return r