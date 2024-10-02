from __future__ import annotations
from threading import Lock
from concurrent.futures import ThreadPoolExecutor, Future
from exllamav2 import ExLlamaV2, ExLlamaV2Tokenizer
from exllamav2.util import timed
from exllamav2.ext import exllamav2_ext as ext_c, none_tensor
import torch

class ExLlamaV2Filter:

    # Internal state

    model: ExLlamaV2
    tokenizer: ExLlamaV2Tokenizer
    sequence_str: str

    background_result: Future | None

    # For compatibility
    allow_return_type_list: bool = True

    def __init__(self,
                 model: ExLlamaV2,
                 tokenizer: ExLlamaV2Tokenizer):

        self.model = model
        self.tokenizer = tokenizer
        self.sequence_str = ""
        self.background_result = None


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


    def can_mask_logits(self) -> bool:
        """
        To indicate whether filter can apply logit mask directly. If all filters in a stack have this property,
        mask_logits will be used exclusively.
        """
        return False


    def prepare_logit_mask(self) -> bool:
        """
        Called in place of next() to precompute logit mask before applying.
        """
        raise NotImplementedError


    def mask_logits(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Directly set all excluded logits in provided tensor to a value of -inf.
        """
        raise NotImplementedError


    def background_next(self, pool: ThreadPoolExecutor):
        """
        Schedule next() via the provided thread pool executor
        """
        assert self.background_result is None
        self.background_result = pool.submit(self.next)


    def background_prepare_logit_mask(self, pool: ThreadPoolExecutor):
        """
        Schedule prepare_logit_mask() via the provided thread pool executor
        """
        assert self.background_result is None
        self.background_result = pool.submit(self.prepare_logit_mask)


    def background_drop(self):
        """
        Clear the result of an asynchronous filter pass. Used when a complex filter reaches an end state and forces
        the selection of eos_token_id. next() could still be scheduled after this selection, leaving a pending result
        that would break subsequent generations with the same filter.
        """
        if self.background_result is not None:
            self.background_result.result()
            self.background_result = None


    def get_next(self, mask: bool = False) -> tuple:
        """
        Return either next() or the result of any scheduled call to next()
        """
        if self.background_result is None:
            return self.prepare_logit_mask() if mask else self.next()
        r = self.background_result.result()
        self.background_result = None
        return r