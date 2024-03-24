import torch
from typing import Callable
from dataclasses import dataclass, field

@dataclass
class ExLlamaV2PostSamplingResult:

    sampled_token: torch.Tensor = None
    sampled_prob: torch.Tensor = None
    candidate_tokens: torch.Tensor = None
    candidate_probs: torch.Tensor = None
    logits: torch.Tensor = None

    feed_filters = True

ExLlamaV2PostSamplingHook = Callable[[ExLlamaV2PostSamplingResult], None]