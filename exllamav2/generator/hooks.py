from __future__ import annotations
import torch
from typing import Callable
from dataclasses import dataclass, field

@dataclass
class ExLlamaV2PostSamplingResult:

    sampled_token: torch.Tensor | None = None
    sampled_prob: torch.Tensor | None = None
    candidate_tokens: torch.Tensor | None = None
    candidate_probs: torch.Tensor | None = None
    logits: torch.Tensor | None = None

    feed_filters: bool = True

ExLlamaV2PostSamplingHook = Callable[[ExLlamaV2PostSamplingResult], None]