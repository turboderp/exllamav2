
import torch.nn as nn
import torch.nn.functional as F

from exllamav2 import ExLlamaV2
from exllamav2.config import ExLlamaV2Config
from exllamav2.module import ExLlamaV2Module
from exllamav2.mlp import ExLlamaV2MLP
from typing import Callable

class ExLlamaV2MultimodalProjector(ExLlamaV2):

    config: ExLlamaV2Config
    modules: list[ExLlamaV2Module]

    def __init__(
        self,
        config: ExLlamaV2Config
    ):
        self.config = config
        cfg = self.config
        self.archparams = cfg.arch.mmp

        self.modules = [
            ExLlamaV2MLP(
                self,
                cfg.arch.mmp_prefix,
                0,
                archparams = cfg.arch.mmp,
                in_features = cfg.vision_hidden_size,
                out_features = cfg.hidden_size,
                interm_features = cfg.hidden_size,
                has_norm = False,
                has_residual = False,
            )
        ]

    def forward(self, x):

        for m in self.modules:
            x = m.forward(x)
        return x
