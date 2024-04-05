from __future__ import annotations
import torch
from torch import nn
from exllamav2.module import ExLlamaV2Module
from exllamav2.ext import exllamav2_ext as ext_c

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from exllamav2.model import ExLlamaV2

class ExLlamaV2RMSNorm(ExLlamaV2Module):

    name: str = "RMSNorm"

    weight: nn.Parameter | None
    bias: nn.Parameter | None
    variance_epsilon: float


    def __init__(self, model, key):
        super().__init__(model, key)

        self.weight = None
        self.bias = None
        self.variance_epsilon = 1e-6


    def load(self):


        w = self.load_weight()

        if isinstance(w, tuple):
            self.weight = w[0]
            self.bias = w[1]
        else:
            self.weight = w
            self.bias = None

        assert isinstance(self.weight, nn.Parameter)
        assert self.bias is None, "RMSNorm does not support bias"
        # or isinstance(self.bias, nn.Parameter)

        self.variance_epsilon = self.model.config.norm_eps

        # Gemma adds 1 to the norm tensor for some reason
        if self.model.config.arch.norm_constant_bias != 0:
            self.weight += self.model.config.arch.norm_constant_bias


    def unload(self):

        if self.weight is not None:
            del self.weight
            self.weight = None

        if self.bias is not None:
            del self.bias
            self.bias = None


    def numel(self):

        return 0
        # return self.weight.numel()


    def get_weight(self) -> torch.Tensor:

        # Make sure to return the original weight tensor for Gemma
        if self.model.config.arch.norm_constant_bias != 0:
            return self.weight.data - self.model.config.arch.norm_constant_bias

        return self.weight.data


    def weight_footprint(self) -> int:

        hidden_size = self.model.config.hidden_size
        return hidden_size * 2


    def scratch_space_fixed(self) -> int:

        return 0


    def scratch_space(self) -> int:

        return 0


    def forward(self,
                hidden_states: torch.Tensor,
                cache = None,
                attn_params = None,
                past_len = None,
                intermediates: bool = False,
                loras = None,
                **kwargs) -> torch.Tensor | dict[str: torch.Tensor]:

        output_shape = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        norm = torch.empty_like(hidden_states)
        ext_c.rms_norm(hidden_states, self.weight, norm, self.variance_epsilon)
        hidden_states = norm.view(output_shape)

        if intermediates:
            return {"hidden_states": hidden_states}
        else:
            return hidden_states


    def forward_torch(self,
                      hidden_states: torch.Tensor,
                      cache = None,
                      attn_params = None,
                      past_len = None,
                      intermediates: bool = False,
                      loras = None,
                      **kwargs) -> torch.Tensor | dict[str: torch.Tensor]:

        hidden_states[hidden_states == -float('inf')] = -65504.0
        hidden_states[hidden_states == float('inf')] = 65504.0

        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim = True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        hidden_states = hidden_states.to(self.weight.dtype)
        hidden_states *= self.weight

        if intermediates:
            return {"hidden_states": hidden_states}
        else:
            return hidden_states


