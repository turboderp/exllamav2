from __future__ import annotations
import torch
from torch import nn
from exllamav2.module import ExLlamaV2Module
from exllamav2.ext import exllamav2_ext as ext_c, none_tensor

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from exllamav2.model import ExLlamaV2

class ExLlamaV2LayerNorm(ExLlamaV2Module):

    name: str = "LayerNorm"

    layernorm: nn.LayerNorm | None
    weight: nn.Parameter | None
    bias: nn.Parameter | None
    variance_epsilon: float


    def __init__(self,
                 model: ExLlamaV2,
                 key: str):
        super().__init__(model, key)

        self.layernorm = None
        self.weight = None
        self.bias = None
        self.variance_epsilon = 1e-6


    @torch.inference_mode
    def load(self):

        w = self.load_weight()

        if isinstance(w, tuple):
            weight = w[0]
            bias = w[1]
        else:
            weight = w
            bias = None

        assert isinstance(weight, nn.Parameter)
        assert bias is None or isinstance(bias, nn.Parameter)

        self.layernorm = nn.LayerNorm(self.model.config.hidden_size,
                                      elementwise_affine = True,
                                      bias = bias is not None)

        self.layernorm.weight = weight
        self.weight = weight

        if bias is not None:
            self.layernorm.bias = bias
            self.bias = bias

        self.variance_epsilon = self.model.config.norm_eps


    def numel(self):

        return 0
        # return self.layernorm.weight.data.numel()


    def unload(self):

        self.layernorm = None
        self.weight = None
        self.bias = None


    def get_weight(self) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:

        if self.bias is not None: return self.weight, self.bias
        return self.weight


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
                output_fp32 = False,  # TODO:
                **kwargs) -> torch.Tensor | dict[str: torch.Tensor]:

        output_shape = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        norm = torch.empty_like(hidden_states)
        ext_c.layer_norm(hidden_states,
                         self.weight.data,
                         self.bias.data if self.bias is not None else none_tensor,
                         norm,
                         self.variance_epsilon)

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
                      output_fp32 = False,  # TODO:
                      **kwargs) -> torch.Tensor | dict[str: torch.Tensor]:

        hidden_states = self.layernorm(hidden_states)

        if intermediates:
            return {"hidden_states": hidden_states}
        else:
            return hidden_states


