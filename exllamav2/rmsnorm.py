from __future__ import annotations
import torch
from torch import nn
from exllamav2.module import ExLlamaV2Module
from exllamav2.ext import exllamav2_ext as ext_c
from exllamav2.compat import safe_move_tensor

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from exllamav2.model import ExLlamaV2

class ExLlamaV2RMSNorm(ExLlamaV2Module):

    name: str = "RMSNorm"

    weight: nn.Parameter | None | list[nn.Parameter | None]
    bias: nn.Parameter | None | list[nn.Parameter | None]
    variance_epsilon: float

    is_tp: bool
    broadcast_type: int | None

    def __init__(self, model, key):
        super().__init__(model, key)

        self.is_tp = False
        self.broadcast_type = None

        self.weight = None
        self.bias = None
        self.variance_epsilon = 1e-6


    @torch.inference_mode
    def load(self, device_context = True):

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


    def scratch_space_tp(self) -> list[int]:

        return [0] * self.model.tp_context.num_devices


    def forward(
        self,
        hidden_states: torch.Tensor,
        cache = None,
        attn_params = None,
        past_len = None,
        intermediates: bool = False,
        loras = None,
        output_fp32 = False,
        **kwargs
    ) -> torch.Tensor | dict[str: torch.Tensor]:

        if self.is_tp:
            return self.forward_tp(
                hidden_states,
                cache,
                attn_params,
                past_len,
                intermediates,
                loras,
                output_fp32,
                **kwargs
            )

        output_shape = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])

        if not output_fp32:
            norm = torch.empty_like(hidden_states, dtype = torch.half)
        else:
            norm = torch.empty_like(hidden_states, dtype = torch.float)

        ext_c.rms_norm(hidden_states, self.weight, norm, self.variance_epsilon)
        hidden_states = norm.view(output_shape)

        if intermediates:
            return {"hidden_states": hidden_states}
        else:
            return hidden_states


    def forward_tp(
        self,
        hidden_states: torch.Tensor,
        cache = None,
        attn_params = None,
        past_len = None,
        intermediates: bool = False,
        loras = None,
        output_fp32 = False,
        **kwargs
    ) -> torch.Tensor | dict[str: torch.Tensor]:

        if isinstance(hidden_states, torch.Tensor):
            output_shape = hidden_states.shape
            hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
            hidden_states = self.model.tp_context.broadcast(0, hidden_states, self.broadcast_type)
        else:
            output_shape = hidden_states[0].shape
            hidden_states = [hs.view(-1, hs.shape[-1]) for hs in hidden_states]

        outputs = [torch.empty_like(hs) for hs in hidden_states]
        ext_c.rms_norm_tp(
            hidden_states,
            self.weight,
            outputs,
            self.variance_epsilon,
            self.model.tp_context.ext_tp_context
        )

        outputs = [x.view(output_shape) for x in outputs]
        return outputs


    def forward_torch(
        self,
        hidden_states: torch.Tensor,
        cache = None,
        attn_params = None,
        past_len = None,
        intermediates: bool = False,
        loras = None,
        output_fp32 = False,
        **kwargs
    ) -> torch.Tensor | dict[str: torch.Tensor]:

        # hidden_states.clamp_(-65504.0, 65504.0)

        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim = True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)

        if not output_fp32:
            hidden_states = hidden_states.to(self.weight.dtype)

        hidden_states *= self.weight

        if intermediates:
            return {"hidden_states": hidden_states}
        else:
            return hidden_states


    def tp_split(self, broadcast_type: int):

        cfg = self.model.config
        self.broadcast_type = broadcast_type
        split = self.model.tp_context.get_split(broadcast_type)
        maxdev = max(dev for dev, _, _ in split)

        new_weight = []
        new_bias = []

        for idx, a, b in split:
            s = b - a
            if s == 0: continue

            if self.weight is not None:
                if self.weight.device.index == idx:
                    new_weight.append(self.weight.data)
                else:
                    new_weight.append(safe_move_tensor(self.weight, idx))

            if self.bias is not None:
                if self.bias.device.index == idx:
                    new_bias.append(self.bias.data)
                else:
                    new_bias.append(safe_move_tensor(self.bias, idx))

        self.weight = new_weight
        self.bias = new_bias
        self.is_tp = True
