from __future__ import annotations
import torch
import torch.nn.functional as F
from torch import nn
from exllamav2 import ext
from exllamav2.ext import exllamav2_ext as ext_c, none_tensor
from exllamav2.module import ExLlamaV2Module
from exllamav2.compat import safe_move_tensor
from exllamav2.tensor_p import BROADCAST_VC
from exllamav2.util import unpack_4bit, pack_4bit
import gc
from exllamav2.experimental.fpx import fpxify

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from exllamav2.lora import ExLlamaV2Lora
    from exllamav2.model import ExLlamaV2

class ExLlamaV2Conv2D(ExLlamaV2Module):

    name: str = "Convolution"

    in_channels: int
    out_channels: int
    kernel_size: tuple[int]
    has_bias: bool

    conv2d: nn.Conv2d | None

    def __init__(
        self,
        model: ExLlamaV2,
        key: str,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple,
        has_bias: bool,
        archparams = None
    ):
        super().__init__(model, key, archparams)

        self.archparams = archparams

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.has_bias = has_bias

        self.conv2d = None
        self.assumed_footprint = self.numel() * 2

    @torch.inference_mode
    def load(
        self,
        w: dict | nn.Parameter | tuple | None = None,
        device_context: bool = True,
        unmap: bool = False,
        output_map: torch.Tensor | None = None
    ):
        cfg = self.model.config

        if w is None: w = self.load_weight(cpu = False)

        assert not isinstance(w, dict), \
            "Quantized Conv2D layer is not implemented."

        # Load FP16 linear layer without bias, optionally quantize to Q4

        if isinstance(w, nn.Parameter):
            assert not self.has_bias, self.key + " has no bias tensor but bias is expected"
            self.conv2d = nn.Conv2d(
                in_channels = self.in_channels,
                out_channels = self.out_channels,
                kernel_size = self.kernel_size,
                stride = self.kernel_size,
                bias = False)
            self.conv2d.weight = w

        # Load FP16 linear layer with bias, optionally quantize to Q4

        elif isinstance(w, tuple):
            assert self.has_bias, self.key + " has bias tensor but bias is not expected"
            ww = w[0]
            wb = w[1]
            self.conv2d = nn.Conv2d(
                in_channels = self.in_channels,
                out_channels = self.out_channels,
                kernel_size = self.kernel_size,
                stride = self.kernel_size,
                bias = False)
            self.conv2d.weight = ww
            self.conv2d.bias = wb


    def numel(self) -> int:

        return self.in_channels * self.out_channels * self.kernel_size[0] * self.kernel_size[1]


    def unload(self):

        if self.conv2d is not None:
            del self.linear
            self.conv2d = None


    def get_weight(self) -> torch.Tensor:

        return self.conv2d.weight.data


    def get_bias_tensor(self) -> torch.Tensor:

        if self.conv2d is not None:
            return self.linear.bias.data
        else:
            raise ValueError(f"Layer {self.key} has no data")


    def is_quant(self) -> bool:

        return False


    def scratch_space_fixed(self) -> int:

        return 0


    def scratch_space(self) -> int:

        return 0


    def temp_dq_size(self, in_features = None, out_features = None) -> int:

        return 0


    def temp_fwd_size(self) -> int:

        return 0


    def forward(
        self,
        hidden_states: torch.Tensor,
        cache = None,
        attn_params = None,
        past_len = None,
        intermediates: bool = False,
        loras: list[ExLlamaV2Lora] | None = None,
        **kwargs
    ) -> torch.Tensor | dict[str: torch.Tensor]:

        hidden_states = self.conv2d.forward(hidden_states)
        hidden_states = hidden_states.flatten(2).permute(0, 2, 1).contiguous()

        if intermediates:
            return {"hidden_states": hidden_states}
        else:
            return hidden_states
