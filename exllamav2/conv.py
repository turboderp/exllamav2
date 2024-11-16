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

class ExLlamaV2Conv(ExLlamaV2Module):

    name: str = "Convolution"

    in_channels: int
    out_channels: int
    kernel_size: tuple
    has_bias: bool

    conv: nn.Conv2d | nn.Conv3d | None

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

        self.conv = None
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
            "Quantized Conv layer is not implemented."

        bias = isinstance(w, tuple)
        if len(self.kernel_size) == 2:
            self.conv = nn.Conv2d(
                in_channels = self.in_channels,
                out_channels = self.out_channels,
                kernel_size = (self.kernel_size[0], self.kernel_size[1]),
                stride = (self.kernel_size[0], self.kernel_size[1]),
                bias = bias)
        elif len(self.kernel_size) == 3:
            self.conv = nn.Conv3d(
                in_channels = self.in_channels,
                out_channels = self.out_channels,
                kernel_size = (self.kernel_size[0], self.kernel_size[1], self.kernel_size[2]),
                stride = (self.kernel_size[0], self.kernel_size[1], self.kernel_size[2]),
                bias = bias)
        else:
            raise ValueError("Only 2D and 3D convolutions allowed")

        # Load FP16 convolution layer without bias, optionally quantize to Q4

        if isinstance(w, nn.Parameter):
            assert not self.has_bias, self.key + " has no bias tensor but bias is expected"
            self.conv.weight = w

        # Load FP16 convolution layer with bias, optionally quantize to Q4

        elif isinstance(w, tuple):
            assert self.has_bias, self.key + " has bias tensor but bias is not expected"
            self.conv.weight = w[0]
            self.conv.bias = w[1]


    def numel(self) -> int:

        return self.in_channels * self.out_channels * self.kernel_size[0] * self.kernel_size[1]


    def unload(self):

        if self.conv is not None:
            del self.conv
            self.conv = None


    def get_weight(self) -> torch.Tensor:

        return self.conv.weight.data


    def get_bias_tensor(self) -> torch.Tensor:

        if self.conv is not None:
            return self.conv.bias.data
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

        if len(self.kernel_size) == 2:
            hidden_states = self.conv.forward(hidden_states)
            hidden_states = hidden_states.flatten(2).permute(0, 2, 1).contiguous()

        elif len(self.kernel_size) == 3:
            hidden_states = hidden_states.view(
                -1, self.in_channels, self.kernel_size[0], self.kernel_size[1], self.kernel_size[2]
            )
            hidden_states = self.conv.forward(hidden_states)
            hidden_states = hidden_states.view(-1, self.out_channels).unsqueeze(0)

        if intermediates:
            return {"hidden_states": hidden_states}
        else:
            return hidden_states
