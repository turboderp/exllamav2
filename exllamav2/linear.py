from __future__ import annotations
import torch
import torch.nn.functional as F
from torch import nn
from exllamav2 import ext
from exllamav2.ext import exllamav2_ext as ext_c, none_tensor
from exllamav2.module import ExLlamaV2Module

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from exllamav2.lora import ExLlamaV2Lora
    from exllamav2.model import ExLlamaV2


class ExLlamaV2Linear(ExLlamaV2Module):

    name: str = "Linear"

    in_features: int
    out_features: int
    has_bias: bool
    prescale: float

    linear: nn.Linear | None
    q_handle: int | None
    q_tensors: dict | None
    q4_weight: torch.Tensor | None
    q4_scales: torch.Tensor | None
    fp16_bias: torch.Tensor | None

    temp_dq: torch.tensor
    padding: int
    max_out_len: int

    lora_a_tensors: dict
    lora_b_tensors: dict

    f_key: str | None
    f_beg: int | None
    f_end: int | None

    def __init__(self,
                 model: ExLlamaV2,
                 key: str,
                 in_features: int,
                 out_features: int,
                 has_bias: bool,
                 pad32: bool = True,
                 max_out_len: int | None = None,
                 prescale: float = 1,
                 f_key: str = None,
                 f_beg: int = None,
                 f_end: int = None):
        super().__init__(model, key)

        if pad32:
            self.padding = -out_features % 32
        else:
            self.padding = 0

        self.in_features = in_features
        self.out_features = out_features + self.padding
        self.has_bias = has_bias
        self.temp_dq = None
        self.footprint = -1
        self.max_out_len = max_out_len
        self.prescale = prescale
        self.prev_prescale = None

        self.linear = None
        self.q_handle = None
        self.q_tensors = None
        self.q4_weight = None
        self.q4_scales = None
        self.fp16_bias = None

        self.lora_a_tensors = {}
        self.lora_b_tensors = {}

        self.f_key = f_key
        self.f_beg = f_beg
        self.f_end = f_end

        self.assumed_footprint = in_features * (out_features + self.padding) * 2 + 128


    def load(self,
             w: dict | nn.Parameter | tuple | None = None,
             device_tensors: bool = True):

        if self.f_key: w = self.load_weight_fused(self.f_key, self.f_beg, self.f_end, self.in_features, self.out_features)
        if w is None: w = self.load_weight()

        # Load quantized linear layer from dictionary

        if isinstance(w, dict):
            assert not self.model.config.load_in_q4, "Can't load quantized layer in Q4 mode"
            if self.has_bias:
                assert "bias" in w, self.key + " has no bias but bias expected"
            else:
                assert "bias" not in w, self.key + " has bias but bias is not expected"
            if device_tensors:
                device_tensors = self.model.get_device_tensors(self.device_idx)
                device_tensors.begin_scratch_alloc()
                self.temp_dq = device_tensors.get_scratch_slice(self.temp_dq_size())
            else:
                self.temp_dq = none_tensor
            self.q_tensors = w
            self.q_handle = ext.make_q_matrix(w,
                                              self.temp_dq,
                                              prescale = self.prescale,
                                              max_dq_rows = self.model.config.max_dq_size // self.out_features)
            self.prev_prescale = self.prescale
            self.prescale = 1

        # Load FP16 linear layer without bias, optionally quantize to Q4

        elif isinstance(w, nn.Parameter):
            assert not self.has_bias, self.key + " has no bias tensor but bias is expected"
            if self.padding > 0: w = nn.Parameter(F.pad(w.data, (0, 0, 0, self.padding)).contiguous())
            if not self.model.config.load_in_q4 or not ".layers." in self.key:
                self.linear = nn.Linear(self.in_features, self.out_features, self.has_bias, device = "meta", dtype = torch.float16)
                self.linear.weight = w
            else:
                self.q4_weight = torch.empty((self.out_features * self.in_features // 2,), device = self.device(), dtype = torch.uint8)
                self.q4_scales = torch.empty((self.out_features * self.in_features // 32,), device = self.device(), dtype = torch.half)
                ext_c.matrix_fp16_to_q4(w.contiguous(), self.q4_weight, self.q4_scales)

        # Load FP16 linear layer with bias, optionally quantize to Q4

        elif isinstance(w, tuple):
            assert self.has_bias, self.key + " has bias tensor but bias is not expected"
            ww = w[0]
            wb = w[1]
            if self.padding > 0:
                ww = nn.Parameter(F.pad(ww.data, (0, 0, 0, self.padding)).contiguous())
                wb = nn.Parameter(F.pad(wb.data, (0, 0, 0, self.padding)).contiguous())
            if not self.model.config.load_in_q4 or not ".layers." in self.key:
                self.linear = nn.Linear(self.in_features, self.out_features, self.has_bias, device = "meta", dtype = torch.float16)
                self.linear.weight = ww
                self.linear.bias = wb
            else:
                self.q4_weight = torch.empty((self.out_features * self.in_features // 2,), device = self.device(), dtype = torch.uint8)
                self.q4_scales = torch.empty((self.out_features * self.in_features // 32,), device = self.device(), dtype = torch.half)
                ext_c.matrix_fp16_to_q4(ww.contiguous(), self.q4_weight, self.q4_scales)
                self.fp16_bias = wb


    def matrix_shape(self):

        return self.in_features, self.out_features


    def numel(self) -> int:

        return self.in_features * self.out_features


    def unload(self):

        if self.linear is not None:
            del self.linear
            self.linear = None

        if self.q_handle is not None:
            ext_c.free_q_matrix(self.q_handle)
            self.q_handle = None

        if self.q_tensors is not None:
            for k, v in self.q_tensors.items(): del v
            self.q_tensors = None

        if self.q4_weight is not None:
            del self.q4_weight
            self.q4_weight = None

        if self.q4_scales is not None:
            del self.q4_scales
            self.q4_scales = None

        if self.fp16_bias is not None:
            del self.fp16_bias
            self.fp16_bias = None

        self.temp_dq = None
        if self.prev_prescale is not None:
            self.prescale = self.prev_prescale
            self.prev_prescale = None


    def get_weight(self) -> torch.Tensor:

        return self.linear.weight.data


    def scratch_space_fixed(self) -> int:

        return self.temp_dq_size() + \
               self.temp_fwd_size()


    def scratch_space(self) -> int:

        return self.temp_dq_size() + \
               self.temp_fwd_size()


    def temp_dq_size(self) -> int:

        dq = self.in_features * self.out_features
        dq = min(dq, self.model.config.max_dq_size)
        dq = dq * 2 + 128
        return dq


    def temp_fwd_size(self) -> int:

        max_len = self.model.config.max_input_len if self.max_out_len is None else \
            min(self.max_out_len, self.model.config.max_input_len)
        return self.out_features * max_len * self.model.config.max_batch_size * 4 + 128


    def forward(self,
                hidden_states: torch.Tensor,
                cache = None,
                attn_params = None,
                past_len = None,
                intermediates: bool = False,
                loras: list[ExLlamaV2Lora] | None = None,
                force_recons: bool = False,
                force_cuda: bool = False,
                **kwargs) -> torch.Tensor | dict[str: torch.Tensor]:

        # Linear forward

        if self.q_handle is not None and not force_recons:

            output_shape = hidden_states.shape[:-1] + (self.out_features,)
            hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
            output = torch.empty((hidden_states.shape[0], self.out_features), dtype = torch.half, device = self.device())
            ext_c.gemm_half_q_half(hidden_states, self.q_handle, output, force_cuda)

            hidden_states_out = output.view(output_shape)

        else:

            matrix = self.get_weight_tensor_dq()
            hidden_states_out = torch.matmul(hidden_states, matrix)
            if self.has_bias:
                bias = self.get_bias_tensor()
                hidden_states_out += bias

            if self.prescale != 1:
                hidden_states_out.mul_(self.prescale)

        # Evaluate LoRAs

        if loras is not None:
            for lora in loras:
                lora_a = self.lora_a_tensors.get(lora)
                lora_b = self.lora_b_tensors.get(lora)
                if lora_a is not None:
                    assert lora_b is not None
                    temp = torch.matmul(hidden_states, lora_a)
                    hidden_states_out += torch.matmul(temp, lora_b)

        if intermediates:
            return {"hidden_states": hidden_states_out}
        else:
            return hidden_states_out


    def get_weight_tensor_dq(self) -> torch.Tensor:

        if self.linear is not None:
            return self.linear.weight.data.T

        elif self.q_handle is not None:
            tensor = torch.empty((self.in_features, self.out_features), dtype = torch.half, device = self.device())
            ext_c.reconstruct(self.q_handle, tensor)
            return tensor
            # ext_c.reconstruct(self.q_handle, self.temp_dq)
            # return self.temp_dq

        elif self.q4_weight is not None:
            tensor = torch.empty((self.out_features, self.in_features), dtype = torch.half, device = self.device())
            ext_c.matrix_q4_to_fp16(self.q4_weight, self.q4_scales, tensor)
            return tensor.T

        else:
            raise ValueError(f"Layer {self.key} has no data")


    def get_bias_tensor(self) -> torch.Tensor:

        if self.linear is not None:
            return self.linear.bias.data

        elif self.q_handle is not None:
            return self.q_tensors["bias"]

        elif self.fp16_bias is not None:
            return self.fp16_bias

        else:
            raise ValueError(f"Layer {self.key} has no data")


    def is_quant(self) -> bool:

        return self.q_handle is not None


    def rank_reduce(self, k: float):

        assert not self.is_quant(), "Can't rank-reduce quantized layer"

        weight = self.linear.weight.data.float()
        max_rank = min(weight.shape[0], weight.shape[1])
        desired_rank = int(max_rank * k)
        results = torch.svd_lowrank(weight, q = desired_rank, niter = 10)
        weight_approx = results[0] @ torch.diag(results[1]) @ results[2].T

        self.linear.weight = nn.Parameter(weight_approx.half())
