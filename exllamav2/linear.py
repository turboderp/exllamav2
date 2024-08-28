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

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from exllamav2.lora import ExLlamaV2Lora
    from exllamav2.model import ExLlamaV2


class ExLlamaV2Linear(ExLlamaV2Module):

    name: str = "Linear"

    in_features: int
    in_features_tp: list[int] | None
    out_features: int
    out_features_tp: list[int] | None
    has_bias: bool
    prescale: float

    linear: nn.Linear | None
    q_handle: int | list[int | None] | None
    q_tensors: dict | list[dict | None] | None
    q4_weight: torch.Tensor | None
    q4_scales: torch.Tensor | None
    fp16_bias: torch.Tensor | None

    temp_dq: torch.Tensor | list[torch.Tensor]
    padding: int
    max_out_len: int

    lora_a_tensors: dict
    lora_b_tensors: dict

    f_key: str | None
    f_beg: int | None
    f_end: int | None

    is_tp: bool
    broadcast_type: int | None
    broadcast_type_out: int | None
    is_sub_module: bool

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
                 f_end: int = None,
                 is_sub_module: bool = True,
                 altpack_qkv: bool = False,
                 normalize_unq: bool = False):
        super().__init__(model, key)

        self.is_sub_module = is_sub_module

        self.is_tp = False
        self.broadcast_type = None
        self.broadcast_type_out = None

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
        self.altpack_qkv = altpack_qkv

        self.assumed_footprint = in_features * (out_features + self.padding) * 2 + 128
        self.normalize_unq = normalize_unq

        self.out_features_tp = None


    @torch.inference_mode
    def load(self,
             w: dict | nn.Parameter | tuple | None = None,
             device_context: bool = True,
             unmap: bool = False,
             output_map: torch.Tensor | None = None):

        cfg = self.model.config

        if self.f_key: w = self.load_weight_fused(self.f_key, self.f_beg, self.f_end, self.in_features, self.out_features, self.altpack_qkv)
        if w is None: w = self.load_weight(cpu = output_map is not None)

        # Load quantized linear layer from dictionary

        if isinstance(w, dict):
            assert not cfg.load_in_q4, "Can't load quantized layer in Q4 mode"
            if self.has_bias:
                assert "bias" in w, self.key + " has no bias but bias expected"
            else:
                assert "bias" not in w, self.key + " has bias but bias is not expected"
            if device_context:
                device_context = self.model.get_device_context(self.device_idx)
                device_context.begin_scratch_alloc()
                self.temp_dq = device_context.get_scratch_slice(self.temp_dq_size())
            else:
                self.temp_dq = none_tensor
            self.q_tensors = w

            if unmap and "q_perm" in w:
                perm = w["q_perm"].cpu()
                del w["q_perm"]
                del w["q_invperm"]
                # w["q_perm"] = torch.arange(0, w["q_perm"].shape[-1], dtype = w["q_perm"].dtype, device = w["q_perm"].device)
                # w["q_invperm"] = w["q_perm"].clone()
            else:
                perm = None

            if output_map is not None:
                ext_c.tensor_remap(w["q_weight"], output_map)
                ext_c.tensor_remap_4bit(w["q_scale"], output_map)
                for k in w.keys():
                    w[k] = safe_move_tensor(w[k], self.device())

            self.q_handle = ext.make_q_matrix(w,
                                              self.temp_dq,
                                              prescale = self.prescale,
                                              max_dq_rows = cfg.max_dq_size // self.out_features,
                                              offset_qzeros = cfg.checkpoint_offset_qzeros)
            self.prev_prescale = self.prescale
            self.prescale = 1

            if unmap:
                return perm

        # Load FP16 linear layer without bias, optionally quantize to Q4

        elif isinstance(w, nn.Parameter):
            assert not self.has_bias, self.key + " has no bias tensor but bias is expected"
            if self.normalize_unq:
                w = self.normalize(w)
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
            if self.normalize_unq:
                w = self.normalize(w[0]), w[1]
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


    def normalize(self, w: torch.Tensor):
        return nn.functional.normalize(w)


    def matrix_shape(self):

        return self.in_features, self.out_features


    def numel(self) -> int:

        return self.in_features * self.out_features


    def unload(self):

        if self.linear is not None:
            del self.linear
            self.linear = None

        if self.q_handle is not None:
            for h in (self.q_handle if isinstance(self.q_handle, list) else [self.q_handle]):
                ext_c.free_q_matrix(h)
            self.q_handle = None

        if self.q_tensors is not None:
            for t in (self.q_tensors if isinstance(self.q_tensors, list) else [self.q_tensors]):
                for k, v in t.items():
                    del v
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
               (self.temp_fwd_size() if self.is_sub_module else 0)


    def scratch_space(self) -> int:

        return self.temp_dq_size() + \
               (self.temp_fwd_size() if self.is_sub_module else 0)


    def temp_dq_size(self, in_features = None, out_features = None) -> int:

        dq = (self.in_features if in_features is None else in_features) * \
             (self.out_features if out_features is None else out_features)
        dq = 2 * min(dq, self.model.config.max_dq_size)
        return dq


    def scratch_space_tp(self, broadcast_type: int = BROADCAST_VC, dim = 1):

        ctx = self.model.tp_context
        split = ctx.get_split(broadcast_type)
        self.out_features_tp = [0] * ctx.num_devices
        for dev, a, b in split:
            self.out_features_tp[dev] = (b - a) * dim

        scratch = [self.in_features * of for of in self.out_features_tp]
        scratch = [2 * min(s, self.model.config.max_dq_size) for s in scratch]
        return scratch


    def temp_fwd_size(self) -> int:

        max_len = self.model.config.max_input_len if self.max_out_len is None else \
            min(self.max_out_len, self.model.config.max_input_len)
        return self.out_features * max_len * self.model.config.max_batch_size * 4 + 128


    def forward(
        self,
        hidden_states: torch.Tensor,
        cache = None,
        attn_params = None,
        past_len = None,
        intermediates: bool = False,
        loras: list[ExLlamaV2Lora] | None = None,
        force_recons: bool = False,
        force_cuda: bool = False,
        **kwargs
    ) -> torch.Tensor | dict[str: torch.Tensor]:

        if self.is_tp:
            if self.out_features_tp:
                return self.forward_tp(
                    hidden_states,
                    cache,
                    attn_params,
                    past_len,
                    intermediates,
                    loras,
                    force_recons,
                    force_cuda,
                    **kwargs
                )
            elif self.in_features_tp:
                return self.forward_tp_row(
                    hidden_states,
                    cache,
                    attn_params,
                    past_len,
                    intermediates,
                    loras,
                    force_recons,
                    force_cuda,
                    **kwargs
                )
            else:
                assert False, "Unitialized TP linear layer"

        # Linear forward

        if self.key == 'lm_head' and loras is not None and loras[0].lm_head is not None:
            hidden_states_out = loras[0].lm_head(hidden_states)

            if intermediates:
                return {"hidden_states": hidden_states_out}
            else:
                return hidden_states_out

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


    def forward_tp(
        self,
        hidden_states: torch.Tensor,
        cache = None,
        attn_params = None,
        past_len = None,
        intermediates: bool = False,
        loras: list[ExLlamaV2Lora] | None = None,
        force_recons: bool = False,
        force_cuda: bool = False,
        output_split: bool = False,
        dim: int = 1,
        **kwargs
    ) -> torch.Tensor | dict[str: torch.Tensor]:

        ctx = self.model.tp_context
        split = ctx.get_split(self.broadcast_type)

        if isinstance(hidden_states, torch.Tensor):
            output_shape = hidden_states.shape[:-1] + (self.out_features,)
            hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
            hidden_states = ctx.broadcast(hidden_states, self.broadcast_type)
        else:
            output_shape = hidden_states[0].shape[:-1] + (self.out_features,)
            hidden_states = [hs.view(-1, hs.shape[-1]) for hs in hidden_states]

        outputs = [
            torch.empty(
                (hs.shape[0], (b - a) * dim),
                device = hs.device,
                dtype = hs.dtype
            ) for hs, (dev, a, b) in zip(hidden_states, split)
        ]

        ext_c.gemm_half_q_half_tp(
            hidden_states,
            self.q_handle,
            outputs,
            force_cuda,
            ctx.ext_tp_context,
            -1
        )

        if output_split:
            return outputs

        output = ctx.gather(0, outputs, self.broadcast_type)
        hidden_states_out = output.view(output_shape)
        return hidden_states_out


    def forward_tp_row(
        self,
        hidden_states: list[torch.Tensor],
        cache = None,
        attn_params = None,
        past_len = None,
        intermediates: bool = False,
        loras: list[ExLlamaV2Lora] | None = None,
        force_recons: bool = False,
        force_cuda: bool = False,
        output_split: bool = False,
        dim: int = 1,
        **kwargs
    ) -> torch.Tensor | dict[str: torch.Tensor]:

        ctx = self.model.tp_context
        split = ctx.get_split(self.broadcast_type)

        assert isinstance(hidden_states, list)
        assert output_split

        rows = hidden_states[0].shape[0]
        dtype = hidden_states[0].dtype
        outputs = ctx.get_temp_tensors_bc(rows, dtype, self.broadcast_type_out)

        ext_c.gemm_half_q_half_tp(
            hidden_states,
            self.q_handle,
            outputs,
            force_cuda,
            ctx.ext_tp_context,
            -1
        )

        return outputs


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


    def tp_split(self, broadcast_type: int, dim = None):
        assert self.q_handle is not None, \
            "Can only split quantized tensor."
        assert all(x in self.q_tensors for x in [
            "q_scale",
            "q_scale_max",
            "q_group_map",
            "q_groups",
            "q_weight"
        ]), "Can only split fully loaded EXL2 tensor."

        cfg = self.model.config
        ctx = self.model.tp_context
        self.broadcast_type = broadcast_type
        split = ctx.get_split(broadcast_type)

        if dim:
            split = [(d, a * dim, b * dim) for (d, a, b) in split]

        self.out_features_tp = [0] * ctx.num_devices
        for dev, a, b in split:
            self.out_features_tp[dev] = b - a

        new_q_handle = []
        new_q_tensors = []
        new_temp_dq = []

        for idx, a, b in split:
            s = b - a
            if s == 0: continue

            w = {
                "q_scale": safe_move_tensor(self.q_tensors["q_scale"][:, a // 8:b // 8], idx).contiguous(),
                "q_scale_max": safe_move_tensor(self.q_tensors["q_scale_max"], idx).contiguous(),
                "q_group_map": safe_move_tensor(self.q_tensors["q_group_map"], idx).contiguous(),
                "q_groups": safe_move_tensor(self.q_tensors["q_groups"], idx).contiguous(),
                "q_weight": safe_move_tensor(self.q_tensors["q_weight"][:, a:b], idx).contiguous()
            }

            if "q_perm" in self.q_tensors:
                w.update({
                    "q_perm": safe_move_tensor(self.q_tensors["q_perm"], idx).contiguous(),
                    "q_invperm": safe_move_tensor(self.q_tensors["q_invperm"], idx).contiguous(),
                })

            if "bias" in self.q_tensors:
                w["bias"] = safe_move_tensor(self.q_tensors["bias"][a:b], idx).contiguous()

            new_q_tensors.append(w)

            device_context = self.model.get_device_context(idx)
            device_context.begin_scratch_alloc()
            new_temp_dq.append(device_context.get_scratch_slice(self.temp_dq_size(out_features = s)))
            max_dq_rows = cfg.max_dq_size // s

            new_q_handle.append(
                ext_c.make_q_matrix_split(
                    w["q_weight"],
                    w.get("q_perm", none_tensor),
                    w.get("q_invperm", none_tensor),
                    w["q_scale"],
                    w["q_scale_max"],
                    w["q_groups"],
                    w["q_group_map"],
                    none_tensor,
                    none_tensor,
                    none_tensor,
                    w.get("bias", none_tensor),
                    new_temp_dq[-1],
                    max_dq_rows
                )
            )

        ext_c.free_q_matrix(self.q_handle)
        self.q_handle = new_q_handle
        self.q_tensors = new_q_tensors
        self.temp_dq = new_temp_dq
        self.is_tp = True


    def tp_split_row(self, broadcast_type_in: int, broadcast_type_out: int, dim = None):
        assert self.q_handle is not None, \
            "Can only split quantized tensor."
        assert all(x in self.q_tensors for x in [
            "q_scale",
            "q_scale_max",
            "q_group_map",
            "q_groups",
            "q_weight"
        ]), "Can only split fully loaded EXL2 tensor."
        assert not any(x in self.q_tensors for x in [
            "q_perm",
            "q_invperm",
        ]), "Tensor not prepared for row split"

        cfg = self.model.config
        ctx = self.model.tp_context
        self.broadcast_type = broadcast_type_in
        self.broadcast_type_out = broadcast_type_out
        split = ctx.get_split(broadcast_type_in)

        if dim:
            split = [(d, a * dim, b * dim) for (d, a, b) in split]

        self.in_features_tp = [0] * ctx.num_devices
        for dev, a, b in split:
            self.in_features_tp[dev] = b - a

        new_q_handle = []
        new_q_tensors = []
        new_temp_dq = []

        q_scale = self.q_tensors["q_scale"]
        q_scale_max = self.q_tensors["q_scale_max"]
        q_group_map = self.q_tensors["q_group_map"]
        q_groups = self.q_tensors["q_groups"]
        q_weight = self.q_tensors["q_weight"]

        assert q_group_map[b * 2 - 1].item() == 1

        for idx, a, b in split:
            s = b - a
            if s == 0: continue

            groups = q_groups.shape[0] // 2
            group_a = q_group_map[a * 2].item()
            group_b = q_group_map[(b - 1) * 2].item() + 1
            row_a = q_groups[group_a * 2 + 1].item()
            row_b = q_weight.shape[0] if group_b == groups else q_groups[group_b * 2 + 1].item()

            tq_scale = q_scale[group_a:group_b, :]
            tq_scale_max = q_scale_max[group_a:group_b]
            tq_group_map = q_group_map[a * 2:b * 2]
            tq_groups = q_groups[group_a * 2:group_b * 2]
            tq_weight = q_weight[row_a:row_b, :]

            tq_group_map[::2] -= group_a
            tq_groups[1::2] -= row_a

            w = {
                "q_scale": safe_move_tensor(tq_scale, idx).clone().contiguous(),
                "q_scale_max": safe_move_tensor(tq_scale_max, idx).clone().contiguous(),
                "q_group_map": safe_move_tensor(tq_group_map, idx).clone().contiguous(),
                "q_groups": safe_move_tensor(tq_groups, idx).clone().contiguous(),
                "q_weight": safe_move_tensor(tq_weight, idx).clone().contiguous()
            }

            if "bias" in self.q_tensors:
                w["bias"] = safe_move_tensor(self.q_tensors["bias"], idx).contiguous()

            new_q_tensors.append(w)

            device_context = self.model.get_device_context(idx)
            device_context.begin_scratch_alloc()
            new_temp_dq.append(device_context.get_scratch_slice(self.temp_dq_size(in_features = s)))
            max_dq_rows = cfg.max_dq_size

            new_q_handle.append(
                ext_c.make_q_matrix_split(
                    w["q_weight"],
                    none_tensor,
                    none_tensor,
                    w["q_scale"],
                    w["q_scale_max"],
                    w["q_groups"],
                    w["q_group_map"],
                    none_tensor,
                    none_tensor,
                    none_tensor,
                    w.get("bias", none_tensor),
                    new_temp_dq[-1],
                    max_dq_rows
                )
            )

        ext_c.free_q_matrix(self.q_handle)
        self.q_handle = new_q_handle
        self.q_tensors = new_q_tensors
        self.temp_dq = new_temp_dq
        self.is_tp = True


