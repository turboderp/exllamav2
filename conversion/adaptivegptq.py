from __future__ import annotations

import gc
import torch
from torch import nn
import torch.nn.functional as F
import math
from exllamav2.ext import exllamav2_ext as ext_c, none_tensor
from exllamav2.util import list_live_tensors
import sys

class AdaptiveQuantizer:

    norm: float = 3.5
    max_p: float = 1.2
    min_p: float = 0.70
    p_grid: int = 96

    bits: int
    scale_bits: int
    scale_range: float = 1.0

    scale: torch.tensor
    qscale: torch.tensor
    qscale_max: float

    maxq: float
    scale_maxq: float
    qzero: float

    def __init__(self,
                 bits: int = 4,
                 scale_bits: int = 4):

        self.bits = bits
        self.scale_bits = scale_bits
        self.maxq = 2 ** bits - 1
        self.qzero = (self.maxq + 1) / 2
        self.scale_maxq = 2 ** scale_bits - 1

        self.scale_maxq = (2 ** self.scale_bits) - 1


    def find_params(self, x):

        xmax, _ = torch.max(torch.abs(x), dim = 0)
        xmax += 1e-12

        base_scale = xmax / (self.maxq / 2)
        qscale_max_t = torch.max(base_scale) * self.scale_range

        scale_tp = base_scale / qscale_max_t
        scale_tp = torch.sqrt(scale_tp)
        scale_tp *= (self.scale_maxq + 1)
        qscale_t = torch.clamp(torch.round(scale_tp), 1, self.scale_maxq + 1)
        qscale_tw = qscale_t / (self.scale_maxq + 1)
        qscale_tw = qscale_tw ** 2
        qscale_tw *= qscale_max_t

        q = torch.zeros((self.p_grid + 1, 128), dtype = torch.float, device = x.device)
        ext_c.quantize_err(x, q, qscale_tw, self.qzero, self.maxq, self.norm, self.min_p, self.max_p, self.p_grid)

        q = torch.sum(q, dim = 1)
        best_pi = torch.argmin(q)
        best_pif = best_pi / self.p_grid
        best_p = self.max_p * best_pif + self.min_p * (1 - best_pif)

        self.qscale = qscale_t.to(torch.short)
        self.scale = qscale_tw * best_p
        self.qscale_max = qscale_max_t * best_p

        # Make sure scales are rounded correctly for sanity test
        prescale = torch.tensor([1 / 256], dtype = torch.half, device = self.scale.device)
        self.scale = ((self.qscale * self.qscale).to(torch.half) * (self.qscale_max.half() * prescale)).float()


class AdaptiveGPTQ:

    percdamp: float = 0.12

    layer: nn.Linear
    device: torch.device

    group_size: int | dict
    bits: list
    bits_groups: list
    scale_bits: int
    hot_bits: int

    columns: int
    rows: int
    hessian: torch.tensor
    total_groups: int

    perm: torch.Tensor | None
    perm_cpu: torch.Tensor | None
    invperm: torch.Tensor | None

    # g_idx: torch.tensor = None
    scale: torch.Tensor | None
    qscale: torch.Tensor | None
    qscale_max: torch.Tensor | None
    qweight: torch.Tensor | None
    qgroups: torch.Tensor | None

    quant: torch.Tensor | None
    weights: torch.Tensor | None
    hessian: torch.Tensor | None
    hessian_inv: torch.Tensor | None
    num_samples: int = 0
    num_batches: int = 0

    quant_device: int = 0
    hessian_device: int = 0


    def __init__(self,
                 layer: nn.Linear):

        self.layer = layer
        self.device = layer.weight.device

        self.rows = self.layer.weight.data.shape[1]
        self.columns = self.layer.weight.data.shape[0]

        # self.weights = self.layer.weight.data.T.clone().float().contiguous()
        self.weights = None
        self.hessian = None
        self.num_samples = 0
        self.num_batches = 0

        self.perm = None
        self.perm_cpu = None
        self.inv_perm = None

        self.scale = None
        self.qscale = None
        self.qscale_max = None
        self.qweight = None
        self.qgroups = None

        self.quant_device = 0
        self.hessian_device = 0


    def drop_buffers(self):

        self.perm = None
        self.perm_cpu = None
        self.invperm = None
        # self.g_idx = None
        self.scale = None
        self.qscale = None
        self.qscale_max = None
        self.qweight = None
        self.qgroups = None
        self.quant = None
        self.weights = None
        self.hessian = None
        self.hessian_inv = None

        gc.collect()
        torch.cuda.empty_cache()


    def configure(self,
                  group_size: dict,
                  bits = None,
                  bits_prop = None,
                  scale_bits: int = 4
                  ):

        self.group_size = group_size
        self.scale_bits = scale_bits
        self.bits = bits

        assert isinstance(bits, list)
        assert isinstance(bits_prop, list)
        assert sum(bits_prop) == 1

        groups = 0
        remaining_rows = self.rows
        self.bits_groups = []
        for b, p in zip(self.bits, bits_prop):
            assert p > 0
            gsz = self.group_size[b]
            g = math.ceil(min(self.rows * p, remaining_rows) / gsz)
            groups += g
            remaining_rows -= g * gsz
            self.bits_groups.append(g)

        assert remaining_rows <= 0

        self.total_groups = groups


    def add_batch(self, inputs):

        with torch.inference_mode():

            # dim = inputs.shape[-1]
            # inputs = inputs.view((-1, dim)).float().T.to("cuda:0")
            # ns = 1
            #
            # if self.hessian is None:
            #     self.hessian = torch.zeros((dim, dim), device = self.device, dtype = torch.float)
            # else:
            #     self.hessian.mul_(self.num_samples / (self.num_samples + ns))
            # self.num_samples += ns
            # inputs.mul_(math.sqrt(2 / self.num_samples))
            # self.hessian.addmm_(inputs, inputs.T)

            # self.num_batches += 1
            # num_samples = len(inputs)
            # # inputs = torch.cat(inputs, dim = 0)
            # inputs = inputs.view((-1, inputs.shape[-1])).float().T.to("cuda:0")
            # inputs *= math.sqrt(2 / num_samples)
            # self.hessian += inputs.matmul(inputs.T)

            if self.hessian is None:
                self.hessian = torch.zeros((self.rows, self.rows), device=self.device, dtype=torch.float)

            self.num_batches += 1
            inputs = inputs.view((-1, inputs.shape[-1])).float().to("cuda:0")
            self.hessian.addmm_(inputs.T, inputs)


    def prepare(self, no_h_inv = False):

        with torch.inference_mode():

            self.hessian /= self.num_batches
            diagonal = torch.diag(self.hessian)

            # Prepare weights

            self.weights = self.layer.weight.data.cpu().T.clone().float().contiguous()

            # Zero weights that have no impact. Disabling this since it feels a little drastic based on just the calibration
            # data. It likely never triggers, anyway.

            # dead = diagonal == 0.0
            # self.hessian[dead, dead] = 1
            # self.weights[dead, :] = 0

            # Activation order

            self.perm = torch.argsort(diagonal, descending = True)
            self.perm_cpu = self.perm.cpu()
            self.weights = self.weights[self.perm_cpu, :]

            if self.hessian.numel() > 6e8:
                hessian_cpu = self.hessian.cpu()
                self.hessian = None
                hessian = hessian_cpu[self.perm_cpu][:, self.perm_cpu]
                hessian = hessian.to("cuda:0")
            else:
                hessian = self.hessian[self.perm][:, self.perm]
                self.hessian = None

            # In case numerical errors have caused some asymmetry in H, assume it's close to symmetrical and force it.
            # (Doesn't seem to be needed)

            # torch.cuda.empty_cache()
            # hessian = (hessian + hessian.T) * 0.5
            # torch.cuda.empty_cache()

            # Damping

            diagonal = torch.diag(hessian)
            damp = torch.clamp(self.percdamp * torch.mean(diagonal), min = 1e-5)

            # Inverse of H

            attempts = 0
            while not no_h_inv:

                try:

                    d = torch.arange(self.rows, device = self.device)
                    hessian[d, d] += damp

                    current_device = self.hessian_device
                    max_devices = torch.cuda.device_count()

                    done = False
                    fail_device = False
                    while not done:
                        try:
                            hessian = hessian.to(torch.device(current_device))

                            hessian_inv = torch.linalg.cholesky(hessian)
                            hessian_inv = torch.cholesky_inverse(hessian_inv)

                            # The Cholesky inverse will sometimes fail to compute due to accumulated rounding errors when H
                            # is very large (e.g. 70B MLP down proj) and a lot of calibration data is used (e.g. 100 rows of
                            # 4096 tokens). This won't always throw an exception and sometimes just results in a NaN tensor.

                            if torch.any(torch.isnan(hessian_inv)): raise RuntimeError

                            # Test inversion

                            hessian_inv = torch.linalg.cholesky(hessian_inv, upper = True)
                            hessian_inv = hessian_inv.contiguous()

                            done = True
                            break

                        except torch.cuda.OutOfMemoryError as e:
                            current_device += 1
                            print(f" !! Out of memory (H), moving to device {current_device}")
                            if current_device == max_devices:
                                raise e
                            self.hessian_device = current_device

                    if done: break

                except RuntimeError as runtime_error:

                    if "out of memory" in str(runtime_error):
                        raise runtime_error

                    # If inverting failed, assume there were non-positive eigenvalues, so apply more damping to shift
                    # the eigenvalues in a positive direction.

                    print(" !! Warning: Applied additional damping")

                    attempts += 1
                    if attempts == 10:
                        raise ValueError("Hessian is not invertible")

            # Swap H to system RAM

            self.hessian_inv = None if no_h_inv else hessian_inv.cpu()
            self.hessian = None


    def reuse_h(self, other):

        with torch.inference_mode():

            # Prepare weights

            self.weights = self.layer.weight.data.cpu().T.clone().float().contiguous()

            self.hessian_inv = other.hessian_inv
            self.hessian = None
            self.perm = other.perm
            self.perm_cpu = other.perm_cpu
            self.weights = self.weights[self.perm_cpu, :]


    def quantize_rtn_inplace(self, keep_qweight = False, apply = False):
        assert apply and keep_qweight

        with torch.inference_mode():

            current_device = self.quant_device
            max_devices = torch.cuda.device_count()

            weights_cpu = self.weights.cpu().contiguous()

            done = False
            while not done:

                try:

                    weights = weights_cpu.to(torch.device(current_device))
                    self.qweight = torch.zeros_like(weights, dtype = torch.short, device = torch.device(current_device))

                    num_groups = 0
                    for bits_idx in range(len(self.bits)):
                        num_groups += self.bits_groups[bits_idx]

                    scale = []
                    qscale = []
                    qscale_max = torch.empty((num_groups,), dtype = torch.float, device = torch.device(current_device))
                    qgroups = []

                    group_idx = 0
                    group_idx_list = []

                    b = 0
                    for bits_idx, bits in enumerate(self.bits):
                        quantizer = AdaptiveQuantizer(bits = bits, scale_bits = self.scale_bits)

                        for group in range(self.bits_groups[bits_idx]):
                            a = b
                            b = min(a + self.group_size[bits], self.rows)

                            qgroups.append(bits)
                            qgroups.append(0)

                            quantizer.find_params(weights[a : b, :])
                            scale.append(quantizer.scale)
                            qscale.append(quantizer.qscale)
                            qscale_max[group_idx] = quantizer.qscale_max

                            ext_c.quantize_range_inplace(weights,
                                                         quantizer.scale,
                                                         self.qweight,
                                                         quantizer.qzero,
                                                         quantizer.maxq,
                                                         a,
                                                         b)

                            group_idx_list += [group_idx] * (b - a)
                            group_idx += 1

                    done = True

                except torch.cuda.OutOfMemoryError as e:
                    current_device += 1
                    print(f" !! Out of memory (Q), moving to device {current_device}")
                    if current_device == max_devices:
                        raise e
                    self.quant_device = current_device

            # Create g_idx to store inverse activation order

            self.invperm = torch.argsort(self.perm).to("cuda:0")

            # Store scales

            self.scale = torch.stack(scale, dim = 0).to("cuda:0")
            self.qscale = torch.stack(qscale, dim = 0).to("cuda:0")
            self.qscale_max = qscale_max.to(torch.float16).to("cuda:0")
            self.qgroups = torch.tensor(qgroups, dtype = torch.short).to("cuda:0")

            # I love Python

            scale = None
            qscale = None
            qscale_max = None
            qgroups = None
            group_idx_list = None

            qc = weights.cpu()
            qc = qc.to(torch.half)
            invperm = self.invperm.cpu()
            q = qc[invperm, :].T
            q = q.reshape(weights.T.shape)

            dev = weights.device
            weights = None
            torch.cuda.synchronize()
            gc.collect()
            torch.cuda.empty_cache()

            q = q.to(dev)
            self.layer.weight.data = q
            self.weights = q.T


    def quantize(self, keep_qweight = False, apply = False):

        with torch.inference_mode():

            current_device = self.quant_device
            max_devices = torch.cuda.device_count()

            weights_cpu = self.weights.cpu()

            done = False
            while not done:

                try:

                    hessian_inv_cuda = self.hessian_inv.to(torch.device(current_device))

                    # if apply:
                    #     weights = self.weights
                    #     self.layer.weight.data = torch.zeros((1, 1), dtype = torch.float32, device = weights.device)
                    # else:
                    #     weights = self.weights.clone()
                    weights = weights_cpu.to(torch.device(current_device))

                    self.quant = torch.zeros_like(weights_cpu, device = torch.device(current_device))

                    if keep_qweight:
                        self.qweight = torch.zeros_like(weights, dtype = torch.short)

                    # Quantize groups

                    num_groups = 0
                    for bits_idx in range(len(self.bits)):
                        num_groups += self.bits_groups[bits_idx]

                    scale = []
                    qscale = []
                    qscale_max = torch.empty((num_groups,), dtype = torch.float, device = torch.device(current_device))
                    qgroups = []

                    error = weights.clone()
                    group_idx = 0
                    group_idx_list = []

                    b = 0
                    for bits_idx, bits in enumerate(self.bits):
                        quantizer = AdaptiveQuantizer(bits = bits, scale_bits = self.scale_bits)

                        for group in range(self.bits_groups[bits_idx]):
                            a = b
                            b = min(a + self.group_size[bits], self.rows)

                            qgroups.append(bits)
                            qgroups.append(0)

                            quantizer.find_params(weights[a : b, :])
                            scale.append(quantizer.scale)
                            qscale.append(quantizer.qscale)
                            qscale_max[group_idx] = quantizer.qscale_max

                            ext_c.quantize_range(self.quant,
                                                 quantizer.scale,
                                                 self.qweight if keep_qweight else none_tensor,
                                                 quantizer.qzero,
                                                 quantizer.maxq,
                                                 hessian_inv_cuda,
                                                 weights,
                                                 error,
                                                 a,
                                                 b)

                            group_idx_list += [group_idx] * (b - a)
                            group_idx += 1

                    done = True

                except torch.cuda.OutOfMemoryError as e:
                    current_device += 1
                    print(f" !! Out of memory (Q), moving to device {current_device}")
                    if current_device == max_devices:
                        raise e
                    self.quant_device = current_device

            # Create g_idx to store inverse activation order

            # self.g_idx = torch.tensor(group_idx_list, dtype = torch.int32, device = self.device)
            # self.g_idx = torch.tensor(group_idx_list, dtype = torch.int32)

            self.quant = self.quant.to("cuda:0")
            self.invperm = torch.argsort(self.perm).to("cuda:0")
            # self.g_idx = self.g_idx[self.invperm]

            # Store scales

            self.scale = torch.stack(scale, dim = 0).to("cuda:0")
            self.qscale = torch.stack(qscale, dim = 0).to("cuda:0")
            self.qscale_max = qscale_max.to(torch.float16).to("cuda:0")
            self.qgroups = torch.tensor(qgroups, dtype = torch.short).to("cuda:0")

            # I love Python

            weights = None
            error = None
            scale = None
            qscale = None
            qscale_max = None
            qgroups = None
            group_idx_list = None

            # Apply

            if apply:
                self.apply_quant()


    def quant_error(self):

        with torch.inference_mode():

            q = self.quant[self.invperm, :]
            diff = torch.abs(q - self.layer.weight.data.T)
            mat_error_1 = (diff > 0.01).sum().item() / diff.numel()
            mat_error_5 = (diff > 0.05).sum().item() / diff.numel()
            mat_error_10 = (diff > 0.10).sum().item() / diff.numel()
            return mat_error_1, mat_error_5, mat_error_10


    def apply_quant(self):

        self.hessian = None

        qc = self.quant.cpu()
        invperm = self.invperm.cpu()
        q = qc[invperm, :].T
        q = q.reshape(self.quant.T.shape)

        gc.collect()
        torch.cuda.empty_cache()

        q = q.to(torch.device(self.quant_device))
        self.layer.weight.data = q


    def apply_temp(self):

        q = self.quant[self.invperm, :].T
        temp_layer = nn.Linear(self.layer.in_features, self.layer.out_features, False, device = "meta", dtype = torch.float16)
        temp_layer.weight = nn.Parameter(q.reshape(self.layer.weight.shape).type_as(self.layer.weight.data))
        return temp_layer


    def pack(self, key, qparams):

        self.qgroups = self.qgroups.to("cuda:0")
        # self.qscale_max = self.qscale_max.to("cude:0")

        assert qparams.scale_bits in [4]
        # assert self.columns % 32 == 0

        output = {}
        output[key + ".q_invperm"] = self.invperm.to(torch.int)
        output[key + ".q_scale_max"] = self.qscale_max
        output[key + ".q_groups"] = self.qgroups
        if self.layer.bias is not None:
            output[key + ".bias"] = self.layer.bias.data

        columns = self.columns
        rem_rows = self.rows
        padding = -columns % 32

        if padding != 0:
            print(f" !! Note: Padding quantized tensor {key}")
            qst = F.pad(self.qscale, (0, padding)).contiguous()
            qwt = F.pad(self.qweight, (0, padding)).contiguous()
        else:
            qst = self.qscale
            qwt = self.qweight

        qst_packed = torch.zeros((qst.shape[0], qst.shape[1] * qparams.scale_bits // 32), dtype = torch.int32, device = self.device)
        if qparams.scale_bits == 4: ext_c.pack_rows_4(qst, qst_packed)
        output[key + ".q_scale"] = qst_packed

        qwt_packed = []

        i = 0
        row = 0
        out_row = 0
        while i < self.qscale.shape[0]:

            bits = self.qgroups[i * 2].item()
            self.qgroups[i * 2 + 1] = out_row
            i += 1

            rows = min(self.group_size[bits], rem_rows)
            wpqr = 32 / bits
            qrows = rows / wpqr
            assert i == self.qgroups.shape[-1] or qrows == int(qrows)
            qrows = math.ceil(qrows)

            g_qwt = qwt[row:row+rows, :].contiguous()
            g_qwt_packed = torch.zeros((qrows, columns + padding), dtype = torch.int32, device = g_qwt.device)

            if padding > 0: g_qwt[:, -padding:] = 2 ** (bits - 1)

            ext_c.pack_columns(g_qwt, g_qwt_packed, bits)
            qwt_packed.append(g_qwt_packed)

            # print(row, rows, bits)

            row += rows
            out_row += qrows
            rem_rows -= rows


        qwt_packed = torch.cat(qwt_packed, dim = 0)
        output[key + ".q_weight"] = qwt_packed

        return output



