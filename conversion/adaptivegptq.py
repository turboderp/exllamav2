import torch
from torch import nn
import torch.nn.functional as F
import math
from exllamav2.ext import exllamav2_ext as ext_c, none_tensor


class AdaptiveQuantizer:

    norm: float = 3.5
    max_p: float = 1.0
    min_p: float = 0.75
    p_grid: int = 48

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

        # best_p = 1.0

        self.qscale = qscale_t.to(torch.short)
        self.scale = qscale_tw * best_p
        self.qscale_max = qscale_max_t * best_p


class AdaptiveGPTQ:

    percdamp: float = 0.07

    layer: nn.Linear
    device: torch.device

    group_size: int
    bits: list
    bits_groups: list
    scale_bits: int
    hot_bits: int

    columns: int
    rows: int
    hessian: torch.tensor
    total_groups: int

    perm: torch.tensor = None
    invperm: torch.tensor = None

    g_idx: torch.tensor = None
    scale: torch.tensor = None
    qscale: torch.tensor = None
    qscale_max: torch.tensor = None
    qweight: torch.tensor = None
    qgroups: torch.tensor = None

    quant: torch.tensor = None
    weights: torch.tensor = None
    hessian: torch.tensor = None
    hessian_inv: torch.tensor = None
    num_samples: int = 0
    num_batches: int = 0


    def __init__(self,
                 layer: nn.Linear):

        self.layer = layer
        self.device = layer.weight.device

        self.rows = self.layer.weight.data.shape[1]
        self.columns = self.layer.weight.data.shape[0]

        self.weights = self.layer.weight.data.T.clone().float().contiguous()
        self.hessian = None
        self.num_samples = 0
        self.num_batches = 0


    def configure(self,
                  group_size: int = 128,
                  bits = None,
                  bits_prop = None,
                  scale_bits: int = 4
                  ):

        self.group_size = group_size
        self.scale_bits = scale_bits

        self.total_groups = (self.rows + self.group_size - 1) // self.group_size

        if isinstance(bits, list):

            self.bits = bits
            g128 = (self.rows + 128 - 1) // 128
            self.bits_groups = [max(round(g128 * p), 1) * 128 // self.group_size for p in bits_prop]
            e = sum(self.bits_groups) - self.total_groups
            self.bits_groups[-1] -= e

        else:

            self.bits = [bits]
            self.bits_groups = [self.total_groups]


    def num_bits(self, subtract_columns = 0):

        gi = self.g_idx.numel() * 32
        qs = self.qscale.numel() * self.scale_bits
        qss = self.qscale_max.numel() * 16

        w = 0
        tr = self.rows
        for g, b in zip(self.bits_groups, self.bits):

            c = self.columns - subtract_columns
            r = self.group_size * g
            if r > tr: r = tr
            tr -= r
            w += r * c * b

        return w + gi + qs + qss


    def add_batch(self, inputs):

        with torch.inference_mode():

            if self.hessian is None:
                self.hessian = torch.zeros((self.rows, self.rows), device=self.device, dtype=torch.float)

            self.num_batches += 1
            num_samples = len(inputs)
            inputs = torch.cat(inputs, dim = 0)
            inputs = inputs.view((-1, inputs.shape[-1])).float().T.to("cuda:0")
            inputs *= math.sqrt(2 / num_samples)
            self.hessian += inputs.matmul(inputs.T)


    def prepare(self):

        with torch.inference_mode():

            self.hessian /= self.num_batches

            diagonal = torch.diag(self.hessian)

            # Zero weights that have no impact. Disabling this since it feels a little drastic based on just the calibration
            # data. It likely never triggers, anyway.

            # dead = diagonal == 0.0
            # self.hessian[dead, dead] = 1
            # self.weights[dead, :] = 0

            # Activation order

            self.perm = torch.argsort(diagonal, descending = True)
            self.weights = self.weights[self.perm, :]
            hessian = self.hessian[self.perm][:, self.perm]
            self.hessian = None

            # In case numerical errors have caused some asymmetry in H, assume it's close to symmetrical and force it.

            torch.cuda.empty_cache()
            hessian = (hessian + hessian.T) * 0.5
            torch.cuda.empty_cache()

            # Damping

            diagonal = torch.diag(hessian)
            damp = torch.clamp(self.percdamp * torch.mean(diagonal), min = 1e-5)

            # Inverse of H

            attempts = 0
            while True:

                try:

                    d = torch.arange(self.rows, device = self.device)
                    hessian[d, d] += damp

                    hessian_inv = torch.linalg.cholesky(hessian)
                    hessian_inv = torch.cholesky_inverse(hessian_inv)
                    hessian_inv = torch.linalg.cholesky(hessian_inv, upper = True)
                    hessian_inv = hessian_inv.contiguous()

                    break

                except RuntimeError:

                    # If inverting failed, assume there were non-positive eigenvalues, so apply more damping to shift the
                    # eigenvalues in a positive direction.

                    attempts += 1
                    if attempts == 5:
                        raise ValueError("Hessian is not invertible")

            self.hessian_inv = hessian_inv
            self.hessian = None

    def reuse_h(self, other):

        with torch.inference_mode():

            self.hessian_inv = other.hessian_inv
            self.hessian = None
            self.perm = other.perm
            self.weights = self.weights[self.perm, :]


    def quantize(self, keep_qweight = False):

        with torch.inference_mode():

            weights = self.weights.clone()
            self.quant = torch.zeros_like(self.weights)

            if keep_qweight:
                self.qweight = torch.zeros_like(weights, dtype = torch.short)

            # Quantize groups

            # self.scale = torch.zeros((self.total_groups, self.columns), dtype = torch.float, device = weights.device)
            scale = []
            qscale = []
            qscale_max = []
            qgroups = []

            bits_idx = -1
            bits_idx_r = 1

            error = weights.clone()

            for group in range(self.total_groups):

                a = group * self.group_size
                b = min(a + self.group_size, self.rows)

                bits_idx_r -= 1
                if bits_idx_r == 0:
                    bits_idx += 1
                    bits_idx_r = self.bits_groups[bits_idx]
                    bits = self.bits[bits_idx]
                    quantizer = AdaptiveQuantizer(bits = bits, scale_bits = self.scale_bits)

                qgroups.append(bits)
                qgroups.append(0)

                quantizer.find_params(weights[a : b, :])
                scale.append(quantizer.scale)
                qscale.append(quantizer.qscale)
                qscale_max.append(quantizer.qscale_max)

                ext_c.quantize_range(self.quant,
                                     quantizer.scale,
                                     self.qweight if keep_qweight else none_tensor,
                                     quantizer.qzero,
                                     quantizer.maxq,
                                     self.hessian_inv,
                                     weights,
                                     error,
                                     a,
                                     b)

            # Create g_idx to store inverse activation order

            rows = [i // self.group_size for i in range(self.rows)]
            self.g_idx = torch.tensor(rows, dtype = torch.int32, device = self.device)

            self.invperm = torch.argsort(self.perm)
            self.g_idx = self.g_idx[self.invperm]

            # Store scales

            self.scale = torch.stack(scale, dim = 0)
            self.qscale = torch.stack(qscale, dim = 0)
            self.qscale_max = torch.tensor(qscale_max, dtype = torch.float16, device = self.device)
            self.qgroups = torch.tensor(qgroups, dtype = torch.short, device = self.device)


    def quant_error(self):

        with torch.inference_mode():

            q = self.quant[self.invperm, :]
            diff = torch.abs(q - self.layer.weight.data.T)
            mat_error_1 = (diff > 0.01).sum().item() / diff.numel()
            mat_error_5 = (diff > 0.05).sum().item() / diff.numel()
            mat_error_10 = (diff > 0.10).sum().item() / diff.numel()
            return mat_error_1, mat_error_5, mat_error_10


    def apply_quant(self):

        q = self.quant[self.invperm, :].T
        # q = self.quant.T
        self.layer.weight.data = q.reshape(self.layer.weight.shape).type_as(self.layer.weight.data)


    def apply_temp(self):

        q = self.quant[self.invperm, :].T
        temp_layer = nn.Linear(self.layer.in_features, self.layer.out_features, False, device = "meta", dtype = torch.float16)
        temp_layer.weight = nn.Parameter(q.reshape(self.layer.weight.shape).type_as(self.layer.weight.data))
        return temp_layer


    def pack(self, key, qparams):

        assert qparams.scale_bits in [4]
        # assert self.columns % 32 == 0

        output = {}
        output[key + ".q_invperm"] = self.invperm.to(torch.int)
        output[key + ".q_scale_max"] = self.qscale_max
        output[key + ".q_groups"] = self.qgroups

        columns = self.columns
        rem_rows = self.rows
        padding = -columns % 32

        if padding != 0:
            print(f" !! Note: Padding quantized tensor {key}")

        qst = F.pad(self.qscale, (0, padding)).contiguous()
        qwt = F.pad(self.qweight, (0, padding)).contiguous()

        qst_packed = torch.zeros((qst.shape[0], qst.shape[1] * qparams.scale_bits // 32), dtype = torch.int32, device = self.device)
        if qparams.scale_bits == 4: ext_c.pack_rows_4(qst, qst_packed)
        # if qparams.scale_bits == 6: ext_c.pack_rows_6(qst, qst_packed) # TODO:
        output[key + ".q_scale"] = qst_packed

        qwt_packed = []

        i = 0
        row = 0
        out_row = 0
        while i < self.qscale.shape[0]:

            bits = self.qgroups[i * 2].item()
            self.qgroups[i * 2 + 1] = out_row
            i += 1

            rows = min(self.group_size, rem_rows)
            wpqr = 32 / bits
            qrows = rows / wpqr
            assert i == self.qgroups.shape[-1] or qrows == int(qrows)
            qrows = math.ceil(qrows)

            g_qwt = qwt[row:row+rows, :].contiguous()
            g_qwt_packed = torch.zeros((qrows, columns + padding), dtype = torch.int32, device = self.device)

            if padding > 0: g_qwt[:, -padding:] = 2 ** (bits - 1)

            ext_c.pack_columns(g_qwt, g_qwt_packed, bits)
            qwt_packed.append(g_qwt_packed)

            row += rows
            out_row += qrows
            rem_rows -= rows

        qwt_packed = torch.cat(qwt_packed, dim = 0)
        output[key + ".q_weight"] = qwt_packed

        return output



