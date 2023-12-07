"""
builds a deep-hole-centered D4 codebook
this is a codebook consisting of points on the lattice in R4
    where each component is a half-integer
    and the components sum to an even number
from this lattice, we select the points that have a norm-squared of at most 9
this results in a codebook of 256 points distributed as follows
    8 with sorted abs of [1/2, 1/2, 1/2, 1/2]
    8                    [3/2, 3/2, 3/2, 3/2]
    4c2 * 8 = 48         [1/2, 1/2. 3/2, 3/2]
    4 * 8 = 32           [1/2, 1/2, 1/2, 3/2]
    4 * 8 = 32           [1/2, 3/2, 3/2, 3/2]
    4 * 8 = 32           [1/2, 1/2, 1/2, 5/2]
    4 * 3 * 8 = 96       [1/2, 1/2, 3/2, 5/2]
"""

import torch
from exllamav2.quip.matmul_had import matmul_hadU_cuda, matmul_hadUt_cuda
from exllamav2.ext import exllamav2_ext as ext_c

_D4_CODESZ = 4


def code3_signs(i3, x):
    if (i3 & (1 << 5)):
        x[2] *= -1
    if (i3 & (1 << 6)):
        x[1] *= -1
    if (sum(x) % 2 != 0):
        x[3] *= -1
    if (i3 & (1 << 7)):
        for j in range(_D4_CODESZ):
            x[j] *= -1
    assert (sum(x) % 2 == 0)
    return x


def code8_to_d4(i8):
    assert ((i8 >= 0) and (i8 < 256))
    i3 = i8 & (7 << 5)
    i8 = i8 & 31
    if i8 < 16:
        if i8 < 8:
            if i8 < 2:
                if i8 < 1:
                    return code3_signs(i3, [0.5] * _D4_CODESZ)
                else:
                    return code3_signs(i3, [1.5] * _D4_CODESZ)
            else:
                ibx = i8 >> 1
                if i8 & 1:
                    x = [0.5] * _D4_CODESZ
                    x[0] = 1.5
                    x[ibx] = 1.5
                else:
                    x = [1.5] * _D4_CODESZ
                    x[0] = 0.5
                    x[ibx] = 0.5
                return code3_signs(i3, x)
        else:
            ibx = (i8 & 3)
            if i8 < 8 + 4:
                x = [0.5] * _D4_CODESZ
                x[ibx] = 1.5
            else:
                x = [1.5] * _D4_CODESZ
                x[ibx] = 0.5
            return code3_signs(i3, x)
    else:
        if i8 < 16 + 4:
            ibx = (i8 & 3)
            x = [0.5] * _D4_CODESZ
            x[ibx] = 2.5
            return code3_signs(i3, x)
        else:
            ibx = i8 - 20
            ib4 = ibx & 3
            ib3 = ibx >> 2
            x = [0.5] * _D4_CODESZ
            x[ib4] = 1.5
            if (ib3 >= ib4):
                ib3 += 1
            x[ib3] = 2.5
            return code3_signs(i3, x)


def build_D4_CB():
    CB = torch.zeros(256, _D4_CODESZ)
    for i in range(256):
        x = code8_to_d4(i)
        for j in range(_D4_CODESZ):
            CB[i, j] = x[j]
    return CB


class QuantizedD4Linear:
    device: torch.device

    def __init__(self, device):
        self.device = device
        self.D4_CB = build_D4_CB().to(torch.float16).to(self.device)

    def forward(self,
                input,
                Qidxs,
                SU,
                SV,
                Wscale,
                had_left,
                had_right,
                K_left,
                K_right,
                A=None,
                B=None,
                rescale_WH=False,
                scaleWH=None,
                packed=False):
        (m, n) = Qidxs.shape

        x = input.view(-1, _D4_CODESZ * n).to(torch.float32)
        if rescale_WH:
            x /= scaleWH
        x = matmul_hadUt_cuda(x * SU, had_left, K_left)

        if A is not None and B is not None:
            Bx = x @ B.t().to(torch.float32)
            ABx = Bx @ A.t().to(torch.float32)

        x = (x / 1024).to(torch.float16)

        if (x.shape[0] <= 8):
            if (x.shape[0] == 8):
                x_padded = x.contiguous()
            else:
                x_padded = torch.zeros(8, n * _D4_CODESZ, dtype=torch.float16, device=self.device)
                x_padded[0:(x.shape[0]), :] = x
            z = torch.zeros(8, m, dtype=x.dtype, device=self.device)
            ext_c.lookupmatmul_d4_k8(x_padded, Qidxs, self.D4_CB, z)
            z = z[0:(x.shape[0]), :]
        elif (x.shape[0] <= 16):
            if (x.shape[0] == 16):
                x_padded = x.contiguous()
            else:
                x_padded = torch.zeros(16, n * _D4_CODESZ, dtype=torch.float16, device=self.device)
                x_padded[0:(x.shape[0]), :] = x
            z = torch.zeros(16, m, dtype=x.dtype, device=self.device)
            ext_c.lookupmatmul_d4_k16(x_padded, Qidxs, self.D4_CB, z)
            z = z[0:(x.shape[0]), :]
        elif (x.shape[0] <= 32):
            if (x.shape[0] == 32):
                x_padded = x.contiguous()
            else:
                x_padded = torch.zeros(32, n * _D4_CODESZ, dtype=torch.float16, device=self.device)
                x_padded[0:(x.shape[0]), :] = x
            z = torch.zeros(32, m, dtype=x.dtype, device=self.device)
            ext_c.lookupmatmul_d4_k32(x_padded, Qidxs, self.D4_CB, z)
            z = z[0:(x.shape[0]), :]
        else:
            # manifest the matrix
            W_decompressed = torch.zeros(m, n * _D4_CODESZ, dtype=torch.float16, device=self.device)
            ext_c.decompress_d4(Qidxs, self.D4_CB, W_decompressed)
            z = x @ W_decompressed.t()

        x = z.to(torch.float32) * (Wscale * 1024)
        if A is not None and B is not None:
            x = x + ABx.to(torch.float32)

        return (matmul_hadU_cuda(x, had_right, K_right) * SV).view(*input.shape[:-1], m)
