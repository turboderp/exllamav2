"""
D8^ = D8 + 1/2 intersected with ball of radius sqrt(10)
|D8^| has 227 entries
We then add 29 entries from the set of vectors with 5 3/2 and 3 1/2
The total codebook is all 2^7 flips of these 256 entries (2^15) +- 1/4
which makes 2^16 entries.
This corresponds to a subset of E8 + 1/4
"""

import torch
from exllamav2.quip.matmul_had import matmul_hadU_cuda, matmul_hadUt_cuda
from exllamav2.ext import exllamav2_ext as ext_c


_E8P_CODESZ = 8

def get_abs_grid():
        intr = torch.arange(-4, 4)
        d8 = torch.cartesian_prod(*[intr] * _E8P_CODESZ).float() + 1 / 2
        d8m2 = (d8.sum(dim=-1) % 2 == 0)
        d8n = d8.norm(dim=-1)**2 <= 10
        d8abs = torch.unique(d8[sorted(torch.where(d8m2 * d8n)[0])].abs(), dim=0)

        norm12 = torch.tensor([
            [3, 1, 1, 1, 3, 3, 3, 3],
            [1, 3, 1, 1, 3, 3, 3, 3],
            [1, 1, 3, 1, 3, 3, 3, 3],
            [1, 1, 1, 3, 3, 3, 3, 3],
            [3, 3, 3, 1, 3, 3, 1, 1],
            [3, 3, 3, 1, 3, 1, 3, 1],
            [3, 3, 3, 1, 1, 3, 3, 1],
            [3, 3, 3, 1, 3, 1, 1, 3],
            [3, 3, 3, 1, 1, 3, 1, 3],
            [3, 3, 3, 1, 1, 1, 3, 3],
            [3, 3, 1, 3, 3, 3, 1, 1],
            [3, 3, 1, 3, 3, 1, 3, 1],
            [3, 3, 1, 3, 1, 3, 3, 1],
            [3, 3, 1, 3, 3, 1, 1, 3],
            [3, 3, 1, 3, 1, 3, 1, 3],
            [3, 3, 1, 3, 1, 1, 3, 3],
            [3, 1, 3, 3, 3, 3, 1, 1],
            [3, 1, 3, 3, 3, 1, 3, 1],
            [3, 1, 3, 3, 1, 3, 3, 1],
            [3, 1, 3, 3, 3, 1, 1, 3],
            [3, 1, 3, 3, 1, 3, 1, 3],
            [1, 3, 3, 3, 1, 1, 3, 3],
            [1, 3, 3, 3, 3, 3, 1, 1],
            [1, 3, 3, 3, 3, 1, 3, 1],
            [1, 3, 3, 3, 1, 3, 3, 1],
            [1, 3, 3, 3, 3, 1, 1, 3],
            [1, 3, 3, 3, 1, 3, 1, 3],
            [1, 3, 3, 3, 1, 1, 3, 3],
            [3, 3, 1, 1, 3, 3, 3, 1],
        ]) / 2
        return torch.concat([d8abs, norm12], dim=0)

_E8P_ABS_CACHED = get_abs_grid()

class QuantizedE8P12Linear:
    _E8P_CODESZ = 8
    grid_abs: torch.Tensor
    grid_abs_even: torch.Tensor
    codebook_matvec: torch.Tensor
    device: torch.device

    def __init__(self, device):
        self.device = device
        self.grid_abs = _E8P_ABS_CACHED.to(self.device).half()
        self.grid_abs_even = (self.grid_abs.sum(dim=-1) % 2 == 0).to(self.device)
        self.codebook_matvec = torch.zeros((256,), dtype=torch.int64, device=self.device)
        for i in range(8):
            chunk = (self.grid_abs[:, i] * 4).to(torch.int64)
            self.codebook_matvec |= chunk << (i * 8)


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

        x = input.view(-1, n * _E8P_CODESZ).to(torch.float32)
        if rescale_WH:
            x /= scaleWH
        x = x * SU
        x = matmul_hadUt_cuda(x, had_left, K_left)

        if A is not None and B is not None:
            Bx = x @ B.t().to(torch.float32)
            ABx = Bx @ A.t().to(torch.float32)

        # TODO: find the optimal threshold
        if x.size(0) < 6:
            x = ext_c.decode_matmul_e8p(x, Qidxs - 0x8000, self.codebook_matvec).to(torch.float32)
        else:
            W_decompressed = torch.zeros(m, n*_E8P_CODESZ, device=Qidxs.device, dtype=torch.float16)
            ext_c.decompress_e8p_origorder(
                Qidxs, self.grid_abs, self.grid_abs_even, W_decompressed)
            x = (x.to(torch.float16) @ W_decompressed.T).to(torch.float32)

        x *= Wscale

        if A is not None and B is not None:
            x = x + ABx.to(torch.float32)

        x = matmul_hadU_cuda(x, had_right, K_right)
        x = x * SV

        output = x.view(*input.shape[:-1], m)
        return output
    
    

