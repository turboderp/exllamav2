import torch
from quip.matmul_had import matmul_hadU_cuda, matmul_hadUt_cuda
from exllamav2.ext import exllamav2_ext as ext_c

def get_grid():
    hintr = torch.arange(-8, 8) + 1 / 2
    return hintr.unsqueeze(-1)

class QuantizedHI4B1CLinear:
    device: torch.device
    grid: torch.Tensor
    packed: int
    packsz = 8

    def __init__(self, device):
        self.device = device
        self.grid = get_grid().to(torch.float16).to(self.device)

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
        n, m = len(SU), len(SV)

        x = input.view(-1, n).to(torch.float32)
        if rescale_WH:
            x /= scaleWH
        x = x * SU
        x = matmul_hadUt_cuda(x, had_left, K_left)

        if A is not None and B is not None:
            Bx = x @ B.t().to(torch.float32)
            ABx = Bx @ A.t().to(torch.float32)

        num_scale = 1024
        x = x / num_scale
        x = x.to(torch.float16)

        if packed:
            W_decompressed = torch.zeros(m, n, dtype=torch.float16, device=self.device)
            ext_c.decompress_hi4b1c_packed(Qidxs, self.grid, W_decompressed)
        else:
            W_decompressed = self.__by_idxs(Qidxs, packed=packed).reshape(-1, n)
        
        z = x @ W_decompressed.t()

        x = z.to(torch.float32)
        x = x * (Wscale * num_scale)

        if A is not None and B is not None:
            x = x + ABx.to(torch.float32)

        x = matmul_hadU_cuda(x, had_right, K_right)
        x = x * SV

        output = x.view(*input.shape[:-1], m)

        return output
    
    def __by_idxs(self, idxs, packed=False):
        if packed:
            idxs = idxs.repeat_interleave(self.packsz, dim=-1)
            idxs[:, 0::self.packsz] = (idxs[:, 0::self.packsz] >> 28) & 15
            idxs[:, 2::self.packsz] = (idxs[:, 2::self.packsz] >> 24) & 15
            idxs[:, 4::self.packsz] = (idxs[:, 4::self.packsz] >> 20) & 15
            idxs[:, 6::self.packsz] = (idxs[:, 6::self.packsz] >> 16) & 15
            idxs[:, 1::self.packsz] = (idxs[:, 1::self.packsz] >> 12) & 15
            idxs[:, 3::self.packsz] = (idxs[:, 3::self.packsz] >> 8) & 15
            idxs[:, 5::self.packsz] = (idxs[:, 5::self.packsz] >> 4) & 15
            idxs[:, 7::self.packsz] = idxs[:, 7::self.packsz] & 15

        return self.grid[idxs.int()]
