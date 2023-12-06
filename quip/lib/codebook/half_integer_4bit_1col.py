import torch
from torch import nn
import quiptools_cuda

from lib.utils.matmul_had import matmul_hadU_cuda, matmul_hadUt_cuda

_HI4B1C_CODESZ = 1


def get_grid():
    hintr = torch.arange(-8, 8) + 1 / 2
    return hintr.unsqueeze(-1)


_HI4B1C_CACHED = get_grid()
_HI4B1C_NORM_CACHED = torch.diag(_HI4B1C_CACHED @ _HI4B1C_CACHED.T)


class HI4B1C_codebook(nn.Module):

    def __init__(self, build_truncated=True):
        super(HI4B1C_codebook, self).__init__()
        self.opt_scale = 2.97
        self.codesz = _HI4B1C_CODESZ
        self.idx_dtype = torch.uint8

        grid = _HI4B1C_CACHED
        self.register_buffer('grid', grid)
        self.register_buffer('grid_norm', _HI4B1C_NORM_CACHED)
        '''
        self.cuda()
        samples = torch.distributions.multivariate_normal.MultivariateNormal(torch.zeros(1), torch.eye(1)).rsample([200000]).cuda()
        print(samples.shape)
        def fn_s(s):
            err = (self.quantize(samples*s, False)/s - samples).float().norm()**2
            err = err.cpu() / torch.numel(samples)
            return err.cpu()        
        import scipy
        print(scipy.optimize.minimize_scalar(fn_s, bounds=(0.1, 100)))
        exit()
        '''

    def round(self, X, grid, grid_norm):
        assert X.shape[-1] == self.codesz
        Xqidx = (2 * X @ grid.T - grid_norm).argmax(-1)
        return grid[Xqidx], Xqidx

    def quantize(self, X, return_idx=True):
        vals, idx = self.round(X, self.grid, self.grid_norm)
        if not return_idx:
            return vals
        return vals, idx.to(self.idx_dtype)

    def by_idxs(self, idxs):
        return self.grid[idxs.int()]


class QuantizedHI4B1CLinear(nn.Module):

    def __init__(self, device):
        super().__init__()
        self.codebook = HI4B1C_codebook(build_truncated=False).to(device)
        self.codebook.grid = self.codebook.grid.to(torch.float16)

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
                rank=-1,
                A=None,
                B=None,
                rescale_WH=False,
                scaleWH=None):
        (m, n) = Qidxs.shape

        x = input.view(-1, n * _HI4B1C_CODESZ).to(torch.float32)
        if rescale_WH:
            x /= scaleWH
        x = x * SU
        x = matmul_hadUt_cuda(x, had_left, K_left)

        if rank > 0:
            Bx = x @ B.t().to(torch.float32)
            ABx = Bx @ A.t().to(torch.float32)

        num_scale = 1024
        x = x / num_scale
        x = x.to(torch.float16)

        W_decompressed = self.codebook.by_idxs(Qidxs).reshape(-1, n * _HI4B1C_CODESZ)
        z = x @ W_decompressed.t()

        x = z.to(torch.float32)
        x = x * (Wscale * num_scale)

        if rank > 0:
            x = x + ABx.to(torch.float32)

        x = matmul_hadU_cuda(x, had_right, K_right)
        x = x * SV

        output = x.view(*input.shape[:-1], m)

        return output
