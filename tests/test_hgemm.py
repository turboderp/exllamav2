import torch
from exllamav2.ext import exllamav2_ext as ext_c, none_tensor
import random

shapes = [  # m, k, n
    [    1,   16,     1 ],
    [   16,   16,    16 ],
    [    8,  256,    32 ],
    [    8,    8,   256 ],
]

for i in range(10):
    shapes.append([random.randint(1, 200), random.randint(1, 16) * 32, random.randint(1, 10)])
    shapes.append([random.randint(1, 10), random.randint(1, 4) * 32, random.randint(1, 200)])

for s in shapes:
    m, k, n = s[0], s[1], s[2]

    print(f" ({m}, {k}) @ ({k}, {n}) -> ({m}, {n}): ".ljust(42), end = "")

    a = torch.randn((m, k), dtype = torch.half, device = "cuda:0")
    b = torch.randn((k, n), dtype = torch.half, device = "cuda:0")
    c = torch.empty((m, n), dtype = torch.half, device = "cuda:0")
    d = torch.empty((m, n), dtype = torch.half, device = "cuda:0")

    ext_c.gemm_half_half_half(a, b, c, 1, 0, False)
    ext_c.gemm_half_half_half(a, b, d, 1, 0, True)
    t = torch.matmul(a, b)

    e_cublas = d - c
    e_torch = t - c
    diff_cublas = torch.max(torch.abs(e_cublas)).item()
    diff_torch = torch.max(torch.abs(e_torch)).item()

    print(f"diff vs cuBLAS: {diff_cublas:.3f}   diff vs Torch: {diff_torch:.3f}")

