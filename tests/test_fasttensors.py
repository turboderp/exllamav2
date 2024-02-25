import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from exllamav2.fasttensors import STFile
from exllamav2.ext import exllamav2_ext as ext_c

import time

# Single tensor

# stfile = "/mnt/str/models/llama2-70b-exl2/3.5bpw/output-00002-of-00004.safetensors"
# stfile_size = os.path.getsize(stfile)
# sttest = STFile(stfile)
# key = "model.layers.45.self_attn.o_proj.q_weight"
# print(key)
# a = sttest.get_tensor(key, device="cuda:0")
# b = sttest.get_tensor(key, device="cuda:0", not_fast = True)
# assert a.equal(b), ":<"


# Multi file

stfiles = \
[
    "/mnt/str/models/llama2-70b-exl2/4.0bpw/output-00001-of-00005.safetensors",
    "/mnt/str/models/llama2-70b-exl2/4.0bpw/output-00002-of-00005.safetensors",
    "/mnt/str/models/llama2-70b-exl2/4.0bpw/output-00003-of-00005.safetensors",
    "/mnt/str/models/llama2-70b-exl2/4.0bpw/output-00004-of-00005.safetensors",
    "/mnt/str/models/llama2-70b-exl2/4.0bpw/output-00005-of-00005.safetensors"
]

for stfile in stfiles:
    stfile_size = os.path.getsize(stfile)
    sttest = STFile(stfile)

    # List tensors

    # for k in sttest.get_dict().keys():
    #     print(k)

    # Test

    tensors1 = {}
    tensors2 = {}

    t = time.time()
    ext_c.safetensors_pinned_buffer()
    t = time.time() - t
    print(f"Time: {t:.4f} s")

    bleh = sttest.get_dict()
    keys = sttest.get_dict().keys()
    keys = sorted(keys, key = lambda d: bleh[d]["data_offsets"][0])

    t = time.time()
    for k in keys:
        tensor = sttest.get_tensor(k, device = "cuda:0")
        tensors1[k] = tensor
    t = time.time() - t
    print(f"Time: {t:.4f} s, {stfile_size / t / 1024**3:.4f} GB/s")

    t = time.time()
    for k in keys:
        tensor = sttest.get_tensor(k, device = "cuda:0", not_fast = True)
        tensors2[k] = tensor
    t = time.time() - t
    print(f"Time: {t:.4f} s, {stfile_size / t / 1024**3:.4f} GB/s")

    for k in sttest.get_dict().keys():
        a = tensors1[k]
        b = tensors2[k]
        assert a.equal(b), k

    print("ok")

xxx = 0