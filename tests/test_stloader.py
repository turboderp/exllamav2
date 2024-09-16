import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from exllamav2.stloader import STFile
import time

# Multi file

stfiles = \
[
    "/mnt/str/models/llama3-70b-exl2/4.0bpw/output-00001-of-00005.safetensors",
    "/mnt/str/models/llama3-70b-exl2/4.0bpw/output-00002-of-00005.safetensors",
    "/mnt/str/models/llama3-70b-exl2/4.0bpw/output-00003-of-00005.safetensors",
    "/mnt/str/models/llama3-70b-exl2/4.0bpw/output-00004-of-00005.safetensors",
    "/mnt/str/models/llama3-70b-exl2/4.0bpw/output-00005-of-00005.safetensors"
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
    t = time.time() - t
    print(f"Time: {t:.4f} s")

    bleh = sttest.get_dict()
    keys = sttest.get_dict().keys()
    keys = sorted(keys, key = lambda d: bleh[d]["data_offsets"][0])

    t = time.time()
    for k in keys:
        tensor = sttest.get_tensor(k, device = "cuda:0")
        tensors2[k] = tensor
    t = time.time() - t
    print(f"Time: {t:.4f} s, {stfile_size / t / 1024**3:.4f} GB/s")

    t = time.time()
    for k in keys:
        tensor = sttest.get_tensor(k, device = "cpu")
        tensors1[k] = tensor
    t = time.time() - t
    print(f"Time: {t:.4f} s, {stfile_size / t / 1024**3:.4f} GB/s")

    for k in sttest.get_dict().keys():
        a = tensors1[k]
        b = tensors2[k]
        assert a.cuda().equal(b), k

    print("ok")

xxx = 0