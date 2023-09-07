
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from exllamav2.model import ExLlamaV2, ExLlamaV2Config, ExLlamaV2Cache, ExLlamaV2Linear
from exllamav2.tokenizer import ExLlamaV2Tokenizer
import argparse, os, math, time
import pandas, fastparquet
import torch
import torch.nn.functional as F
from conversion.tokenize import get_tokens
from conversion.quantize import list_live_tensors
from exllamav2.ext import exllamav2_ext as ext_c, none_tensor

with torch.no_grad():

    # Full-precision model

    config_full = ExLlamaV2Config()
    # config_full.model_dir = "/mnt/str/models/llama-7b"
    config_full.model_dir = "/mnt/str/models/_exl2/llama2-7b"
    config_full.prepare()
    model_full = ExLlamaV2(config_full)
    model_full.load(lazy = True)

    tokenizer = ExLlamaV2Tokenizer(config_full)

    # Quantized model

    config_quant = ExLlamaV2Config()
    # config_quant.model_dir = "/mnt/str/models/_exl2/llama-7b-4.0bpw-h6-exl2/"
    config_quant.model_dir = "/mnt/str/models/_exl2/llama2-7b-5.0bpw-h6-exl2/"
    # config_quant.model_dir = "/mnt/str/models/llama-7b-4bit-128g/"
    # config_quant.model_dir = "/mnt/str/models/_test_models/TheBloke_WizardLM-30B-Uncensored-GPTQ/"
    config_quant.prepare()
    model_quant = ExLlamaV2(config_quant)
    model_quant.load(lazy = True)

    # Create input state for layer 0 (should also be normalized, really)

    embed = model_full.modules_dict["model.embed_tokens"]
    embed.load()
    test_ids = tokenizer.encode("Hello there!")
    test_state = embed.forward(test_ids).cuda()

    # Forward through full and quant layers

    linear_full = model_full.modules_dict["model.layers.0.mlp.gate_proj"]
    # linear_full = model_full.modules_dict["model.layers.0.self_attn.q_proj"]
    linear_full.load()
    test_state_full = linear_full.forward(test_state)

    linear_quant = model_quant.modules_dict["model.layers.0.mlp.gate_proj"]
    # linear_quant = model_quant.modules_dict["model.layers.0.self_attn.q_proj"]
    linear_quant.load()
    test_state_quant = linear_quant.forward(test_state, force_cuda = True)

    test_state_recons = linear_quant.forward(test_state, force_recons = True)

    # Measure differences in output state

    print()

    diff_max = torch.max((test_state_quant - test_state_full).abs())
    diff_mse = F.mse_loss(test_state_quant, test_state_full)
    print(f"Quant vs. original (max, mse): {diff_max.item():.8f}, {diff_mse.item():.8f}")
    diff_max = torch.max((test_state_recons - test_state_full).abs())
    diff_mse = F.mse_loss(test_state_recons, test_state_full)
    print(f"Recons vs. original (max, mse): {diff_max.item():.8f}, {diff_mse.item():.8f}")
    diff_max = torch.max((test_state_quant - test_state_recons).abs())
    diff_mse = F.mse_loss(test_state_quant, test_state_recons)
    print(f"Recons vs. quant (max, mse): {diff_max.item():.8f}, {diff_mse.item():.8f}")

    # Print some stuff

    print()
    print("full: ", test_state_full)
    print("quant: ", test_state_quant)
    print("dequant: ", test_state_recons)

    # Allocate some input states and initialize some more linear layers. Using multiple layers here for a more
    # realistic benchmark, since individual quantized layers can be small enough to fit entirely in the GPU's L2 cache

    for size_m in [1]: #, 2, 3, 4, 8, 16, 32, 64]:

        itr = 5000
        a_num = 113
        b_num = 27
        print()
        print(f"size_m: {size_m}, iter: {itr}")

        a = [torch.randn((size_m, linear_full.in_features), dtype=torch.half, device="cuda:0") for i in range(a_num)]

        linear_full_arr = []
        linear_quant_arr = []
        for i in range(b_num):
            linear_full = model_full.modules_dict[f"model.layers.{i}.self_attn.q_proj"]
            linear_full.load()
            linear_full_arr.append(linear_full)
            linear_quant = model_quant.modules_dict[f"model.layers.{i}.self_attn.q_proj"]
            linear_quant.load()
            linear_quant_arr.append(linear_quant)

        # Time some forward passes through original and quantized (with and without reconstruction)

        begin = time.time()
        torch.cuda.synchronize()
        for i in range(itr):
            c = linear_full_arr[i % b_num].forward(a[i % a_num])
        torch.cuda.synchronize()
        end = time.time()
        print(f"Torch time: {(end - begin) / itr * 1000:.4f} ms")

        begin = time.time()
        torch.cuda.synchronize()
        for i in range(itr):
            c = linear_quant_arr[i % b_num].forward(a[i % a_num], force_cuda = True)
        torch.cuda.synchronize()
        end = time.time()
        print(f"Quant time: {(end - begin) / itr * 1000:.4f} ms")

        begin = time.time()
        torch.cuda.synchronize()
        for i in range(itr):
            c = linear_quant_arr[i % b_num].forward(a[i % a_num], force_recons = True)
        torch.cuda.synchronize()
        end = time.time()
        print(f"Recons time: {(end - begin) / itr * 1000:.4f} ms")

    # Load all matrices in a full layer of the quant model

    target_layer = 4
    prefix = f"layers.{target_layer}."

    for k in model_quant.modules_dict.keys():
        if not prefix in k: continue
        module_quant = model_quant.modules_dict[k]
        module_quant.load()

    # Test that result of multiplication with identity and random matrix is the same with and without reconstruction

    print()

    for k in model_quant.modules_dict.keys():

        if not prefix in k and not "head" in k: continue

        module_quant = model_quant.modules_dict[k]
        module_quant.load()
        if isinstance(module_quant, ExLlamaV2Linear):

            gi = module_quant.dump_group_info()

            mat = torch.eye(module_quant.in_features, dtype = torch.half).cuda()
            test1 = module_quant.forward(mat, force_cuda = True)
            test2 = module_quant.forward(mat, force_recons = True)
            diff_i = torch.max((test1 - test2).abs())

            mat = torch.randn((module_quant.in_features, module_quant.in_features), dtype = torch.half).cuda()
            test1 = module_quant.forward(mat, force_cuda = True)
            test2 = module_quant.forward(mat, force_recons = True)
            diff_r = F.mse_loss(test1, test2)

            print (f"{k:40}  {gi:30}  ident: {diff_i.item():.6f}  u: {diff_r.item():.6f}")

    xx = 0
