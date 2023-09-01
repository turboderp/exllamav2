import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from exllamav2 import(
    ExLlamaV2,
    ExLlamaV2Config,
    ExLlamaV2Cache,
    ExLlamaV2Tokenizer,
)

from exllamav2.generator import (
    ExLlamaV2BaseGenerator,
    ExLlamaV2Sampler
)

import time
import torch


# The allocation to test

# model_directory =  "/mnt/str/models/_exl2/openllama-3b-3.0bpw-h6-exl2/"
# model_directory =  "/mnt/str/models/_exl2/llama-7b-3.0bpw-h6-exl2/"
model_directory =  "/mnt/str/models/_exl2/llama2-70b-chat-4.0bpw-h6-exl2/"

allocation = [16, 24]


# Prime CUDA and initialize mem measurement

torch_devices = [f"cuda:{i}" for i in range(torch.cuda.device_count())]

torch.cuda.init()
temp = [torch.randn((1024, 1024), dtype = torch.float, device = x) for x in torch_devices]
temp2 = [x * 2 for x in temp]
temp = []
temp2 = []
torch.cuda.empty_cache()

mem_base = {}
for dev in torch_devices:
    mem_base[dev] = torch.cuda.max_memory_allocated(dev)


# Initialize and load model

config = ExLlamaV2Config()
config.model_dir = model_directory
config.prepare()

model = ExLlamaV2(config)
print("Loading model: " + model_directory)

_, stats = model.load(allocation, stats = True)
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats(dev)


# Load tokenizer

tokenizer = ExLlamaV2Tokenizer(config)


# Initialize and measure cache

cache = ExLlamaV2Cache(model)
cache_fp = cache.footprint()

expected = [(ab - rb) for (ab, rb) in zip(allocation, stats)]
expected_with_cache = [e for e in expected]
for idx, c in enumerate(cache_fp): expected_with_cache[idx] += c / 1024**3


# Initialize generator

generator = ExLlamaV2BaseGenerator(model, cache, tokenizer)


# Generate some text

settings = ExLlamaV2Sampler.Settings()
settings.temperature = 0.85
settings.top_k = 50
settings.top_p = 0.8
settings.token_repetition_penalty = 1.15
settings.disallow_tokens(tokenizer, [tokenizer.eos_token_id])

prompt = "Our story begins in the Scottish town of Auchtermuchty, where once"

max_new_tokens = 150

generator.warmup()
time_begin = time.time()

output = generator.generate_simple(prompt, settings, max_new_tokens, seed = 1234)

time_end = time.time()
time_total = time_end - time_begin

print(output)
print()
print(f"Response generated in {time_total:.2f} seconds, {max_new_tokens} tokens, {max_new_tokens / time_total:.2f} tokens/second")
print()


# Report

res1 = f" ** VRAM reported by Torch     : "
res2 = f" ** VRAM expected              : "
res3 = f" ** VRAM expected (with cache) : "
res4 = f" ** VRAM allocated (max)       : "
first = True

for idx, device in enumerate(torch_devices):
    mem_this = torch.cuda.max_memory_allocated(device) - mem_base[device]
    if not first: res1 += " - "
    if not first: res2 += " - "
    if not first: res3 += " - "
    if not first: res4 += " - "
    first = False
    res1 += f"[{device}] {mem_this / (1024 ** 2):,.2f} MB"
    res2 += f"[{device}] {expected[idx] * 1024:,.2f} MB"
    res3 += f"[{device}] {expected_with_cache[idx] * 1024:,.2f} MB"
    res4 += f"[{device}] {allocation[idx] * 1024:,.2f} MB"

print(res4)
print(res2)
print(res3)
print(res1)
