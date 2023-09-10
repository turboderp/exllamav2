import sys, os, math
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
# model_directory =  "/mnt/str/models/_exl2/llama2-70b-chat-2.5bpw-h6-exl2/"
model_directory = "/mnt/str/models/_exl2/codellama-34b-instruct-4.0bpw-h6-exl2/"

allocation = [18, 24]


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
    torch.cuda.reset_peak_memory_stats(dev)
    mem_base[dev] = torch.cuda.max_memory_allocated(dev)


# Initialize and load model

config = ExLlamaV2Config()
config.model_dir = model_directory
config.prepare()
config.max_seq_len = 8192

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

print(f"Prompt processing, {model.config.max_seq_len - 1} tokens...")

cache.current_seq_len = 0
time_begin = time.time()
input_ids = torch.randint(0, model.config.vocab_size - 1, (1, model.config.max_seq_len - 1))
model.forward(input_ids, cache, preprocess_only = True)
torch.cuda.synchronize()
time_end = time.time()
time_total = time_end - time_begin

print(f"Prompt processed in {time_total:.2f} seconds, {(model.config.max_seq_len - 1) / time_total:.2f} tokens/second")
print()

# Report

res1 = f" ** VRAM reported by Torch     : "
res2 = f" ** VRAM expected              : "
res3 = f" ** VRAM expected (with cache) : "
res4 = f" ** VRAM allocated (max)       : "
res5 = f" ** Cache size                 : "
first = True

mem_total = 0
mem_exp = 0
for idx, device in enumerate(torch_devices):
    mem_this = torch.cuda.max_memory_allocated(device) - mem_base[device]
    mem_total += mem_this
    mem_exp += expected_with_cache[idx] * 1024 ** 3
    if not first: res1 += " - "
    if not first: res2 += " - "
    if not first: res3 += " - "
    if not first: res4 += " - "
    if not first: res5 += " - "
    first = False
    res1 += f"[{device}] {mem_this / (1024 ** 2):,.2f} MB"
    res2 += f"[{device}] {expected[idx] * 1024:,.2f} MB"
    res3 += f"[{device}] {expected_with_cache[idx] * 1024:,.2f} MB"
    res4 += f"[{device}] {allocation[idx] * 1024:,.2f} MB"
    res5 += f"[{device}] {cache_fp[idx] / (1024 ** 2) if idx < len(cache_fp) else 0:,.2f} MB"

print(res4)
print(res2)
print(res5)
print(res3)
print(res1)

print()
print(f"Max sequence length:  {config.max_seq_len}")
print(f"Hidden size:          {config.hidden_size}")
print(f"Attention heads:      {config.num_attention_heads}")
print(f"Key/value heads:      {config.num_key_value_heads}")
print(f"Max attention size:   {math.sqrt(config.max_attention_size)} ** 2")
print(f"Max input len:        {config.max_input_len}")
# print(f"Correction amount:    {mem_total - mem_exp:,.2f} B")




