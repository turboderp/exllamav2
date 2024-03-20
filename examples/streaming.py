
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from exllamav2 import (
    ExLlamaV2,
    ExLlamaV2Config,
    ExLlamaV2Cache,
    ExLlamaV2Tokenizer,
)

from exllamav2.generator import (
    ExLlamaV2StreamingGenerator,
    ExLlamaV2Sampler
)

import time
import torch

# Initialize model and cache

# model_directory = "/mnt/str/models/mistral-7b-instruct-exl2/4.0bpw/"
# model_directory = "/mnt/str/models/mistral-7b-instruct"
# model_directory = "/mnt/str/models/starcoder2-7b"
model_directory = "/mnt/str/models/command-r-exl2/6.0bpw"
# model_directory = "/mnt/str/models/command-r"

torch.set_printoptions(precision = 5, sci_mode = False)

config = ExLlamaV2Config()
config.model_dir = model_directory
config.prepare()
# config.load_in_q4 = True
config.max_seq_len = 300
config.no_flash_attn = True

model = ExLlamaV2(config)
print("Loading model: " + model_directory)

cache = ExLlamaV2Cache(model, lazy = True)
model.load_autosplit(cache)

tokenizer = ExLlamaV2Tokenizer(config)

# Initialize generator

generator = ExLlamaV2StreamingGenerator(model, cache, tokenizer)

# Settings

settings = ExLlamaV2Sampler.Settings()
settings.temperature = 1.0
settings.top_k = 0
settings.top_p = 0.8
settings.top_a = 0.0
settings.token_repetition_penalty = 1.02
settings.disallow_tokens(tokenizer, [tokenizer.eos_token_id])

max_new_tokens = 250

# Prompt

prompt = "Once upon a time,"

input_ids = tokenizer.encode(prompt, add_bos = True)
prompt_tokens = input_ids.shape[-1]

# Make sure CUDA is initialized so we can measure performance

# generator.warmup()

# Send prompt to generator to begin stream

time_begin_prompt = time.time()

print (prompt, end = "")
sys.stdout.flush()

generator.set_stop_conditions([])
generator.begin_stream_ex(input_ids, settings)

# Streaming loop. Note that repeated calls to sys.stdout.flush() adds some latency, but some
# consoles won't update partial lines without it.

time_begin_stream = time.time()
generated_tokens = 0

while True:
    res = generator.stream_ex()
    chunk = res["chunk"]
    eos = res["eos"]

    generated_tokens += 1
    print (chunk, end = "")
    sys.stdout.flush()
    if eos or generated_tokens == max_new_tokens: break

time_end = time.time()

time_prompt = time_begin_stream - time_begin_prompt
time_tokens = time_end - time_begin_stream

print()
print()
print(f"Prompt processed in {time_prompt:.2f} seconds, {prompt_tokens} tokens, {prompt_tokens / time_prompt:.2f} tokens/second")
print(f"Response generated in {time_tokens:.2f} seconds, {generated_tokens} tokens, {generated_tokens / time_tokens:.2f} tokens/second")
