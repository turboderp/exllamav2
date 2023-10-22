
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from exllamav2 import(
    ExLlamaV2,
    ExLlamaV2Config,
    ExLlamaV2Cache,
    ExLlamaV2Cache_8bit,
    ExLlamaV2Tokenizer,
)

from exllamav2.generator import (
    ExLlamaV2BaseGenerator,
    ExLlamaV2Sampler
)

import time

# Initialize model and cache

model_directory =  "/mnt/str/models/_exl2/llama2-70b-exl2/2.5bpw/"

config = ExLlamaV2Config()
config.model_dir = model_directory
config.prepare()
config.max_seq_len = 4096
config.max_input_len = 1024
config.max_attn_size = 1024**2

model = ExLlamaV2(config)
print("Loading model: " + model_directory)

def progress_rep(module, num_modules):
    yield f"Progress: {100 * module / num_modules:.2f}%"

cache = ExLlamaV2Cache_8bit(model, lazy = True)

f = model.load_autosplit_gen(cache, last_id_only = True, callback_gen = progress_rep)
for item in f:
    print(item)

tokenizer = ExLlamaV2Tokenizer(config)

# Initialize generator

generator = ExLlamaV2BaseGenerator(model, cache, tokenizer)

# Generate some text

settings = ExLlamaV2Sampler.Settings()
settings.temperature = 0.85
settings.top_k = 50
settings.top_p = 0.8
settings.token_repetition_penalty = 1.15
settings.disallow_tokens(tokenizer, [tokenizer.eos_token_id])

prompt = "All work and no play makes Jack a dull boy. " * 1000
prompt_ids = tokenizer.encode(prompt)
prompt_ids = prompt_ids[:, :2048 - 50 - 1]
prompt = tokenizer.decode(prompt_ids)

max_new_tokens = 50

generator.warmup()
time_begin = time.time()

output = generator.generate_simple(prompt, settings, max_new_tokens, seed = 1234)

time_end = time.time()
time_total = time_end - time_begin

print(output)
print()
print(f"Response generated in {time_total:.2f} seconds, {max_new_tokens} tokens, {max_new_tokens / time_total:.2f} tokens/second")
