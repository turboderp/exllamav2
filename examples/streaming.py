
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

# Initialize model and cache

model_directory =  "/mnt/str/models/_exl2/llama2-70b-3.0bpw-h6-exl2/"

config = ExLlamaV2Config()
config.model_dir = model_directory
config.prepare()

model = ExLlamaV2(config)
print("Loading model: " + model_directory)
model.load([16, 24])

tokenizer = ExLlamaV2Tokenizer(config)

cache = ExLlamaV2Cache(model)

# Initialize generator

generator = ExLlamaV2StreamingGenerator(model, cache, tokenizer)

# Settings

settings = ExLlamaV2Sampler.Settings()
settings.temperature = 0.85
settings.top_k = 50
settings.top_p = 0.8
settings.token_repetition_penalty = 1.15
settings.disallow_tokens(tokenizer, [tokenizer.eos_token_id])

max_new_tokens = 250

# Prompt

prompt = "Our story begins in the Scottish town of Auchtermuchty, where once"

input_ids = tokenizer.encode(prompt)
prompt_tokens = input_ids.shape[-1]

# Make sure CUDA is initialized so we can measure performance

generator.warmup()

# Send prompt to generator to begin stream

time_begin_prompt = time.time()

print (prompt, end = "")
sys.stdout.flush()

generator.set_stop_conditions([])
generator.begin_stream(input_ids, settings)

# Streaming loop. Note that repeated calls to sys.stdout.flush() adds some latency, but some
# consoles won't update partial lines without it.

time_begin_stream = time.time()
generated_tokens = 0

while True:
    chunk, eos, _ = generator.stream()
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
