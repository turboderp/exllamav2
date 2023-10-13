
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

# Initialize model and draft model

model_directory = "/mnt/str/models/_exl2/codellama-34b-instruct-exl2/4.0bpw"
draft_directory = "/mnt/str/models/_exl2/tinyllama-1b-ckpt503-exl2/3.5bpw"

model_config = ExLlamaV2Config()
model_config.model_dir = model_directory
model_config.prepare()
model_config.max_seq_len = 2048

draft_config = ExLlamaV2Config()
draft_config.model_dir = draft_directory
draft_config.prepare()
draft_config.max_seq_len = 2048

draft = ExLlamaV2(draft_config)
draft.load([24, 0])

model = ExLlamaV2(model_config)
model.load([14, 24])

model_cache = ExLlamaV2Cache(model)
draft_cache = ExLlamaV2Cache(draft)

tokenizer = ExLlamaV2Tokenizer(model_config)

# Initialize generator

# generator = ExLlamaV2StreamingGenerator(model, model_cache, tokenizer)
generator = ExLlamaV2StreamingGenerator(model, model_cache, tokenizer, draft, draft_cache, 5)

# Settings

settings = ExLlamaV2Sampler.Settings()
settings.temperature = 0.6
settings.top_k = 50
settings.top_p = 0.6
settings.token_repetition_penalty = 1.0
settings.disallow_tokens(tokenizer, [tokenizer.eos_token_id])

max_new_tokens = 250

# Prompt

prompt = "Here is a simple Quicksort implementation in C++:"

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
    # sys.stdout.flush()
    if eos or generated_tokens == max_new_tokens: break

time_end = time.time()

time_prompt = time_begin_stream - time_begin_prompt
time_tokens = time_end - time_begin_stream

print()
print()
print(f"Prompt processed in {time_prompt:.2f} seconds, {prompt_tokens} tokens, {prompt_tokens / time_prompt:.2f} tokens/second")
print(f"Response generated in {time_tokens:.2f} seconds, {generated_tokens} tokens, {generated_tokens / time_tokens:.2f} tokens/second")
