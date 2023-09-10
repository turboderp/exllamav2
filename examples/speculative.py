
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from exllamav2 import(
    ExLlamaV2,
    ExLlamaV2Config,
    ExLlamaV2Cache,
    ExLlamaV2Tokenizer,
)

from exllamav2.generator import (
    ExLlamaV2SpeculativeGenerator,
    ExLlamaV2Sampler
)

import time

# Initialize model and draft model

model_directory = "/mnt/str/models/_exl2/codellama-34b-instruct-4.0bpw-h6-exl2/"
draft_directory = "/mnt/str/models/_exl2/tinyllama-1b-4.0bpw-h6-exl2"

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
model.load([16, 24])

model_cache = ExLlamaV2Cache(model)
draft_cache = ExLlamaV2Cache(draft)

tokenizer = ExLlamaV2Tokenizer(model_config)

# Initialize generator

generator = ExLlamaV2SpeculativeGenerator(model, model_cache, draft, draft_cache, tokenizer)

# Generate some text

prompt = "The world can be a scary place sometimes"

settings = ExLlamaV2Sampler.Settings()
settings.temperature = 0.85
settings.top_k = 50
settings.top_p = 0.65
settings.token_repetition_penalty = 1.15

max_new_tokens = 200

generator.warmup()
time_begin = time.time()

output = generator.generate_simple(prompt, settings, max_new_tokens, seed = 31337)

time_end = time.time()
time_total = time_end - time_begin

print(output)
print()
print(f"Response generated in {time_total:.2f} seconds, {max_new_tokens} tokens, {max_new_tokens / time_total:.2f} tokens/second")
print()
print("Prediction attempts:", generator.attempts)
print("Prediction hits:", generator.hits)