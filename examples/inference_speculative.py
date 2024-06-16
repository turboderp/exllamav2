
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from exllamav2 import ExLlamaV2, ExLlamaV2Config, ExLlamaV2Cache, ExLlamaV2Tokenizer, Timer
from exllamav2.generator import ExLlamaV2DynamicGenerator, ExLlamaV2Sampler
from util import format_prompt, get_stop_conditions

# Load model and draft model

total_cache_tokens = 16384

draft_model_dir = "/mnt/str/models/qwen2-1.5b-instruct-exl2/4.0bpw"
draft_config = ExLlamaV2Config(draft_model_dir)
draft_model = ExLlamaV2(draft_config)
draft_cache = ExLlamaV2Cache(draft_model, max_seq_len = total_cache_tokens, lazy = True)
draft_model.load_autosplit(draft_cache, progress = True)

model_dir = "/mnt/str/models/qwen2-72b-instruct-exl2/6.0bpw"
config = ExLlamaV2Config(model_dir)
model = ExLlamaV2(config)
cache = ExLlamaV2Cache(model, max_seq_len = total_cache_tokens, lazy = True)
model.load_autosplit(cache, progress = True)

print("Loading tokenizer...")
tokenizer = ExLlamaV2Tokenizer(config)

# Create prompt. Don't use stop condition so we can measure speed over a set number of output tokens

prompt_format = "chatml"
prompt = format_prompt(
    prompt_format,
    "You are an AI coding model",
    "Implement QuickSort in Java, C# and Rust."
    # "You are an AI writing assistant",
    # "Write a short story about the Scottish town of Auchtermuchty."
)
max_new_tokens = 250
gen_settings = ExLlamaV2Sampler.Settings.greedy()

# Initialize generator without draft model, warm up to make sure we get correct timing results

print("-----------------------------------------------------------------------------------")
print("- No draft model")
print("-----------------------------------------------------------------------------------")

generator = ExLlamaV2DynamicGenerator(
    model = model,
    cache = cache,
    tokenizer = tokenizer,
)
generator.warmup()

with Timer() as t_no_draft:
    output = generator.generate(
        prompt = prompt,
        max_new_tokens = max_new_tokens,
        encode_special_tokens = True,
        gen_settings = gen_settings
    )

print(output)
print()

# Initialize and warm up generator with draft

print("-----------------------------------------------------------------------------------")
print("- With draft model")
print("-----------------------------------------------------------------------------------")

generator = ExLlamaV2DynamicGenerator(
    model = model,
    cache = cache,
    draft_model = draft_model,
    draft_cache = draft_cache,
    tokenizer = tokenizer,
    num_draft_tokens = 4,
)
generator.warmup()

with Timer() as t_draft:
    output = generator.generate(
        prompt = prompt,
        max_new_tokens = max_new_tokens,
        encode_special_tokens = True,
        gen_settings = gen_settings
    )

print(output)
print()

print("-----------------------------------------------------------------------------------")
print(f"speed, -SD: {max_new_tokens / t_no_draft.interval:.2f} tokens/second")
print(f"speed, +SD: {max_new_tokens / t_draft.interval:.2f} tokens/second")
