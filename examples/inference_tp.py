
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from exllamav2 import ExLlamaV2, ExLlamaV2Config, ExLlamaV2Cache_TP, ExLlamaV2Tokenizer, Timer
from exllamav2.generator import ExLlamaV2DynamicGenerator, ExLlamaV2Sampler

model_dir = "/mnt/str/models/llama3.1-70b-instruct-exl2/6.0bpw"
config = ExLlamaV2Config(model_dir)
config.arch_compat_overrides()
config.no_graphs = True
model = ExLlamaV2(config)

# Load the model in tensor-parallel mode. With no gpu_split specified, the model will attempt to split across
# all visible devices according to the currently available VRAM on each. expect_cache_tokens is necessary for
# balancing the split, in case the GPUs are of uneven sizes, or if the number of GPUs doesn't divide the number
# of KV heads in the model
#
# The cache type for a TP model is always ExLlamaV2Cache_TP and should be allocated after the model. To use a
# quantized cache, add a `base = ExLlamaV2Cache_Q6` etc. argument to the cache constructor. It's advisable
# to also add `expect_cache_base = ExLlamaV2Cache_Q6` to load_tp() as well so the size can be correctly
# accounted for when splitting the model.

model.load_tp(progress = True, expect_cache_tokens = 16384)
cache = ExLlamaV2Cache_TP(model, max_seq_len = 16384)

# After loading the model, all other functions should work the same

print("Loading tokenizer...")
tokenizer = ExLlamaV2Tokenizer(config)

# Initialize the generator with all default parameters

generator = ExLlamaV2DynamicGenerator(
    model = model,
    cache = cache,
    tokenizer = tokenizer,
)

max_new_tokens = 200

# Warmup generator. The function runs a small completion job to allow all the kernels to fully initialize and
# autotune before we do any timing measurements. It can be a little slow for larger models and is not needed
# to produce correct output.

generator.warmup()

# Generate one completion, using default settings

prompt = "Our story begins in the Scottish town of"

with Timer() as t_single:
    output = generator.generate(
        prompt = prompt,
        max_new_tokens = max_new_tokens,
        add_bos = True,
    )

print("-----------------------------------------------------------------------------------")
print("- Single completion")
print("-----------------------------------------------------------------------------------")
print(output)
print()

# Do a batched generation

prompts = [
    "Once upon a time,",
    "The secret to success is",
    "There's no such thing as",
    "Here's why you should adopt a cat:",
]

with Timer() as t_batched:
    outputs = generator.generate(prompt = prompts, max_new_tokens = max_new_tokens, add_bos = True)

for idx, output in enumerate(outputs):
    print("-----------------------------------------------------------------------------------")
    print(f"- Batched completion #{idx + 1}")
    print("-----------------------------------------------------------------------------------")
    print(output)
    print()

print("-----------------------------------------------------------------------------------")
print(f"speed, bsz 1: {max_new_tokens / t_single.interval:.2f} tokens/second")
print(f"speed, bsz {len(prompts)}: {max_new_tokens * len(prompts) / t_batched.interval:.2f} tokens/second")
