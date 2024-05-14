
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from exllamav2 import *
from exllamav2.generator import *

# Initialize model and cache

model_directory = "/mnt/str/models/llama2-70b-chat-exl2/4.0bpw"

config = ExLlamaV2Config()
config.model_dir = model_directory
config.max_batch_size = 2
config.no_flash_attn = True
config.no_xformers = True
config.prepare()

model = ExLlamaV2(config)
print("Loading model: " + model_directory)

cache = ExLlamaV2Cache(model, lazy = True, batch_size = 2)
model.load_autosplit(cache)

tokenizer = ExLlamaV2Tokenizer(config)

# Initialize generator

generator = ExLlamaV2StreamingGenerator(model, cache, tokenizer)

# Settings

settings = ExLlamaV2Sampler.Settings()
settings.temperature = 0.85
settings.top_k = 50
settings.top_p = 0.8
settings.top_a = 0.0
settings.token_repetition_penalty = 1.05

max_new_tokens = 250

# Prompt

positive = \
"""[INST] <<SYS>>
You are a cheerful, bubbly and respectful assistant.
<</SYS>>
{prompt} [/INST]"""

negative = \
"""[INST] <<SYS>>
You are a rude and obnoxious assistant.
<</SYS>>
{prompt} [/INST]"""

q = """Tell me about Homer Simpson."""

prompt_a = positive.replace("{prompt}", q)
prompt_b = negative.replace("{prompt}", q)

print("-------------------------------------------")
print("Prompt a:\n" + prompt_a + "\n")

print("-------------------------------------------")
print("Prompt b:\n" + prompt_b + "\n")

for x in range(11):

    # cfg_scale is the weight of the first prompt in the batch, while the second prompt is weighted as (1 - cfg_scale).
    #
    # - at cfg_scale == 0, only the second prompt is effective
    # - at 0 < cfg_scale < 1, the sampled logits will be a weighted average of the normalized outputs of both prompts
    # - at cfg_scale == 1, only the first prompt is effective
    # - at cfg_scale > 1, the second prompt will have a negative weight, emphasizing the difference between the two

    settings.cfg_scale = x / 5

    # Start a batched generation. CFG requires a batch size of exactly 2. Offsets and padding mask are required

    input_ids, offsets = tokenizer.encode([prompt_a, prompt_b], encode_special_tokens = True, return_offsets = True)
    mask = tokenizer.padding_mask(input_ids)
    generator.begin_stream(input_ids, settings, input_mask = mask, position_offsets = offsets)
    generator.set_stop_conditions([tokenizer.eos_token_id])

    print(f"---------------------------------------------------------------------------------------")
    print(f"cfg_scale = {settings.cfg_scale:.1f}")
    print()

    generated_tokens = 0
    max_new_tokens = 200
    while True:
        chunk, eos, _ = generator.stream()
        generated_tokens += 1
        print (chunk, end = "")
        sys.stdout.flush()
        if eos or generated_tokens == max_new_tokens: break
    print()
