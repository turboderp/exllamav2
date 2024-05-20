
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

# Input prompts

batch_size = 5

prompts = \
[
    "How do I open a can of beans?",
    "How do I open a can of soup?",
    "How do I open a can of strawberry jam?",
    "How do I open a can of raspberry jam?",
    "What's the tallest building in Paris?",
    "What's the most populous nation on Earth?",
    "What's the most populous nation on Mars?",
    "What do the Mole People actually want and how can we best appease them?",
    "Why is the sky blue?",
    "Where is Waldo?",
    "Who is Waldo?",
    "Why is Waldo?",
    "Is it legal to base jump off the Eiffel Tower?",
    "Is it legal to base jump into a volcano?",
    "Why are cats better than dogs?",
    "Why is the Hulk so angry all the time?",
    "How do I build a time machine?",
    "Is it legal to grow your own catnip?"
]

# Sort by length to minimize padding

s_prompts = sorted(prompts, key = len)

# Apply prompt format

def format_prompt(sp, p):
    return f"[INST] <<SYS>>\n{sp}\n<</SYS>>\n\n{p} [/INST]"

system_prompt = "Answer the question to the best of your ability."
f_prompts = [format_prompt(system_prompt, p) for p in s_prompts]

# Split into batches

batches = [f_prompts[i:i + batch_size] for i in range(0, len(prompts), batch_size)]

# Initialize model and cache

model_directory =  "/mnt/str/models/mistral-7b-instruct-exl2/4.0bpw/"

config = ExLlamaV2Config(model_directory)
config.max_output_len = 1  # We're only generating one token at a time, so no need to allocate VRAM for bsz*max_seq_len*vocab_size logits

config.max_batch_size = batch_size  # Model instance needs to allocate temp buffers to fit the max batch size

model = ExLlamaV2(config)
print("Loading model: " + model_directory)

cache = ExLlamaV2Cache(model, lazy = True, batch_size = batch_size)  # Cache needs to accommodate the batch size
model.load_autosplit(cache)

tokenizer = ExLlamaV2Tokenizer(config)

# Initialize generator

generator = ExLlamaV2BaseGenerator(model, cache, tokenizer)

# Sampling settings

settings = ExLlamaV2Sampler.Settings()
settings.temperature = 0.85
settings.top_k = 50
settings.top_p = 0.8
settings.token_repetition_penalty = 1.01

max_new_tokens = 512

# generator.warmup()  # Only needed to fully initialize CUDA, for correct benchmarking

# Generate for each batch

collected_outputs = []
for b, batch in enumerate(batches):

    print(f"Batch {b + 1} of {len(batches)}...")

    outputs = generator.generate_simple(batch, settings, max_new_tokens, seed = 1234, add_bos = True)

    trimmed_outputs = [o[len(p):] for p, o in zip(batch, outputs)]
    collected_outputs += trimmed_outputs

# Print the results

for q, a in zip(s_prompts, collected_outputs):
    print("---------------------------------------")
    print("Q: " + q)
    print("A: " + a.strip())

# print(f"Response generated in {time_total:.2f} seconds, {max_new_tokens} tokens, {max_new_tokens / time_total:.2f} tokens/second")
