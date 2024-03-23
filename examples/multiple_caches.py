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

import torch
import time
import random

# Initialize model

model_directory =  "/mnt/str/models/llama2-7b-exl2/4.0bpw/"
print("Loading model: " + model_directory)

config = ExLlamaV2Config(model_directory)
model = ExLlamaV2(config)
model.load()
tokenizer = ExLlamaV2Tokenizer(config)

# Cache mode

cache_8bit = False

# Create some sampling settings

settings_proto = ExLlamaV2Sampler.Settings()
settings_proto.temperature = 0.8
settings_proto.top_p = 0.75
# settings_proto.mirostat = True
# settings_proto.mirostat_tau = 5
# settings_proto.top_k = 1000

# Define some prompts to inference in parallel

prompts = ["Once you eliminate all the",
           "C++ is",
           "Once upon a time, I had the pleasure of meeting Toni Morrison. I was attending an event at the University of North Carolina Chapel Hill when she came to speak. She",
           "A bird in the hand is worth two in the bush, but",
           "Too many cooks spoil the",
           "A lynx is a type of",
           "Standing before the gates of"]

max_parallel_seqs = 3

# Active sequences and corresponding caches and settings

input_ids = []
caches = []
settings = []

# Stats

total_gen_tokens = 0
total_prompt_tokens = 0
prompt_time = 0
token_time = 0

# Continue generating as long as there is work to do

while len(prompts) or len(input_ids):

    # If doing less than max_parallel_seqs, start some more. Prompt processing isn't batched in this example, but
    # would benefit much less from batching anyway

    while len(input_ids) < max_parallel_seqs and len(prompts):

        time_begin = time.time()

        prompt = prompts.pop()
        ids = tokenizer.encode(prompt)
        if cache_8bit:
            cache = ExLlamaV2Cache_8bit(model, max_seq_len = 256)  # (max_seq_len could be different for each cache)
        else:
            cache = ExLlamaV2Cache(model, max_seq_len = 256)  # (max_seq_len could be different for each cache)

        model.forward(ids[:, :-1], cache, preprocess_only = True)
        input_ids.append(ids)
        caches.append(cache)
        settings.append(settings_proto.clone())  # Need individual settings per prompt to support Mirostat

        total_prompt_tokens += ids.shape[-1] -1
        prompt_time += time.time() - time_begin

    # Create a batch tensor of the last token in each active sequence, forward through the model using the list of
    # active caches rather than a single, batched cache. Then sample for each token indidividually with some
    # arbitrary stop condition

    time_begin = time.time()

    inputs = torch.cat([x[:, -1:] for x in input_ids], dim = 0)
    logits = model.forward(inputs, caches, input_mask = None).float().cpu()

    eos = []
    r = random.random()
    for i in range(len(input_ids)):

        token, _, _, _, _ = ExLlamaV2Sampler.sample(logits[i:i+1, :, :], settings[i], input_ids[i], r, tokenizer)
        input_ids[i] = torch.cat([input_ids[i], token], dim = 1)
        total_gen_tokens += 1

        if token.item() == tokenizer.newline_token_id or caches[i].current_seq_len == caches[i].max_seq_len:
            eos.insert(0, i)

    token_time += time.time() - time_begin

    # Output and drop any sequences completed in this step

    for i in eos:

        output = tokenizer.decode(input_ids[i])[0]
        print("-----")
        print(output.strip())

        input_ids.pop(i)
        caches.pop(i)
        settings.pop(i)

# Stats

print("-----")
print(f"Prompts: {total_prompt_tokens} tokens, {total_prompt_tokens / prompt_time:.2f} tokens/second")
print(f"Tokens: {total_gen_tokens} tokens, {total_gen_tokens / token_time:.2f} tokens/second")

