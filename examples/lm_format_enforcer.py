
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pydantic import BaseModel, conlist
from typing import Literal
from lmformatenforcer.integrations.exllamav2 import ExLlamaV2TokenEnforcerFilter
from lmformatenforcer import JsonSchemaParser

from exllamav2 import (
    ExLlamaV2,
    ExLlamaV2Config,
    ExLlamaV2Cache,
    ExLlamaV2Tokenizer,
)

from exllamav2.generator import (
    ExLlamaV2StreamingGenerator,
    ExLlamaV2Sampler,
)

from exllamav2.generator.filters import (
    ExLlamaV2PrefixFilter
)

import time, json

# Initialize model and cache

model_directory = "/mnt/str/models/llama2-13b-exl2/4.0bpw/"
print("Loading model: " + model_directory)

config = ExLlamaV2Config(model_directory)
model = ExLlamaV2(config)
cache = ExLlamaV2Cache(model, lazy = True)
model.load_autosplit(cache)
tokenizer = ExLlamaV2Tokenizer(config)

# Initialize generator

generator = ExLlamaV2StreamingGenerator(model, cache, tokenizer)
generator.speculative_ngram = True
generator.warmup()  # for more accurate timing

# Generate with or without filter

def completion(prompt, filters = None, max_new_tokens = 200, eos_bias = False):

    settings = ExLlamaV2Sampler.Settings()
    settings.temperature = 0.75
    settings.top_k = 0
    settings.top_p = 0.5
    settings.token_repetition_penalty = 1.0

    settings.filters = filters

    # If using a filter, sample the EOS token as soon as filter allows it

    settings.filter_prefer_eos = eos_bias

    # Send prompt to generator to begin stream

    input_ids = tokenizer.encode(prompt)
    prompt_tokens = input_ids.shape[-1]

    time_begin_prompt = time.time()

    generator.set_stop_conditions([tokenizer.eos_token_id])
    generator.begin_stream_ex(input_ids, settings)

    # Streaming loop

    time_begin_stream = time.time()
    generated_tokens = 0

    print("--------------------------------------------------")
    print(prompt)
    print(" ------>" + (" (filtered)" if len(filters) > 0 else ""))

    result = ""
    while True:
        res = generator.stream_ex()
        result += res["chunk"]
        generated_tokens += 1
        print(res["chunk"], end = "")
        sys.stdout.flush()
        if res["eos"] or generated_tokens == max_new_tokens: break

    time_end = time.time()

    time_prompt = time_begin_stream - time_begin_prompt
    time_tokens = time_end - time_begin_stream

    print("\n")
    print(f"Prompt processed in {time_prompt:.2f} seconds, {prompt_tokens} tokens, {prompt_tokens / time_prompt:.2f} tokens/second")
    print(f"Response generated in {time_tokens:.2f} seconds, {generated_tokens} tokens, {generated_tokens / time_tokens:.2f} tokens/second")
    print("\n\n")

    return result

# Configure filter

class SuperheroAppearance(BaseModel):
    title: str
    issue_number: int
    year: int

class Superhero(BaseModel):
    name: str
    secret_identity: str
    superpowers: conlist(str, max_length = 5)
    first_appearance: SuperheroAppearance
    gender: Literal["male", "female"]

schema_parser = JsonSchemaParser(Superhero.schema())
lmfe_filter = ExLlamaV2TokenEnforcerFilter(schema_parser, tokenizer)
prefix_filter = ExLlamaV2PrefixFilter(model, tokenizer, "{")  # Make sure we start JSONing right away

# Run some tests

prompt = "Here is some information about Superman:\n"
completion(prompt, [])
result = completion(prompt, [lmfe_filter, prefix_filter], eos_bias = True)

j = json.loads(result)
print("Parsed JSON:" , j)

prompt = "Here is some information about Batman:\n"
completion(prompt, [])
result = completion(prompt, [lmfe_filter, prefix_filter], eos_bias = True)

j = json.loads(result)
print("Parsed JSON:" , j)
