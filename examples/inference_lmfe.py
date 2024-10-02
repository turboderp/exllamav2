
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from exllamav2 import ExLlamaV2, ExLlamaV2Config, ExLlamaV2Cache, ExLlamaV2Tokenizer
from exllamav2.generator import ExLlamaV2DynamicGenerator
from exllamav2.generator.filters import ExLlamaV2PrefixFilter
from inference_lmfe_wrapper import ExLlamaV2TokenEnforcerFilter
from lmformatenforcer import JsonSchemaParser
from pydantic import BaseModel, conlist
from typing import Literal
import json

model_dir = "/mnt/str/models/mistral-7b-exl2/4.0bpw"
config = ExLlamaV2Config(model_dir)
config.arch_compat_overrides()
model = ExLlamaV2(config)
cache = ExLlamaV2Cache(model, max_seq_len = 32768, lazy = True)
model.load_autosplit(cache, progress = True)

print("Loading tokenizer...")
tokenizer = ExLlamaV2Tokenizer(config)

# Initialize the generator with all default parameters

generator = ExLlamaV2DynamicGenerator(
    model = model,
    cache = cache,
    tokenizer = tokenizer,
)

# JSON schema

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

# Create prompts with and without filters

i_prompts = [
    "Here is some information about Superman:\n\n",
    "Here is some information about Batman:\n\n",
    "Here is some information about Aquaman:\n\n",
]

prompts = []
filters = []

for p in i_prompts:
    prompts.append(p)
    filters.append(None)
    prompts.append(p)
    filters.append([
        ExLlamaV2TokenEnforcerFilter(model, tokenizer, schema_parser),
        ExLlamaV2PrefixFilter(model, tokenizer, ["{", " {"])
    ])

# Generate

print("Generating...")

outputs = generator.generate(
    prompt = prompts,
    filters = filters,
    filter_prefer_eos = True,
    max_new_tokens = 300,
    add_bos = True,
    stop_conditions = [tokenizer.eos_token_id],
    completion_only = True
)

# Print outputs:

for i in range(len(i_prompts)):

    print("---------------------------------------------------------------------------------")
    print(i_prompts[i].strip())
    print()
    print("Without filter:")
    print("---------------")
    print(outputs[i * 2])
    print()
    print("With filter:")
    print("------------")
    print(json.dumps(json.loads(outputs[i * 2 + 1]), indent = 4).strip())
    print()