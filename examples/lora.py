
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from exllamav2 import ExLlamaV2, ExLlamaV2Config, ExLlamaV2Cache, ExLlamaV2Tokenizer, ExLlamaV2Lora
from exllamav2.generator import ExLlamaV2Sampler, ExLlamaV2DynamicGenerator

model_dir = "/mnt/str/models/llama2-7b-exl2/5.0bpw"
config = ExLlamaV2Config(model_dir)
model = ExLlamaV2(config)
cache = ExLlamaV2Cache(model, max_seq_len = 32768, lazy = True)
model.load_autosplit(cache, progress = True)

print("Loading tokenizer...")
tokenizer = ExLlamaV2Tokenizer(config)

# Load LoRA. Using https://huggingface.co/nealchandra/llama-2-7b-hf-lora-alpaca-json for this example,
# since it very clearly shows the model adapting to an input/output format, as opposed to Alpaca-style
# formats which base models can usually adapt to without finetuning.

print("Loading LoRA...")
lora_dir = "/mnt/str/loras/llama2-7b-hf-lora-alpaca-json/"
lora = ExLlamaV2Lora.from_directory(model, lora_dir)

# Initialize the generator with all default parameters

generator = ExLlamaV2DynamicGenerator(
    model = model,
    cache = cache,
    tokenizer = tokenizer,
)

# Alpaca-style prompt

prompt_format = (
    """### INPUT:\n"""
    """```json\n"""
    """{"instructions": "<INSTRUCTIONS>", "input": "<INPUT>"}\n"""
    """```\n"""
    """\n"""
    """### OUTPUT:\n"""
)

inputs = (
    "Jim only understands analogies involving spaghetti and meatballs, and he really "
    "appreciates emojis."
)
instructions = (
    "Write a series of four tweets explaining that the Earth is not flat, tailored for Jim. "
    "Present them as a numbered list, and be sure to include at least one hashtag in each tweet."
)

prompt = prompt_format.replace("<INPUT>", inputs).replace("<INSTRUCTIONS>", instructions)

# Without LoRA

output = generator.generate(
    prompt = prompt,
    max_new_tokens = 500,
    add_bos = True,
    stop_conditions = [tokenizer.eos_token_id, "###"],
    gen_settings = ExLlamaV2Sampler.Settings.greedy()
)

print("-----------------------------------------------------------------------------------")
print("- Without LoRA")
print("-----------------------------------------------------------------------------------")
print(output)
print()

# With LoRA

generator.set_loras(lora)

output = generator.generate(
    prompt = prompt,
    max_new_tokens = 500,
    add_bos = True,
    stop_conditions = [tokenizer.eos_token_id, "###"],
    gen_settings = ExLlamaV2Sampler.Settings.greedy()
)

print("-----------------------------------------------------------------------------------")
print("- With LoRA")
print("-----------------------------------------------------------------------------------")
print(output)
print()
