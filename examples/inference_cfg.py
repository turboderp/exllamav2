
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from exllamav2 import ExLlamaV2, ExLlamaV2Config, ExLlamaV2Cache, ExLlamaV2Tokenizer, Timer
from exllamav2.generator import ExLlamaV2DynamicGenerator, ExLlamaV2Sampler
from util import format_prompt, get_stop_conditions

model_dir = "/mnt/str/models/llama3-8b-instruct-exl2/4.0bpw"
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

max_new_tokens = 100

# Create our prompts

prompt_format = "llama3"

prompt_a = format_prompt(
    prompt_format,
    "You are a cheerful, bubbly and respectful assistant.",
    "Can i base jump off the Eiffel Tower?"
)

prompt_b = format_prompt(
    prompt_format,
    "You are a rude and obnoxious assistant.",
    "Can i base jump off the Eiffel Tower?"
)

# Generate responses at various CFG scales. CFG evaluates two sequences in parallel, appending the same token to
# both of them but sampling that token from a linear combination of the respective logits from each. The logits are
# mixed as:
#
# mixed_logits = log_softmax(logits_a) * cfg_scale + log_softmax(logits_b) * (1 - cfg_scale)
#
# If cfg_scale < 0, prompt_a has a negative weight in the sum. If cfg_scale > 1, prompt_b has a negative weight and
# can be thought of as a "negative prompt."

cfg_scales = [-1.0, 0.0, 0.25, 0.5, 0.75, 1.0, 2.0]
prompts = [(prompt_a, prompt_b) for _ in cfg_scales]
gen_settings = [ExLlamaV2Sampler.Settings.greedy(cfg_scale = s) for s in cfg_scales]

outputs = generator.generate(
    prompt = prompts,
    max_new_tokens = max_new_tokens,
    gen_settings = gen_settings,
    stop_conditions = get_stop_conditions(prompt_format, tokenizer),
    completion_only = True,
    encode_special_tokens = True
)

for cfg_scale, output in zip(cfg_scales, outputs):
    print(f"Scale: {cfg_scale}")
    print("------------")
    print(output)
    print()