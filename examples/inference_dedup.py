
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from exllamav2 import ExLlamaV2, ExLlamaV2Config, ExLlamaV2Cache, ExLlamaV2Tokenizer, Timer
from exllamav2.generator import ExLlamaV2DynamicGenerator, ExLlamaV2Sampler
from util import format_prompt, get_stop_conditions

model_dir = "/mnt/str/models/llama3-8b-instruct-exl2/4.0bpw"
config = ExLlamaV2Config(model_dir)
model = ExLlamaV2(config)
cache = ExLlamaV2Cache(model, max_seq_len = 8192, lazy = True)
model.load_autosplit(cache, progress = True)

print("Loading tokenizer...")
tokenizer = ExLlamaV2Tokenizer(config)

generator = ExLlamaV2DynamicGenerator(
    model = model,
    cache = cache,
    tokenizer = tokenizer,
)

# Load a short story and prepare some questions about it.

dirname = os.path.dirname(os.path.abspath(__file__))
filename = os.path.join(dirname, "the_black_veil.utf8")
with open(filename, "r", encoding = "utf8") as f:
    input_text = f.read()

questions = [
    "What are the themes of the story?",
    "What is the setting for the story?",
    "List the characters mentioned in the story.",
    "Write a short summary of the story.",
    "Does this story have a happy ending?",
    "Does this story relate to any other works by the same author?",
    "Is the text appropriate for all ages?",
    "Provide up to five examples of outmoded language in the text."
]

# Create prompts to evaluate in parallel. The prompts will all contain the full context, but identical pages are
# deduplicated in the cache, so keys/values for the common prefix of all the prompts are only evaluated and stored
# once. Each prompt works out to about 6200 tokens including the response, but with deduplication up to 5 prompts
# can be batched together in the 8192-token cache

prompt_format = "llama3"

prompts = [
    format_prompt(prompt_format,"You are a helpful AI assistant.", input_text + "\n###\n\n" + question)
    for question in questions
]

# Generate

with Timer() as timer:
    outputs = generator.generate(
        prompt = prompts,
        max_new_tokens = 400,
        stop_conditions = get_stop_conditions(prompt_format, tokenizer),
        completion_only = True,
        encode_special_tokens = True
    )

for question, output in zip(questions, outputs):
    print("-----------------------------------------------------------------------------------")
    print("Q: " + question)
    print()
    print("A: " + output)
    print()

print("-----------------------------------------------------------------------------------")
print(f"Total time: {timer.interval:.2f} seconds")
