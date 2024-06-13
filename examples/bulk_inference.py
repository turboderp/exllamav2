from __future__ import annotations
import sys, os, json, time
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from util import get_dataset, format_prompt, get_stop_conditions
from exllamav2 import ExLlamaV2, ExLlamaV2Config, ExLlamaV2Cache, ExLlamaV2Tokenizer
from exllamav2.generator import ExLlamaV2DynamicGenerator, ExLlamaV2DynamicJob, ExLlamaV2Sampler

# This script loads a HF dataset of prompts and generates a response for each using a Llama3 instruct
# model, then collects all the responses

dataset = "alespalla/chatbot_instruction_prompts"
dataset_category = None
dataset_split = "test"
dataset_column = "prompt"

model_dir = "/mnt/str/models/llama3-8b-instruct-exl2/6.0bpw"
prompt_format = "llama3"
system_prompt = "You are a helpful AI assistant."
max_response_len = 768
cache_size = 100*1024  # Adjust as needed, 100k seems to be a safe size for L3-8B on a single 24 GB GPU
max_rows = 10000

out_file = "output_prompts.json"

# Create model and generator

config = ExLlamaV2Config(model_dir)
model = ExLlamaV2(config)
cache = ExLlamaV2Cache(model, max_seq_len = cache_size, lazy = True)
model.load_autosplit(cache, progress = True)
tokenizer = ExLlamaV2Tokenizer(config)

generator = ExLlamaV2DynamicGenerator(
    model = model,
    cache = cache,
    tokenizer = tokenizer,
    max_batch_size = 1024,
    max_q_size = 1
)

gen_settings = ExLlamaV2Sampler.Settings(
    token_repetition_penalty = 1.0,
    temperature = 1.0,
    top_k = 0,
    top_p = 0.6,
)

# Load a dataset of prompts, print the first couple of entries

dataset_list = get_dataset(dataset, dataset_category, dataset_split)
dataset_list = dataset_list[:max_rows]

print()
print(f"Dataset loaded, {len(dataset_list)} rows:")
print()
for i in range(10):
    print(f"{i}: {dataset_list[i][dataset_column]}")

# Create job list

print()
print("Creating jobs...")

completions = []

for idx, p in enumerate(dataset_list):
    prompt = p["prompt"]
    f_prompt = format_prompt(prompt_format, system_prompt, prompt)
    completions.append(f_prompt)
    prompt_ids = tokenizer.encode(f_prompt, encode_special_tokens = True)
    job = ExLlamaV2DynamicJob(
        input_ids = prompt_ids,
        gen_settings = gen_settings,
        max_new_tokens = max_response_len,
        identifier = idx,
        stop_conditions = get_stop_conditions(prompt_format, tokenizer)
    )
    generator.enqueue(job)
    if (idx + 1) % 1000 == 0 or (idx + 1) == len(dataset_list):
        print(f"{idx + 1} / {len(dataset_list)}")

# Generate

print()
print("Generating...")

num_completions = 0
num_tokens = 0
time_begin = time.time()

while generator.num_remaining_jobs():
    results = generator.iterate()

    # We'll always get at least one result for each active job, even if the result contains no output text
    bsz = len(set([r["identifier"] for r in results]))

    for result in results:
        if not result["eos"]: continue

        # EOS signal is always accompanied by the full completion, so we don't need to collect text chunks
        idx = result["identifier"]
        response = result["full_completion"]
        completions[idx] += response

        # Measure performance
        num_completions += 1
        num_tokens += result["new_tokens"]
        elapsed_time = time.time() - time_begin
        rpm = num_completions / (elapsed_time / 60)
        tps = num_tokens / elapsed_time
        print()
        print("---------------------------------------------------------------------------")
        print(f"Current batch size: {bsz}")
        print(f"Avg. completions/minute: {rpm:.2f}")
        print(f"Avg. output tokens/second: {tps:.2f}")
        print("---------------------------------------------------------------------------")

        # Spam completions to the console
        print()
        print(f"Completion {idx}:")
        print()
        print(completions[idx])

# Save output

with open(out_file, "w") as f:
    json.dump(completions, f, indent = 4)