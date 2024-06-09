from __future__ import annotations
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from exllamav2 import model_init
from exllamav2 import ExLlamaV2Cache, ExLlamaV2Cache_Q4, ExLlamaV2Cache_Q6, ExLlamaV2Cache_Q8
from exllamav2.generator import ExLlamaV2DynamicGenerator, ExLlamaV2DynamicJob, ExLlamaV2Sampler
import argparse, contextlib
import torch
import util
import random

# Args

parser = argparse.ArgumentParser(description = "Run MMLU evaluation on EXL2 model")
parser.add_argument("-cs", "--cache_size", type = int, default = None)
parser.add_argument("-cq4", "--cache_q4", action = "store_true", help = "Use Q4 cache")
parser.add_argument("-cq6", "--cache_q6", action = "store_true", help = "Use Q6 cache")
parser.add_argument("-cq8", "--cache_q8", action = "store_true", help = "Use Q8 cache")
parser.add_argument("-sub", "--subjects", type = str, default = "all", help = "Comma-separated list of categories to test, or 'all'")
parser.add_argument("-fs", "--fewshot_examples", type = int, default = 5, help = "Number of examples for fewshot examples, max 5")
parser.add_argument("-shf", "--shuffle", action = "store_true", help = "Shuffle choices randomly")
model_init.add_args(parser)
args = parser.parse_args()

# Init model and cache

model_init.check_args(args)
model_init.print_options(args)
model, tokenizer = model_init.init(
    args,
    allow_auto_split = True,
    progress = True,
    max_output_len = 1,
    max_input_len = 2048
)

if args.cache_q4: cache_type = ExLlamaV2Cache_Q4
elif args.cache_q6: cache_type = ExLlamaV2Cache_Q6
elif args.cache_q8: cache_type = ExLlamaV2Cache_Q8
else: cache_type = ExLlamaV2Cache
cache = cache_type(
    model,
    lazy = not model.loaded,
    max_seq_len = args.cache_size or model.config.max_seq_len
)

if not model.loaded:
    model.load_autosplit(cache, progress = True)

# Generator

generator = ExLlamaV2DynamicGenerator(
    model = model,
    cache = cache,
    tokenizer = tokenizer,
    max_batch_size = 1024,
    max_q_size = 1
)

c_options = "ABCD"

gen_settings = ExLlamaV2Sampler.Settings(
    token_repetition_penalty = 1.0,
    temperature = 1.0,
    top_k = 10,
    top_p = 1.0,
)

token_map = [tokenizer.single_id(piece) for piece in [" " + c for c in c_options]]
token_rmap = { token_map[i]: i for i in range(len(c_options)) }
gen_settings.allow_tokens(tokenizer, token_map)

# Get dataset

dataset_dev = util.get_dataset("cais/mmlu", "all", "dev")
dataset_all = util.get_dataset("cais/mmlu", "all", "test")
dataset_dev = sorted(dataset_dev, key = lambda q: q["subject"])
dataset_all = sorted(dataset_all, key = lambda q: q["subject"])

all_subjects = set([q["subject"] for q in dataset_dev])
if args.subjects != "all":
    sel_subjects = args.subjects.split(",")
    for s in sel_subjects:
        if s not in all_subjects:
            print(f"Subject: {s} is not present in dataset")
            sys.exit()
    all_subjects = set(sel_subjects)

# Optionally shuffle

if args.shuffle:
    for problem in dataset_all:
        if problem["subject"] in all_subjects:
            perm = random.sample(range(4), k = 4)
            problem["choices"] = [problem["choices"][i] for i in perm]
            problem["answer"] = perm.index(problem["answer"])

# Format

def format_question(question: str, choices: list[str], answer: int | None):
    f = question + "\n"
    for i, c in enumerate(c_options):
        f += c + ". " + choices[i] + "\n"
    f += "Answer:"
    if answer is not None:
        f += " " + c_options[answer] + "\n\n"
    return f

# Fewshot preprompts

preprompt_ids = {}
with util.get_progress() as progress:
    task1 = progress.add_task("[red]Preprompts", total = len(all_subjects), name = "Preparing preprompts")
    for subject in all_subjects:

        preprompt = f"The following are multiple choice questions (with answers) about {subject.replace('_', ' ')}.\n\n"
        fewshots = 0
        for pq in dataset_dev:
            if fewshots == args.fewshot_examples: break
            if pq["subject"] != subject: continue
            preprompt += format_question(pq["question"], pq["choices"], pq["answer"])
        preprompt_ids[subject] = tokenizer.encode(preprompt, add_bos = True)
        progress.update(task1, advance = 1)

# Questions

total_jobs = 0
for q in dataset_all:
    if q["subject"] in all_subjects:
        total_jobs += 1

with util.get_progress() as progress:
    task1 = progress.add_task("[red]Questions", total=total_jobs, name="Preparing questions")

    for q in dataset_all:
        if q["subject"] not in all_subjects:
            continue

        prompt = format_question(q["question"], q["choices"], None)
        prompt_ids = tokenizer.encode(prompt, add_bos = False)

        job = ExLlamaV2DynamicJob(
            input_ids = torch.cat([preprompt_ids[q["subject"]], prompt_ids], dim = -1),
            gen_settings = gen_settings,
            max_new_tokens = 1,
            return_top_tokens = 4,
            identifier = q,
        )

        generator.enqueue(job)
        progress.update(task1, advance = 1)

# Work

with util.get_progress() as progress:
    task1 = progress.add_task("[red]Sample", total = total_jobs, name = "Testing")

    while generator.num_remaining_jobs():

        results = generator.iterate()
        for result in results:

            if not result["eos"]:
                continue

            # Ignore completion and use top-K tokens only

            top_tokens = result["top_k_tokens"]
            top_probs = result["top_k_probs"]
            q = result["identifier"]

            correct_answer = q["answer"]
            for i in range(top_tokens.shape[-1]):
                if top_tokens[0, 0, i].item() == token_map[correct_answer]:
                    confidence = top_probs[0, 0, i].item()

            q["correct_answer_confidence"] = confidence
            q["answer_correct"] = token_rmap[top_tokens[0, 0, 0].item()] == correct_answer

            progress.update(task1, advance = 1)

# Summarize

total = 0
correct = 0
confidence_sum = 0.0

for q in dataset_all:
    if not "answer_correct" in q:
        continue
    total += 1
    if q["answer_correct"]:
        correct += 1
    confidence_sum += q["correct_answer_confidence"]

print(f"Correct answers: {correct}/{total} = {correct/total*100:.2f}%")
print(f"Confidence: {confidence_sum/total*100:.2f}%")