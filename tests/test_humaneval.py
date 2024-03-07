import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from human_eval.data import write_jsonl, read_problems

from exllamav2 import(
    ExLlamaV2,
    ExLlamaV2Config,
    ExLlamaV2Cache,
    ExLlamaV2Cache_Q4,
    ExLlamaV2Cache_8bit,
    ExLlamaV2Tokenizer,
    model_init
)

from exllamav2.generator import(
    ExLlamaV2BaseGenerator,
    ExLlamaV2Sampler
)

import torch, argparse
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn

# Args

parser = argparse.ArgumentParser(description = "Run HumanEval evaluation on EXL2 model")
parser.add_argument("-o", "--output", type = str, help = "Output .jsonl filename", required = True)
parser.add_argument("-bs", "--batch_size", type = int, default = 10)
parser.add_argument("-spt", "--samples_per_task", type = int, default = 200)
parser.add_argument("-c8", "--cache_8bit", action = "store_true", help = "Use 8-bit (FP8) cache")
parser.add_argument("-cq4", "--cache_q4", action = "store_true", help = "Use Q4 cache")
parser.add_argument("--max_tokens", type = int, default = 768)
model_init.add_args(parser)
args = parser.parse_args()

# Validate args

directory = os.path.dirname(args.output)
if directory and not os.path.isdir(directory):
    print(f" ## Directory for output file {args.output} does not exist.")
    sys.exit()
if os.path.exists(args.output):
    print(f" !! Warning: Output file exists and will be overwritten.")

# Init model

model_init.check_args(args)
model_init.print_options(args)
model, tokenizer = model_init.init(args, allow_auto_split = True, max_batch_size = args.batch_size)

# Create cache

if args.cache_8bit: cache_type = ExLlamaV2Cache_8bit
elif args.cache_q4: cache_type = ExLlamaV2Cache_Q4
else: cache_type = ExLlamaV2Cache
cache = cache_type(model, lazy = not model.loaded, batch_size = args.batch_size)

# Load model

if not model.loaded:

    print(" -- Loading model...")
    model.load_autosplit(cache)

# Generator

gen = ExLlamaV2BaseGenerator(model, cache, tokenizer)
gen_settings = ExLlamaV2Sampler.Settings()
gen_settings.token_repetition_penalty = 1.0
gen_settings.temperature = 0.8
gen_settings.top_k = 100
gen_settings.top_p = 0.8

# Get problems

problems = read_problems()
num_samples_per_task = args.samples_per_task
samples = []
sub_progress = num_samples_per_task > args.batch_size

with Progress(
    TextColumn("[bold blue]{task.fields[name]}", justify = "left"),
    BarColumn(bar_width = None),
    "[progress.percentage]{task.percentage:>3.0f}%",
    TextColumn("{task.completed: 4} of {task.total: 4}", justify = "right"),
    TimeRemainingColumn(),
) as progress:

    task1 = progress.add_task("[red]Problem", total = len(problems), name = "Problems")
    for task_id in problems:

        rem_samples = num_samples_per_task
        if sub_progress: task2 = progress.add_task("[red]Sample", total = num_samples_per_task, name = "Samples", parent = task1)
        while rem_samples:
            bs = min(args.batch_size, rem_samples)

            # Get problem and batch of completions

            problem = [problems[task_id]["prompt"]] * bs
            responses = gen.generate_simple(problem, gen_settings, args.max_tokens, stop_token = tokenizer.eos_token_id)

            for response in responses:

                # Simplified cleanup of response: remove all lines starting from the first line with no indentation,
                # i.e. keep exactly one function

                r = response[len(problem[0]):]
                s =r.split("\n")
                crop = len(s)
                for l in range(1, len(s)):
                    if len(s[l]) > 0:
                        b = s[l][0:1]
                        if b != " " and b != "\t" and b != "#":
                            crop = l
                            break
                r = "\n".join(s[:crop])

                # Store sample

                samples.append(dict(task_id = task_id, completion = r))

            rem_samples -= bs
            if sub_progress: progress.advance(task2, bs)

        if sub_progress: progress.remove_task(task2)
        progress.update(task1, advance = 1)

    # Save output

    print(f" -- Saving: {args.output}")
    write_jsonl(args.output, samples)
