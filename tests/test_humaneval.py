import sys, os, gc
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from human_eval.data import write_jsonl, read_problems

from exllamav2 import(
    ExLlamaV2,
    ExLlamaV2Config,
    ExLlamaV2Cache,
    ExLlamaV2Tokenizer,
)

from exllamav2.generator import(
    ExLlamaV2BaseGenerator,
    ExLlamaV2Sampler
)

import torch

# Models to test

# model_base = "/mnt/str/models/"
# variants = ["mistral-7b-instruct"]
# model_base = "/mnt/str/models/mistral-7b-instruct-exl2"
# variants = ["2.5bpw"]

model_base = "/mnt/str/models/mixtral-8x7b-instruct-exl2/"
# model_base = "/mnt/str/models/tiefighter-13b-exl4/"

# variants = [v for v in os.listdir(model_base) if os.path.isdir(os.path.join(model_base, v))]

variants = \
[
    "2.4bpw",
    "2.5bpw",
    "2.7bpw",
    "3.0bpw",
    "4.0bpw",
    "6.0bpw",
    "8.0bpw",
]

gpu_split = (16, 16, 24)

# Load model

def get_model(base, variant_, gpu_split_, batch_size_):

    model_dir = os.path.join(base, variant_)

    config = ExLlamaV2Config()
    config.model_dir = model_dir
    config.prepare()
    config.max_seq_len = 2048
    config.max_batch_size = batch_size_

    model_ = ExLlamaV2(config)
    print(" -- Loading model: " + model_dir)

    model_.load(gpu_split_)

    tokenizer_ = ExLlamaV2Tokenizer(config)

    cache_ = ExLlamaV2Cache(model_, batch_size = batch_size)
    # cache_ = None

    return model_, cache_, tokenizer_


problems = read_problems()

for variant in variants:

    # Model

    model = None
    cache = None
    tokenizer = None

    gc.collect()
    torch.cuda.empty_cache()
    gc.collect()

    batch_size = 20
    num_samples_per_task = 5
    samples = []

    model, cache, tokenizer = get_model(model_base, variant, gpu_split, batch_size)

    gen = ExLlamaV2BaseGenerator(model, cache, tokenizer)
    gen_settings = ExLlamaV2Sampler.Settings()
    # gen_settings.top_k = 1

    for task_id in problems:
        print(task_id)
        for _ in range(num_samples_per_task):

            # Get problem and batch of completions

            problem = [problems[task_id]["prompt"]] * batch_size
            responses = gen.generate_simple(problem, gen_settings, 500, stop_token = tokenizer.eos_token_id)

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

    # Save output

    write_jsonl(f"samples-{variant}.jsonl", samples)
