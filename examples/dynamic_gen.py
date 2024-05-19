import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from exllamav2 import ExLlamaV2, ExLlamaV2Config, ExLlamaV2Cache, ExLlamaV2Tokenizer
from exllamav2.generator import ExLlamaV2DynamicGenerator, ExLlamaV2DynamicJob, ExLlamaV2Sampler
from blessed import Terminal
import pprint

# Display modes for this demo:
# 1: One line per job, updated continuously
# 2: Print completions as jobs finish
# 3: Step over output iteration by iteration
# 4: Space heater mode (no output)
display_mode = 1

# Where to find our model
model_dir = "/mnt/str/models/mistral-7b-instruct-v0.2-exl2/4.0bpw"

# Total number of tokens to allocate space for. This is not the max_seq_len supported by the model but
# the total to distribute dynamically over however many jobs are active at once
total_context = 16384

# N-gram or draft model speculative decoding. Largely detrimental to performance at higher batch sizes.
use_ngram = False
use_draft_model = False
draft_model_dir = "/mnt/str/models/tinyllama-1b-32k-exl2/4.0bpw"

# Max number of batches to run at once, assuming the sequences will fit within total_context.
max_batch_size = 20

# Max chunk size. Determines the size of prefill operations. Can be reduced to reduce pauses whenever a
# new job is started, but at the expense of overall prompt ingestion speed.
max_chunk_size = 2048

# Max new tokens per completion. For this example applies to all jobs.
max_new_tokens = 500

# Use LMFE to constrain the output to JSON format. See schema and details below.
json_mode = False

# Demonstrate token healing
healing = False

# Some prompts to feed the generator
prompt_format = "llama"
system_prompt = "You are an AI assistant"
prompts = [
    "What is 2+2 and why?",
    "Can you guess the next number in this sequence: " + ", ".join(str(n) for n in range(500)),
    "Can you guess the next number in this sequence: " + ", ".join(str(n) for n in range(400)),
    "Can you guess the next number in this sequence: " + ", ".join(str(n) for n in range(200)),
    "Can you write a C++ quicksort implementation pretty please?",
    "Hello!",
    "Hi there!",
    "What's the difference smoke and vapor?",
    "What seems out of place in this sequence: " + ", ".join(str(n if n != 123 else 69) for n in range(200)),
    "What seems out of place in this sequence: " + ", ".join(str(n if n != 42 else 111) for n in range(200)),
    "What seems out of place in this sequence: " + ", ".join(str(n if n != 42 else 111) for n in range(200)),
    "What seems out of place in this sequence: " + ", ".join(str(n if n != 42 else 111) for n in range(200)),
    "What seems out of place in this sequence: " + ", ".join(str(n if n != 42 else 111) for n in range(200)),
    "Please guess the next 20 numbers in this sequence: " + ", ".join(str(n) for n in range(700)),
    "Write a short essay about cell membranes.",
    "What's up?",
    "How do I open a can of beans?",
    "How do I open a can of soup?",
    "How do I open a can of strawberry jam?",
    "How do I open a can of raspberry jam?",
    "What's the tallest building in Paris?",
    "What's the most populous nation on Earth?",
    "What's the most populous nation on Mars?",
    "What do the Mole People actually want and how can we best appease them?",
    "Why is the sky blue?",
    "Where is Waldo?",
    "Who is Waldo?",
    "Why is Waldo?",
    "Is it legal to base jump off the Eiffel Tower?",
    "Is it legal to base jump into a volcano?",
    "Why are cats better than dogs?",
    "Why is the Hulk so angry all the time?",
    "How do I build a time machine?",
    "What seems out of place in this sequence: " + ", ".join(str(n if n != 123 else 69) for n in range(200)),
    "Is it legal to grow your own catnip?",
    "What seems out of place in this sequence: " + ", ".join(str(n if n != 160 else 420) for n in range(400)),
    "What seems out of place in this sequence: " + ", ".join(str(n if n != 161 else 421) for n in range(400)),
    "What's inside a black hole?",
    "What do the numbers 2, 4, 8, 16, 32 and 64 have in common?",
    "What do the numbers 2, 3, 5, 7, 11 and 13 have in common?",
    "Is there life on Mars?",
    "Hello!",
    "Hi!",
    "Boop!",
    "Why are cats better than dogs?",
    "Why are cats better than dogs?",
    "Why are cats better than dogs?",
    "Why are cats better than dogs?",
]

term = Terminal()

def format_prompt(sp, p):
    if prompt_format == "llama":
        return f"<s>[INST] <<SYS>>\n{sp}\n<</SYS>>\n\n{p} [/INST]"
    elif prompt_format == "llama3":
        return (
            f"<|begin_of_text|>"
            f"<|start_header_id|>system<|end_header_id|>\n\n"
            f"{sp}<|eot_id|>"
            f"<|start_header_id|>user<|end_header_id|>\n\n"
            f"{p}<|eot_id|>"
            f"<|start_header_id|>assistant<|end_header_id|>\n\n"
        )

def stop_conditions(tokenizer):
    if prompt_format == "llama":
        return [tokenizer.eos_token_id]
    elif prompt_format == "llama3":
        return [tokenizer.single_id("<|eot_id|>")]


# Only import lmfe if json_mode is set

if json_mode:
    import json
    from lmformatenforcer.integrations.exllamav2 import ExLlamaV2TokenEnforcerFilter
    from lmformatenforcer import JsonSchemaParser
    from exllamav2.generator.filters import ExLlamaV2PrefixFilter
    from pydantic import BaseModel
    from typing import Literal

    class JSONResponse(BaseModel):
        response: str
        confidence: Literal["low", "medium", "high"]
        is_subjective: Literal["no", "yes", "possibly"]

    schema_parser = JsonSchemaParser(JSONResponse.schema())


def main():

    if use_draft_model:

        draft_config = ExLlamaV2Config(draft_model_dir)
        draft_model = ExLlamaV2(draft_config)

        draft_cache = ExLlamaV2Cache(
            draft_model,
            max_seq_len = total_context,
            lazy = True
        )

        draft_model.load_autosplit(draft_cache, progress = True)

    else:

        draft_model = None
        draft_cache = None

    # Create config. We use the default max_batch_size of 1 for the model and the default max_input_len of
    # 2048, which will also be the limit of the chunk size for prefill used by the dynamic generator.

    config = ExLlamaV2Config(model_dir)
    config.max_input_len = max_chunk_size
    config.max_attention_size = max_chunk_size ** 2
    model = ExLlamaV2(config)

    # Configure the cache. The dynamic generator expects a batch size of 1 and a max_seq_len equal to
    # the total number of cached tokens. The flat cache will be split dynamically

    cache = ExLlamaV2Cache(
        model,
        max_seq_len = total_context,
        lazy = True
    )

    model.load_autosplit(cache, progress = True)

    # Also, tokenizer

    print("Loading tokenizer...")
    tokenizer = ExLlamaV2Tokenizer(config)

    # Initialize the generator

    generator = ExLlamaV2DynamicGenerator(
        model = model,
        cache = cache,
        draft_model = draft_model,
        draft_cache = draft_cache,
        tokenizer = tokenizer,
        max_batch_size = max_batch_size,
        use_ngram_draft = use_ngram,
        max_chunk_size = max_chunk_size
    )

    # Create jobs

    if json_mode:
        print("Creating jobs... (initializing JSON filters could take a moment.)")

    jobs = []
    for prompt in prompts:
        if json_mode:
            prompt += "\n\n Answer in JSON syntax."
            filters = [
                ExLlamaV2TokenEnforcerFilter(schema_parser, tokenizer),
                ExLlamaV2PrefixFilter(model, tokenizer, "{")
            ]
        else:
            filters = None
        fprompt =  format_prompt(system_prompt, prompt)
        if healing:
            # To test/demonstrate healing, add a broken response prefix
            fprompt += " The an"
        input_ids = tokenizer.encode(fprompt, encode_special_tokens = True)
        job = ExLlamaV2DynamicJob(
            input_ids = input_ids,
            max_new_tokens = max_new_tokens,
            stop_conditions = stop_conditions(tokenizer),
            gen_settings = ExLlamaV2Sampler.Settings(),
            filters = filters,
            filter_prefer_eos = True,
            token_healing = healing
        )
        jobs.append(job)

    # Enqueue all the jobs at once

    generator.enqueue(jobs)

    # To see what's going on, mode 1

    class JobStatusDisplay:

        def __init__(self, job, console_line):
            self.console_line = console_line
            self.job = job
            self.prefill = 0
            self.max_prefill = 0
            self.collected_output = ""
            self.tokens = 0
            self.spaces = " " * 80
            text = term.black(f"{self.console_line:3}:")
            text += term.blue("enqueued")
            print(term.move_xy(0, self.console_line) + text)

        def update(self, r):

            stage = r["stage"]
            stage = r.get("eos_reason", stage)

            self.collected_output += r.get("text", "").replace("\n", "\\n")

            token_ids = r.get("token_ids", None)
            if token_ids is not None: self.tokens += token_ids.shape[-1]

            self.prefill = r.get("curr_progress", self.prefill)
            self.max_prefill = r.get("max_progress", self.max_prefill)

            text = term.black(f"{self.console_line:3}:")
            text += term.blue(f"{stage:16}")
            text += "prefill [ " + term.yellow(f"{self.prefill: 5} / {self.max_prefill: 5}")+" ]"
            text += "   "
            text += term.green(f"{self.tokens: 5} t")
            text += term.black(" -> ")
            text += (self.spaces + self.collected_output)[-80:].replace("\t", " ")

            if "accepted_draft_tokens" in r:
                acc = r["accepted_draft_tokens"]
                rej = r["rejected_draft_tokens"]
                eff = acc / (acc + rej) * 100.0
                text += term.bright_magenta(f"   SD eff.: {eff:6.2f}%")

            print(term.move_xy(0, self.console_line) + text)

    if display_mode == 1:
        print(term.enter_fullscreen())
        displays = { job: JobStatusDisplay(job, line) for line, job in enumerate(jobs) }

    # Streaming loop

    while generator.num_remaining_jobs():
        results = generator.iterate()
        for r in results:
            if display_mode == 1:
                job = r["job"]
                displays[job].update(r)
            if display_mode == 2:
                if r["stage"] == "streaming" and r["eos"]:
                    job = r["job"]
                    print("\n")
                    print(term.black("Input: "))
                    print(term.yellow(tokenizer.decode(job.sequences[0].input_ids.torch(), decode_special_tokens = True)[0]))
                    print()
                    print(term.black("Output:"))
                    if json_mode:
                        try:
                            print(json.dumps(json.loads(r["full_completion"]), indent = 4))
                        except json.decoder.JSONDecodeError:
                            print(term.red("Invalid JSON: ") + r["full_completion"])
                    else:
                        print(r["full_completion"])
                    print()
                    print(term.black("New tokens:  ") + term.green(f"{r['new_tokens']:7} t"))
                    print(term.black("Enqueued:    ") + term.blue(f"{r['time_enqueued']:7.2f} s"))
                    print(term.black("Prefill:     ") + term.blue(f"{r['time_prefill']:7.2f} s"))
                    print(term.black("Generation:  ") + term.blue(f"{r['time_generate']:7.2f} s"))
                    if "accepted_draft_tokens" in r:
                        acc = r["accepted_draft_tokens"]
                        rej = r["rejected_draft_tokens"]
                        eff = acc / (acc + rej) * 100.0
                        print(term.black("SD efficiency:") + term.bright_magenta(f"{eff:7.2f}%"))
        if display_mode == 3 and len(r) > 0:
            print()
            pprint.pprint(results, indent = 4)
            print()
            print("Press any key to continue...")
            with term.cbreak():
                term.inkey()

    if display_mode == 1:
        print(term.move_xy(0, len(displays) + 1) + "Press any key to continue...")
        with term.cbreak():
            term.inkey()

if __name__ == "__main__":
    try:
        main()
    finally:
        pass
        if display_mode == 1:
            print(term.exit_fullscreen())
