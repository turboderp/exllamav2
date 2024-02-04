
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from exllamav2 import (
    ExLlamaV2,
    ExLlamaV2Config,
    ExLlamaV2Cache,
    ExLlamaV2Tokenizer,
)

from exllamav2.generator import (
    ExLlamaV2StreamingGenerator,
    ExLlamaV2Sampler
)

import time, torch

# Initialize model and draft model

torch.set_num_threads(1)

# model_directory = "/mnt/str/models/codellama-34b-instruct-exl2/4.0bpw"
model_directory = "/mnt/str/models/_gptq/TheBloke_Phine-CodeLlama-34B-v2-GPTQ/"
draft_directory = "/mnt/str/models/tinyllama-1b-ckpt503-exl2/3.5bpw"

model_config = ExLlamaV2Config()
model_config.model_dir = model_directory
model_config.prepare()
model_config.max_seq_len = 2048

draft_config = ExLlamaV2Config()
draft_config.model_dir = draft_directory
draft_config.prepare()
draft_config.max_seq_len = 2048

draft = ExLlamaV2(draft_config)
model = ExLlamaV2(model_config)
model_cache = ExLlamaV2Cache(model, lazy = True)
draft_cache = ExLlamaV2Cache(draft, lazy = True)
draft.load_autosplit(draft_cache)
model.load_autosplit(model_cache)

tokenizer = ExLlamaV2Tokenizer(model_config)

# Initialize generators

normal_generator = ExLlamaV2StreamingGenerator(model, model_cache, tokenizer)
speculative_generator = ExLlamaV2StreamingGenerator(model, model_cache, tokenizer, draft, draft_cache, num_speculative_tokens = 5)

# Make sure CUDA is initialized so we can measure performance

normal_generator.warmup()


def test_gen(generator, prompt, settings, max_new_tokens):
    global tokenizer

    # Prompt

    input_ids = tokenizer.encode(prompt)
    prompt_tokens = input_ids.shape[-1]

    # Send prompt to generator to begin stream

    time_begin_prompt = time.time()

    print (prompt, end = "")
    sys.stdout.flush()

    generator.set_stop_conditions([])
    generator.begin_stream(input_ids, settings)

    # Streaming loop. Note that repeated calls to sys.stdout.flush() adds some latency, but some
    # consoles won't update partial lines without it.

    time_begin_stream = time.time()
    generated_tokens = 0

    while True:
        chunk, eos, _ = generator.stream()
        generated_tokens += 1
        print (chunk, end = "")
        sys.stdout.flush()
        if eos or generated_tokens == max_new_tokens: break

    time_end = time.time()

    time_prompt = time_begin_stream - time_begin_prompt
    time_tokens = time_end - time_begin_stream

    print()
    print()
    print(f"Prompt processed in {time_prompt:.2f} seconds, {prompt_tokens} tokens, {prompt_tokens / time_prompt:.2f} tokens/second")
    print(f"Response generated in {time_tokens:.2f} seconds, {generated_tokens} tokens, {generated_tokens / time_tokens:.2f} tokens/second")


# Settings

gen_prompt = "Here is a simple Quicksort implementation in C++:"
# gen_prompt = "What's the best way to learn a new language?"

gen_settings = ExLlamaV2Sampler.Settings()
gen_settings.temperature = 0.6
gen_settings.top_k = 50
gen_settings.top_p = 0.6
gen_settings.token_repetition_penalty = 1.0
gen_settings.disallow_tokens(tokenizer, [tokenizer.eos_token_id])

gen_max_tokens = 250

print()
print("---------------------------------------------------------------------------------")
print("Normal decoding:")
print()

test_gen(normal_generator, gen_prompt, gen_settings, gen_max_tokens)

print()
print("---------------------------------------------------------------------------------")
print("Speculative decoding:")
print()

test_gen(speculative_generator, gen_prompt, gen_settings, gen_max_tokens)

efficiency, accuracy, total_tokens, total_draft_tokens, accepted_draft_tokens = speculative_generator.get_sd_stats()

print("efficiency:", efficiency)
print("accuracy:", accuracy)
print("total_tokens:", total_tokens)
print("total_draft_tokens:", total_draft_tokens)
print("accepted_draft_tokens:", accepted_draft_tokens)
