
import sys, os, gc, time, random
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from exllamav2 import(
    ExLlamaV2,
    ExLlamaV2Config,
    ExLlamaV2CacheBase,
    ExLlamaV2Cache,
    ExLlamaV2Cache_8bit,
    ExLlamaV2Tokenizer,
)

from exllamav2.generator import (
    ExLlamaV2BaseGenerator,
    ExLlamaV2StreamingGenerator,
    ExLlamaV2Sampler
)

import time

model: ExLlamaV2
config: ExLlamaV2Config
tokenizer: ExLlamaV2Tokenizer
cache: ExLlamaV2CacheBase

def unload():
    global model, config, tokenizer, cache

    model.unload()
    model = None
    config = None
    cache = None
    tokenizer = None

    gc.collect()
    torch.cuda.empty_cache()


def load_model(model_dir, split = None, cache_8bit = False):
    global model, config, tokenizer, cache

    config = ExLlamaV2Config()
    config.model_dir = model_dir
    config.prepare()

    model = ExLlamaV2(config)
    print(" -- Loading model: " + model_dir)

    model.load(split)

    tokenizer = ExLlamaV2Tokenizer(config)

    if cache_8bit:
        print(" -- Creating 8-bit cache")
        cache = ExLlamaV2Cache_8bit(model, batch_size = 4)
    else:
        print(" -- Creating 16-bit cache")
        cache = ExLlamaV2Cache(model, batch_size = 4)


def test_gen_normal(prompt, max_new_tokens):
    global model, config, tokenizer, cache

    print("--------------------------------")
    print("Generating, normal")
    print()

    generator = ExLlamaV2BaseGenerator(model, cache, tokenizer)

    settings = ExLlamaV2Sampler.Settings()
    settings.temperature = 0.85
    settings.top_k = 50
    settings.top_p = 0.8
    settings.top_a = 0.0
    settings.token_repetition_penalty = 1.15
    settings.disallow_tokens(tokenizer, [tokenizer.eos_token_id])

    generator.warmup()
    time_begin = time.time()

    output = generator.generate_simple(prompt, settings, max_new_tokens, seed=1234)

    time_end = time.time()
    time_total = time_end - time_begin

    print(output)
    print()
    print(f"Response generated in {time_total:.2f} seconds, {max_new_tokens} tokens, {max_new_tokens / time_total:.2f} tokens/second")


def test_gen_streaming(prompt, max_new_tokens):
    global model, config, tokenizer, cache

    print("--------------------------------")
    print("Generating, streaming")
    print()

    generator = ExLlamaV2StreamingGenerator(model, cache, tokenizer)

    settings = ExLlamaV2Sampler.Settings()
    settings.temperature = 0.85
    settings.top_k = 50
    settings.top_p = 0.8
    settings.top_a = 0.0
    settings.token_repetition_penalty = 1.15
    settings.disallow_tokens(tokenizer, [tokenizer.eos_token_id])

    input_ids = tokenizer.encode(prompt)
    prompt_tokens = input_ids.shape[-1]

    print(prompt, end = "")
    sys.stdout.flush()

    time_begin_prompt = time.time()

    generator.set_stop_conditions([])
    generator.begin_stream(input_ids, settings)

    time_begin_stream = time.time()
    generated_tokens = 0

    while True:
        chunk, eos, _ = generator.stream()
        generated_tokens += 1
        print(chunk, end = "")
        sys.stdout.flush()
        if eos or generated_tokens == max_new_tokens: break

    time_end = time.time()

    time_prompt = time_begin_stream - time_begin_prompt
    time_tokens = time_end - time_begin_stream

    print()
    print()
    print(f"Prompt processed in {time_prompt:.2f} seconds, {prompt_tokens} tokens, {prompt_tokens / time_prompt:.2f} tokens/second")
    print(f"Response generated in {time_tokens:.2f} seconds, {generated_tokens} tokens, {generated_tokens / time_tokens:.2f} tokens/second")


def test_gen_batch(max_new_tokens):

    print("--------------------------------")
    print("Generating, batched")
    print()

    generator = ExLlamaV2BaseGenerator(model, cache, tokenizer)

    settings = ExLlamaV2Sampler.Settings()
    settings.temperature = 0.85
    settings.top_k = 50
    settings.top_p = 0.8
    settings.top_a = 0.0
    settings.token_repetition_penalty = 1.15
    settings.disallow_tokens(tokenizer, [tokenizer.eos_token_id])

    generator.warmup()
    time_begin = time.time()

    prompts = ["Here's how to create a powerful love potio",
               "For once,",
               "The events of the American Civil W",
               "A bird in the hand is worth"]

    output = generator.generate_simple(prompts, settings, max_new_tokens, seed = 1234, token_healing = True)

    time_end = time.time()
    time_total = time_end - time_begin

    for o in output:
        print(o)
        print("---")
    print()
    print(f"Response generated in {time_total:.2f} seconds, {max_new_tokens} tokens, throughput {4 * max_new_tokens / time_total:.2f} tokens/second")


def test_multicache(max_new_tokens):

    print("--------------------------------")
    print("Generating, batched multi cache")
    print()

    settings = ExLlamaV2Sampler.Settings()
    settings.temperature = 0.85
    settings.top_k = 50
    settings.top_p = 0.8
    settings.top_a = 0.0
    settings.token_repetition_penalty = 1.15
    settings.disallow_tokens(tokenizer, [tokenizer.eos_token_id])

    prompts = ["Here's how to create a powerful love potion",
               "For once,",
               "The events of the American Civil War",
               "A bird in the hand is worth"]

    caches = [ExLlamaV2Cache(model, max_seq_len = 256) for _ in range(len(prompts))]
    input_ids = []

    for i in range(len(prompts)):

        input_ids.append(tokenizer.encode(prompts[i]))
        model.forward(input_ids[i][:, :-1], caches[i], input_mask = None, preprocess_only = True)

    time_begin = time.time()

    for i in range(max_new_tokens):

        inputs = torch.cat([x[:, -1:] for x in input_ids], dim = 0)
        logits = model.forward(inputs, caches, input_mask = None).float().cpu()

        r = random.random()
        for j in range(len(input_ids)):
            token, _, _ = ExLlamaV2Sampler.sample(logits[j:j + 1, :, :], settings, input_ids[j], r, tokenizer)
            input_ids[j] = torch.cat([input_ids[j], token], dim = 1)

    output = [tokenizer.decode(ids)[0] for ids in input_ids]

    time_end = time.time()
    time_total = time_end - time_begin

    for o in output:
        print(o)
        print("---")
    print()
    print(f"Response generated in {time_total:.2f} seconds, {max_new_tokens} tokens, throughput {4 * max_new_tokens / time_total:.2f} tokens/second")


def tests(model_dir, cache_8bit, use_split):

    if use_split: split = [1, 24]
    else: split = None
    print("--------------------------------")
    print(f" -- Split: {split}")
    load_model(model_dir, split = split, cache_8bit = cache_8bit)

    test_gen_normal("Our story begins in the Scottish town of Auchtermuchty, where once", 150)
    test_gen_streaming("Our story begins in the Scottish town of Auchtermuchty, where once", 150)
    test_gen_batch(40)
    if model.is_quant(): test_multicache(40)

    unload()


q_model_directory = "/mnt/str/models/mistral-7b-instruct-exl2/4.0bpw/"
f_model_directory = "/mnt/str/models/tinyllama-1b-ckpt503/"

tests(q_model_directory, False, False)
tests(q_model_directory, False, True)
tests(q_model_directory, True, False)
tests(q_model_directory, True, True)
tests(f_model_directory, False, False)
tests(f_model_directory, False, True)
tests(f_model_directory, True, False)
tests(f_model_directory, True, True)