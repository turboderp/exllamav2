
from exllamav2 import(
    ExLlamaV2,
    ExLlamaV2Config,
    ExLlamaV2Cache,
    ExLlamaV2Cache_8bit,
    ExLlamaV2Tokenizer,
    model_init,
)

from exllamav2.generator import (
    ExLlamaV2BaseGenerator,
    ExLlamaV2Sampler
)

import argparse, os, math, time
import pandas, fastparquet
import torch
import torch.nn.functional as F
from conversion.tokenize import get_tokens
from conversion.quantize import list_live_tensors

import sys
import json

torch.cuda._lazy_init()
torch.set_printoptions(precision = 10)
# torch.backends.cuda.matmul.allow_tf32 = True
# torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
# torch.set_float32_matmul_precision("medium")

parser = argparse.ArgumentParser(description = "Test inference on ExLlamaV2 model")
parser.add_argument("-ed", "--eval_dataset", type = str, help = "Perplexity evaluation dataset (.parquet file)")
parser.add_argument("-er", "--eval_rows", type = int, default = 128, help = "Number of rows to apply from dataset")
parser.add_argument("-el", "--eval_length", type = int, default = 2048, help = "Max no. tokens per sample")
parser.add_argument("-et", "--eval_token", action = "store_true", help = "Evaluate perplexity on token-by-token inference using cache")
parser.add_argument("-e8", "--eval_token_8bit", action = "store_true", help = "Evaluate perplexity on token-by-token inference using 8-bit cache")
parser.add_argument("-p", "--prompt", type = str, help = "Generate from prompt (basic sampling settings)")
parser.add_argument("-t", "--tokens", type = int, default = 128, help = "Max no. tokens")
parser.add_argument("-ps", "--prompt_speed", action = "store_true", help = "Test prompt processing (batch) speed over context length")
parser.add_argument("-s", "--speed", action = "store_true", help = "Test raw generation speed over context length")

# Initialize model and tokenizer

model_init.add_args(parser)
args = parser.parse_args()
model_init.check_args(args)
model_init.print_options(args)
model, tokenizer = model_init.init(args, allow_auto_split = True)

# Auto split

if not model.loaded:

    print(" -- Loading model...")
    cache = ExLlamaV2Cache(model, lazy = True)
    model.load_autosplit(cache)
    cache = None

# Test generation

if args.prompt:

    with torch.inference_mode():

        cache = ExLlamaV2Cache(model)

        ids = tokenizer.encode(args.prompt)
        tokens_prompt = ids.shape[-1]

        print(f" -- Warmup...")

        generator = ExLlamaV2BaseGenerator(model, cache, tokenizer)
        generator.warmup()

        print(f" -- Generating...")
        print()

        settings = ExLlamaV2Sampler.Settings()
        settings.temperature = 0.85
        settings.top_k = 50
        settings.top_p = 0.8
        settings.token_repetition_penalty = 1.15
        settings.disallow_tokens(tokenizer, [tokenizer.eos_token_id])

        time_begin = time.time()

        output = generator.generate_simple(args.prompt, settings, args.tokens, token_healing = True)

        torch.cuda.synchronize()
        time_prompt = time.time()

        time_end = time.time()

    print(output)
    print()

    total_gen = time_end - time_begin
    print(f" -- Response generated in {total_gen:.2f} seconds, {args.tokens} tokens, {args.tokens / total_gen:.2f} tokens/second (includes prompt eval.)")

    cache = None


# Test perplexity

if args.eval_dataset:

    with torch.inference_mode():

        eval_dataset = args.eval_dataset
        eval_rows = args.eval_rows
        eval_length = args.eval_length

        print(f" -- Running perplexity test")
        print(f" -- Dataset: {eval_dataset}")
        print(f" -- Tokenizing eval data, {eval_rows} rows x {eval_length} tokens...")

        eval_tokens = get_tokens(eval_rows, eval_length, eval_dataset, tokenizer)

        print(f" -- Inference", end = "")
        sys.stdout.flush()

        logprob_sum = 0.0
        logprob_count = 0

        cache = ExLlamaV2Cache(model, max_seq_len = eval_length) if eval_length > model.config.max_input_len else None

        for i in range(eval_tokens.shape[0]):

            if i % 10 == 0: print(".", end = "")
            sys.stdout.flush()

            input_ids = eval_tokens[i:i+1, :]

            input_ids = input_ids[:, :]
            if cache is not None: cache.current_seq_len = 0
            logits = model.forward(input_ids, cache)
            logits = logits[:, :-1, :]
            logits = logits.float() + 1e-10

            target_ids = input_ids[:, 1:].to(logits.device)

            log_probs = F.log_softmax(logits, dim = -1)
            token_log_probs = log_probs.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
            logprob_sum += token_log_probs.sum().item()
            logprob_count += target_ids.numel()

        print()

        mean_log_prob = logprob_sum / logprob_count
        perplexity = math.exp(-mean_log_prob)
        print(f" -- Evaluation perplexity: {perplexity:.4f}")

        def test_ppl_token():
            global logprob_sum, logprob_count, i, input_ids
            global logits, target_ids, log_probs, token_log_probs
            global mean_log_prob, perplexity

            logprob_sum = 0
            logprob_count = 0

            for i in range(eval_tokens.shape[0]):

                cache.current_seq_len = 0

                for j in range(eval_tokens.shape[1] - 1):
                    if j % 256 == 0: print(".", end="")
                    sys.stdout.flush()

                    input_ids = eval_tokens[i:i + 1, j:j + 1]
                    logits = model.forward(input_ids, cache)
                    logits = logits.float() + 1e-10

                    log_probs = F.log_softmax(logits, dim = -1)
                    logprob_sum += log_probs[0, 0, eval_tokens[i, j+1]]
                    logprob_count += 1

            print()

            mean_log_prob = logprob_sum / logprob_count
            perplexity = math.exp(-mean_log_prob)
            print(f" -- Evaluation perplexity: {perplexity:.4f}")


        if args.eval_token:
            print(f" -- Inference (token)", end="")
            sys.stdout.flush()
            cache = ExLlamaV2Cache(model, max_seq_len = eval_length)
            test_ppl_token()

        if args.eval_token_8bit:
            print(f" -- Inference (token, 8-bit cache)", end="")
            sys.stdout.flush()
            cache = ExLlamaV2Cache_8bit(model, max_seq_len = eval_length)
            test_ppl_token()


# Test prompt speed

if args.prompt_speed:

    with torch.inference_mode():

        cache = ExLlamaV2Cache(model)

        ids = torch.randint(0, model.config.vocab_size - 1, (1, model.config.max_seq_len))

        print(f" -- Warmup...")

        model.forward(ids[:, -1:])

        print(f" -- Measuring prompt speed...")

        current_len = 128
        while True:

            time_begin = time.time()

            cache.current_seq_len = 0
            model.forward(ids[:, :current_len], cache, preprocess_only = True)
            torch.cuda.synchronize()

            time_end = time.time()
            tps = current_len / (time_end - time_begin)

            print(f" ** Length {current_len:>5} tokens: {tps:>11.4f} t/s")

            current_len_ = current_len
            current_len = min(current_len + 128, model.config.max_seq_len)
            if current_len == current_len_: break

    cache = None


# Test token speed

if args.speed:

    with torch.inference_mode():

        cache = ExLlamaV2Cache(model)

        print(f" -- Measuring token speed...")
        ids = tokenizer.encode("X")
        model.forward(ids[:, :])

        current_idx = ids.shape[-1]
        next_stop = 128

        while True:

            time_begin = time.time()

            tokens = next_stop - current_idx
            for i in range(tokens):

                logits = model.forward(ids[:, -1:], cache)
                sample = torch.argmax(logits[0, -1]).cpu().unsqueeze(0).unsqueeze(0)
                ids = torch.cat((ids, sample), dim=-1)

            time_end = time.time()
            tps = tokens / (time_end - time_begin)

            print(f" ** Position {current_idx:>5} + {tokens:>3} tokens: {tps:>9.4f} t/s")

            current_idx = next_stop
            next_stop = min(next_stop + 128, model.config.max_seq_len)
            if next_stop == current_idx: break
