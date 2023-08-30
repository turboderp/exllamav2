from exllamav2.model import ExLlamaV2, ExLlamaV2Config, ExLlamaV2Cache
from exllamav2.tokenizer import ExLlamaV2Tokenizer
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

parser = argparse.ArgumentParser(description = "Test inference on model")
parser.add_argument("-m", "--model_dir", type = str, help = "Model directory", default = "")
parser.add_argument("-gs", "--gpu_split", type = str, help = "VRAM allocation per GPU in GB", default = "")
parser.add_argument("-ed", "--eval_dataset", type = str, help = "Perplexity evaluation dataset (.parquet file)")
parser.add_argument("-er", "--eval_rows", type = int, default = 128, help = "Number of rows to apply from dataset")
parser.add_argument("-el", "--eval_length", type = int, default = 2048, help = "Max no. tokens per sample")
parser.add_argument("-p", "--prompt", type = str, help = "Prompt")
parser.add_argument("-t", "--tokens", type = int, default = 128, help = "Max no. tokens")
parser.add_argument("-rs", "--rope_scale", type = float, default = 1.0, help = "RoPE scaling factor")
parser.add_argument("-ra", "--rope_alpha", type = float, default = 1.0, help = "RoPE NTK alpha value")

args = parser.parse_args()

print(f" -- Loading model: {args.model_dir}")

config = ExLlamaV2Config()
config.model_dir = args.model_dir
config.max_seq_len = 2048
config.max_input_len = 2048
config.scale_pos_emb = args.rope_scale
config.scale_alpha_value = args.rope_alpha
config.prepare()
model = ExLlamaV2(config)
model.load([16, 24])
#model.load()

tokenizer = ExLlamaV2Tokenizer(config)

# Test generation

if args.prompt is not None:

    cache = ExLlamaV2Cache(model)

    with torch.no_grad():

        ids = tokenizer.encode(args.prompt)
        tokens_prompt = ids.shape[-1]

        print(f" -- Warmup...")

        model.forward(ids[:, :])

        print(f" -- Generating...")
        print()
        print(args.prompt, end = "")
        sys.stdout.flush()

        time_begin = time.time()
        if ids.shape[-1] > 1: model.forward(ids[:, :-1], cache)

        time_prompt = time.time()

        for i in range(args.tokens):

            text1 = tokenizer.decode(ids[:, -2:])[0]

            logits = model.forward(ids[:, -1:], cache)
            sample = torch.argmax(logits[0, -1]).cpu().unsqueeze(0).unsqueeze(0)
            ids = torch.cat((ids, sample), dim = -1)

            text2 = tokenizer.decode(ids[:, -3:])[0]
            text2 = text2[len(text1):]

            print (text2, end = "")
            # sys.stdout.flush()

        time_end = time.time()

    print()
    print()

    total_prompt = time_prompt - time_begin
    total_gen = time_end - time_prompt
    print(f"Prompt processed in {total_prompt:.2f} seconds, {tokens_prompt} tokens, {tokens_prompt / total_prompt:.2f} tokens/second")
    print(f"Response generated in {total_gen:.2f} seconds, {args.tokens} tokens, {args.tokens / total_gen:.2f} tokens/second")

# Test perplexity

if args.eval_dataset is not None:

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

        for i in range(eval_tokens.shape[0]):
        #for i in range(126, 127):

            if i % 10 == 0: print(".", end = "")
            sys.stdout.flush()

            input_ids = eval_tokens[i:i+1, :]

            input_ids = input_ids[:, :-1]
            logits = model.forward(input_ids)

            # print (tokenizer.decode(input_ids))

            target_ids = input_ids[:, 1:].to(logits.device)

            log_probs = F.log_softmax(logits, dim=-1)
            token_log_probs = log_probs.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
            logprob_sum += token_log_probs.sum().item()
            logprob_count += target_ids.numel()

        print()

        mean_log_prob = logprob_sum / logprob_count
        perplexity = math.exp(-mean_log_prob)

        print(f" -- Evaluation perplexity: {perplexity:.4f}")

        xx = 0
