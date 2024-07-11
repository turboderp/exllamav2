
from exllamav2 import(
    ExLlamaV2,
    ExLlamaV2Config,
    ExLlamaV2Cache,
    ExLlamaV2Cache_8bit,
    ExLlamaV2Tokenizer,
    model_init,
)

from exllamav2.attn import ExLlamaV2Attention

import argparse, os, math, time
import pandas, fastparquet
import torch
import torch.nn.functional as F
from exllamav2.conversion.tokenize import get_tokens
from exllamav2.util import list_live_tensors
import gc

import sys
import json

torch.cuda._lazy_init()
torch.set_printoptions(precision = 10)

parser = argparse.ArgumentParser(description = "Test layer-by-layer hidden state difference between two models")
parser.add_argument("-ed", "--eval_dataset", type = str, help = "Perplexity evaluation dataset (.parquet file)")
parser.add_argument("-er", "--eval_rows", type = int, default = 20, help = "Number of rows to apply from dataset")
parser.add_argument("-el", "--eval_length", type = int, default = 2048, help = "Max no. tokens per sample")
parser.add_argument("-ma", "--model_a", type = str, help = "Path to model A")
parser.add_argument("-mb", "--model_b", type = str, help = "Path to model B")
parser.add_argument("-k", "--keep_layers", type = int, default = 0, help = "Maintain state from model A for this many layers")
parser.add_argument("-tkm", "--topk_max", type = int, default = 5, help = "Max top-K interval to test")

args = parser.parse_args()

# Initialize both models

print(f" -- Model A: {args.model_a}")
print(f" -- Model B: {args.model_b}")

config = (ExLlamaV2Config(), ExLlamaV2Config())
config[0].model_dir = args.model_a
config[1].model_dir = args.model_b
config[0].prepare()
config[1].prepare()
config[0].max_batch_size = 1
config[1].max_batch_size = 1
config[0].arch_compat_overrides()
config[1].arch_compat_overrides()

model = (ExLlamaV2(config[0]), ExLlamaV2(config[1]))
model[0].load(lazy = True)
model[1].load(lazy = True)

num_modules = len(model[0].modules)
assert len(model[1].modules) == num_modules

# Tokenizer

print(f" -- Loading tokenizer")
tokenizer = ExLlamaV2Tokenizer(config[0])

with torch.no_grad():

    # Input

    print(f" -- Tokenizing eval data")
    eval_tokens = get_tokens(args.eval_rows, args.eval_length, args.eval_dataset, tokenizer)
    num_rows, seq_len = eval_tokens.shape

    eval_tokens = [eval_tokens[i:i+1, :] for i in range(eval_tokens.shape[0])]
    attn_params = ExLlamaV2Attention.Params(1, seq_len, 0, None, None)

    # Get embeddings

    print(f" -- Embeddings")
    hidden_state = [[], []]
    for i in [0, 1]:
        module = model[i].modules[0]
        module.load()
        for j in range(num_rows):
            hidden_state[i].append(module.forward(eval_tokens[j]))
        module.unload()

    # Forward

    rfn_error = []

    for idx in range(1, num_modules):

        for i in [0, 1]:

            module = model[i].modules[idx]
            if i == 0:
                print(f" -- {module.key + ' (' + module.name + ')':40}", end = "")

            module.load()

            for j in range(num_rows):
                if i == 1 and idx <= args.keep_layers:
                    hidden_state[1][j] = hidden_state[0][j].clone()
                else:
                    x = hidden_state[i][j].to("cuda:0")
                    x = module.forward(x, cache = None, attn_params = attn_params, past_len = 0, loras = None)
                    hidden_state[i][j] = x.to("cpu")
                    x = None

            module.unload()
            module = None

        max_error_ = 0
        rfn_error_sum = 0
        mse_sum = 0

        for j in range(num_rows):

            x = hidden_state[0][j].to("cuda:0").float()
            y = hidden_state[1][j].to("cuda:0").float()
            rfn_error_sum += torch.linalg.norm(y[0] - x[0], 'fro') / torch.linalg.norm(x[0], 'fro').item()
            x = None
            y = None

        rfn_error_ = rfn_error_sum / num_rows
        print(f" rfn_error: {rfn_error_:8.6f}")
        rfn_error.append(rfn_error_)


    # Test outputs

    def ppl(input_ids_, logits_):

        logprob_sum_ = 0.0
        logprob_count_ = 0

        chunksize = logits_.shape[1] * 16000 // logits_.shape[2]
        b_ = 0
        while b_ < logits_.shape[1]:
            a_ = b_
            b_ = min(b_ + chunksize, logits_.shape[1])

            logits_f = logits_[:, a_:b_, :].float() + 1e-10
            target_ids = input_ids_[:, a_ + 1:b_ + 1].to(logits_.device)

            log_probs = F.log_softmax(logits_f, dim=-1)
            token_log_probs = log_probs.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
            logprob_sum_ += token_log_probs.sum().item()
            logprob_count_ += target_ids.numel()

        return logprob_sum_, logprob_count_

    topk_max = args.topk_max
    logprob_sum = [0, 0]
    logprob_count = [0, 0]
    kl_div_sum = 0
    kl_div_count = 0
    mse_sum = 0
    mse_count = 0
    topk_hits_sum = [[0] * topk_max, [0] * topk_max]
    topk_hits_count = [[0] * topk_max, [0] * topk_max]
    topk_agreement_sum = [0] * topk_max
    topk_agreement_count = [0] * topk_max

    print(f" -- Testing outputs")

    b = 0
    for j in range(num_rows):

        # Perplexity

        x = (hidden_state[0][j].to("cuda:0"), hidden_state[1][j].to("cuda:0"))
        input_ids = eval_tokens[j]

        top_indices = []

        for i in [0, 1]:
            logits = x[i][:, :-1, :]
            logprob_sum__, logprob_count__ = ppl(input_ids, logits)
            logprob_sum[i] += logprob_sum__
            logprob_count[i] += logprob_count__

            _, top_index = torch.topk(logits, topk_max, dim = -1)
            top_index = top_index.cpu().view(-1, topk_max)
            top_indices.append(top_index)
            targets = input_ids[:, 1:].view(-1, 1)

            for t in range(topk_max):
                top_slice = top_index[:, :t + 1]
                hits = torch.eq(targets, top_slice)
                row_hits = hits.any(dim = 1)
                topk_hits_sum[i][t] += row_hits.sum().item()
                topk_hits_count[i][t] += top_slice.shape[0]

        for t in range(topk_max):
            top_slice_a = top_indices[0][:, :t + 1]
            top_slice_b = top_indices[1][:, :t + 1]
            hits = torch.eq(top_slice_a, top_slice_b)
            row_hits = hits.all(dim = 1)
            topk_agreement_sum[t] += row_hits.sum().item()
            topk_agreement_count[t] += top_slice_a.shape[0]

        epsilon = 1e-10
        probs_a = torch.softmax(x[0].float(), dim = -1)
        probs_b = torch.softmax(x[1].float(), dim = -1)
        kl_div = F.kl_div(torch.log(probs_a + epsilon), probs_b, reduction = 'none')
        kl_div_sum += kl_div.sum(dim = -1).mean().item()

        mse_sum += F.mse_loss(probs_a, probs_b)
        mse_count += 1

    perplexity = (math.exp(-logprob_sum[0] / logprob_count[0]), math.exp(-logprob_sum[1] / logprob_count[1]))
    mse = mse_sum / mse_count
    kl_div = kl_div_sum / num_rows

    a_acc = []
    b_acc = []
    a_acc_str = ""
    b_acc_str = ""
    agree_str = ""
    topk_agree = []
    for t in range(topk_max):
        a_acc_ = topk_hits_sum[0][t] / topk_hits_count[0][t]
        b_acc_ = topk_hits_sum[1][t] / topk_hits_count[1][t]
        topk_agree_ = topk_agreement_sum[t] / topk_agreement_count[t]
        a_acc.append(a_acc_)
        b_acc.append(b_acc_)
        topk_agree.append(topk_agree_)
        a_acc_str += f"{a_acc_:6.4f}   "
        b_acc_str += f"{b_acc_:6.4f}   "
        agree_str += f"{topk_agree_:6.4f}   "

# CSV output

print()
print("-----------------")
print()
print(";".join([f"{p:.8f}" for p in perplexity]))
print()
print(f"{kl_div:.8f}")
print(f"{mse:.8f}")
print()
for i in range(topk_max):
    print(f"{i+1};{a_acc[i]:.8f};{b_acc[i]:.8f};{topk_agree[i]:.8f}")
print()
for idx, err in enumerate(rfn_error):
    print(f"{idx};{err:.8f}")
print()
print("-----------------")
print()

# Results

print(f" -- A, ppl: {perplexity[0]:11.8f}   acc: {a_acc_str}")
print(f" -- B, ppl: {perplexity[1]:11.8f}   acc: {b_acc_str}")
print(f" -- Top-K agreement: {agree_str}")
print(f" -- KL divergence: {kl_div:11.8f}")
print(f" -- MSE: {mse:11.8f}")



