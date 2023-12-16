from conversion.qparams import QParams, qparams_options
from exllamav2 import ExLlamaV2, ExLlamaV2Config, ExLlamaV2Tokenizer
from conversion.tokenize import get_tokens, get_standard_calibration
from conversion.qparams_stats import qparams_stats
import torch
import torch.nn.functional as F
import sys, math, json, os
from safetensors import safe_open

model_dir = "/mnt/str/models/_exl2/llama2-7b/"
tensor_dir = "/mnt/str/models/_exl2/__giga/out_tensor/"
measurement_file = "/mnt/str/models/_exl2/__giga/measurement_1.json"
log_file = "/mnt/str/models/_exl2/__giga/log.csv"

config = ExLlamaV2Config()
config.model_dir = model_dir
config.prepare()

model = ExLlamaV2(config)
model.load()

tokenizer = ExLlamaV2Tokenizer(config)
# eval_tokens = get_standard_calibration(True, tokenizer)

eval_dataset = "/mnt/str/datasets/c4_sample.parquet"
eval_rows = 32
eval_length = 2048
eval_tokens = get_tokens(eval_rows, eval_length, eval_dataset, tokenizer)

ref_probs = []

def ppl_test(reference = False):
    global ref_probs

    cache = None
    logprob_sum = 0.0
    logprob_count = 0
    for i in range(eval_tokens.shape[0]):

        # if i % 10 == 0: print(".", end="")
        # sys.stdout.flush()

        input_ids = eval_tokens[i:i + 1, :]

        input_ids = input_ids[:, :]
        if cache is not None: cache.current_seq_len = 0
        logits = model.forward(input_ids, cache)
        logits = logits[:, :-1, :]
        logits = logits.float() + 1e-10

        target_ids = input_ids[:, 1:].to(logits.device)

        lg = logits[0].to("cuda:1")
        probs = F.softmax(lg, dim = -1)
        if reference:
            ref_probs.append(probs)
            avg_kl_div = 0
        else:
            rprobs = torch.log(ref_probs[i] + 1e-10)
            kl_div = F.kl_div(rprobs, probs, reduction = 'none')
            avg_kl_div = kl_div.sum(dim = 1).mean().item()

        log_probs = F.log_softmax(logits, dim=-1)
        token_log_probs = log_probs.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
        logprob_sum += token_log_probs.sum().item()
        logprob_count += target_ids.numel()

    mean_log_prob = logprob_sum / logprob_count
    perplexity = math.exp(-mean_log_prob)
    return perplexity, avg_kl_div

base_ppl, _ = ppl_test(reference = True)
print("Base perplexity:", base_ppl)

def replace_layer(key, qp):

    # Remove original weight

    original_file = config.tensor_file_map[key + ".weight"]
    if key + ".weight" in config.tensor_file_map:
        del config.tensor_file_map[key + ".weight"]

    module = model.modules_dict[key]
    module.unload()

    # Get new quantized tensor to test

    fdesc = qp.get_desc(True)
    tensor_file = os.path.join(tensor_dir, fdesc + "____" + key + ".safetensors")
    assert os.path.exists(tensor_file)

    # Insert q tensor components into map

    keys_to_unset = []
    with safe_open(tensor_file, framework="pt", device="cpu") as f:
        for k in f.keys():
            config.tensor_file_map[k] = tensor_file
            keys_to_unset.append(k)

    # Load the quantized layer

    module.load()

    for k in keys_to_unset: del config.tensor_file_map[k]
    config.tensor_file_map[key + ".weight"] = original_file

def unreplace_layer(key):

    module = model.modules_dict[key]
    module.unload()
    module.load()

layers = [0, 1, 2, 16, 24, 31]
results = []

print("qparams_stats = \\")
print("[")

for qps in qparams_stats:

    print("    [")

    for x in qps:
        if x is None:
            print("        None,")
        elif isinstance(x, QParams):
            print("        " + str(x) + ",")
        else:
            print(f"        {x:1.10f},")

    if len(qps) == 7:
        for i in layers:

            lkey = "model.layers." + str(i)

            # Get new strat

            s_q, s_k, s_v, s_o, s_g, s_u, s_d = qps

            # Replace

            if s_q: replace_layer(lkey + ".self_attn.q_proj", s_q)
            if s_k: replace_layer(lkey + ".self_attn.k_proj", s_k)
            if s_v: replace_layer(lkey + ".self_attn.v_proj", s_v)
            if s_o: replace_layer(lkey + ".self_attn.o_proj", s_o)
            if s_g: replace_layer(lkey + ".mlp.gate_proj", s_g)
            if s_u: replace_layer(lkey + ".mlp.up_proj", s_u)
            if s_d: replace_layer(lkey + ".mlp.down_proj", s_d)

            # Test

            new_ppl, kldiv = ppl_test()

            print(f"        {kldiv:1.10f},")

            if s_q: unreplace_layer(lkey + ".self_attn.q_proj")
            if s_k: unreplace_layer(lkey + ".self_attn.k_proj")
            if s_v: unreplace_layer(lkey + ".self_attn.v_proj")
            if s_o: unreplace_layer(lkey + ".self_attn.o_proj")
            if s_g: unreplace_layer(lkey + ".mlp.gate_proj")
            if s_u: unreplace_layer(lkey + ".mlp.up_proj")
            if s_d: unreplace_layer(lkey + ".mlp.down_proj")

    print("    ],")

print("]")