import torch
from tqdm import tqdm
from datasets import load_dataset
import torch.nn as nn
from awq import AutoAWQForCausalLM
from auto_gptq import AutoGPTQForCausalLM
from transformers import AutoTokenizer
import os, json, gc
from exllamav2 import ExLlamaV2, ExLlamaV2Config, ExLlamaV2Cache, ExLlamaV2Tokenizer
from py_markdown_table.markdown_table import markdown_table

seqlen = 2048

def evaluate_perplexity(dataset, model:any, tokenizer, get_tokens, get_logits):
    global seqlen

    def _perplexity(nlls, n_samples, seqlen):
        return torch.exp(torch.stack(nlls).sum() / (n_samples * seqlen))

    data = get_tokens(tokenizer, model, dataset)
    n_samples = data.numel() // seqlen

    nlls = []

    with tqdm(range(n_samples), desc = "Perplexity -") as progress_bar:

        for i in progress_bar:

            start_index = i * seqlen
            end_index = (i + 1) * seqlen
            batch = data[:, start_index:end_index]
            with torch.no_grad():
                logits = get_logits(model, batch)

            shift_logits = logits[:, :-1, :].contiguous().float()
            shift_labels = data[:, start_index:end_index][:, 1:].to(shift_logits.device)
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
            )
            neg_log_likelihood = loss.float() * seqlen
            nlls.append(neg_log_likelihood)

            curr_ppl = _perplexity(nlls, i + 1, seqlen)
            progress_bar.set_description(f"Perplexity {curr_ppl:.3f}")

    ppl = _perplexity(nlls, n_samples, seqlen)

    return ppl.item()

def get_logits_hf(model, batch):
    return model(batch).logits

def get_tokens_hf(tokenizer, model, text):
    data = tokenizer("\n\n".join(text), return_tensors="pt")
    data = data.input_ids.to(model.device)
    return data

def get_logits_exl2(model, batch):
    return model.forward(batch)

def get_tokens_exl2(tokenizer, model, text):
    data = tokenizer.encode("\n\n".join(text), add_bos = True)
    return data

def get_dataset():
    module_dir = os.path.dirname(os.path.abspath(__file__))
    filename = os.path.join(module_dir, "wikitext_cached.json")
    if os.path.exists(filename):
        print("Loading cached dataset...")
        with open(filename, "r") as f:
            return json.load(f)
    else:
        print("Loading dataset...")
        c_dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")["text"]
        with open(filename, "w") as f:
            f.write(json.dumps(c_dataset, indent = 4))
        return c_dataset

def model_instance_awq(model_dir):
    model = AutoAWQForCausalLM.from_pretrained(model_dir, device_map="auto").model
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    return model.eval(), tokenizer

def model_instance_gptq(model_dir):
    model = AutoGPTQForCausalLM.from_quantized(model_dir, device_map="auto").model
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    return model.eval(), tokenizer

def model_instance_exl2(model_dir):
    config = ExLlamaV2Config(model_dir)
    config.max_input_len = seqlen
    config.max_output_len = seqlen
    config.max_seq_len = seqlen
    model = ExLlamaV2(config)
    model.load()
    # model.load(gpu_split = [0,24,0,0])
    # cache = ExLlamaV2Cache(model, lazy = True)
    # model.load_autosplit(cache)
    tokenizer = ExLlamaV2Tokenizer(config, lazy_init = True)
    return model, tokenizer

def flush():
    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()

def run_tests(test_models):

    wikitext = get_dataset()
    # wikitext = wikitext[:200]

    results = []

    num_devices = torch.cuda.device_count()

    for (fw, model_dir, text) in test_models:

        flush()
        for d in range(num_devices): torch.cuda.reset_peak_memory_stats(d)

        print(f"Loading {fw}: {model_dir}")

        if fw == "awq":
            model, tokenizer = model_instance_awq(model_dir)
            ppl = evaluate_perplexity(wikitext, model, tokenizer, get_tokens_hf, get_logits_hf)
            model, tokenizer = None, None

        if fw in ["exl2", "exl2_fp16"]:
            model, tokenizer = model_instance_exl2(model_dir)
            ppl = evaluate_perplexity(wikitext, model, tokenizer, get_tokens_exl2, get_logits_exl2)
            model.unload()
            model, tokenizer = None, None

        if fw == "gptq":
            model, tokenizer = model_instance_gptq(model_dir)
            ppl = evaluate_perplexity(wikitext, model, tokenizer, get_tokens_hf, get_logits_hf)
            model, tokenizer = None, None

        total_memory = sum(torch.cuda.max_memory_allocated(d) for d in range(num_devices))
        max_mem_gb = total_memory / (1024 ** 3)
        print(f"Max CUDA memory: {max_mem_gb:.2f} GB")

        results.append({
            "": text,
            "ppl": f"{ppl:.30f}",
            "Max VRAM": f"{max_mem_gb:.2f} GB"
        })

    return results


if __name__ == "__main__":

    all_models = [
        ("exl2_fp16", "/mnt/str/models/llama3-8b-instruct", "FP16"),
        ("awq", "/mnt/str/models/_awq/llama3-8b-instruct-awq/", "AWQ"),
        ("gptq", "/mnt/str/models/_gptq/mistral-8b-instruct-v0.2-gptq/4bit-act-128g/", "GPTQ 4b-128g-act"),
        ("exl2", "/mnt/str/models/llama3-8b-instruct-exl2/4.0bpw", "EXL2 4.00 bpw"),
        ("exl2", "/mnt/str/models/llama3-8b-instruct-exl2/4.15bpw", "EXL2 4.15 bpw"),
        ("exl2", "/mnt/str/models/llama3-8b-instruct-exl2/5.0bpw", "EXL2 5.00 bpw"),
        ("exl2", "/mnt/str/models/llama3-8b-instruct-exl2/5.3bpw", "EXL2 5.30 bpw"),
    ]

    results = run_tests(all_models)
    markdown = markdown_table(results).set_params(row_sep = 'markdown', quote = False).get_markdown()
    print(markdown)


