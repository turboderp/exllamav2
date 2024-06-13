import os, json
from datasets import load_dataset

def format_prompt(prompt_format, sp, p):
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
    elif prompt_format == "granite":
        return (
            f"System:\n"
            f"{sp}\n\n"
            f"Question:\n"
            f"{p}\n\n"
            f"Answer:\n"
        )
    elif prompt_format == "chatml":
        return (
            f"<|im_start|>system\n"
            f"{sp}<|im_end|>\n"
            f"<|im_start|>user\n"
            f"{p}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

def get_stop_conditions(prompt_format, tokenizer):
    if prompt_format == "llama":
        return [tokenizer.eos_token_id]
    elif prompt_format == "llama3":
        return [tokenizer.single_id("<|eot_id|>")]
    elif prompt_format == "granite":
        return [tokenizer.eos_token_id, "\n\nQuestion:"]


# Cached dataset loader

def get_dataset(ds_name, category, split):

    cpath = os.path.dirname(os.path.abspath(__file__))
    cpath = os.path.join(cpath, "dataset_cache")
    if not os.path.exists(cpath):
        os.mkdir(cpath)

    filename = ds_name + (("-" + category) if category else "") + "-" + split + ".jsonl"
    filename = filename.replace("/", "_")
    filename = os.path.join(cpath, filename)

    if os.path.exists(filename):
        print(f" -- Loading dataset: {ds_name}/{category if category else '_'}/{split} (cached)...")
        with open(filename, "r") as f:
            return json.load(f)
    else:
        print(f" -- Loading dataset: {ds_name}/{category if category else '_'}/{split}...")
        dataset = load_dataset(ds_name, category, split = split)
        rows = [example for example in dataset]
        with open(filename, "w") as f:
            f.write(json.dumps(rows, indent = 4))
        return rows

