import torch
import pandas, fastparquet
import os
from safetensors.torch import save_file
import random
from exllamav2.conversion.bot_status import print_stage

def get_tokens(num_rows, length, filename, tokenizer):

    min_tokens = num_rows * length

    df = pandas.read_parquet(filename, engine = "fastparquet")
    df['concatenated'] = df.apply(lambda r: ' '.join([str(v) for v in r.values]), axis = 1)

    all_tokens = torch.empty((1,0), dtype = torch.long)

    for _, row in df['concatenated'].items():
        tokens = tokenizer.encode(row)
        all_tokens = torch.cat((all_tokens, tokens), dim = -1)
        if all_tokens.shape[-1] >= min_tokens: break

    if all_tokens.shape[-1] < min_tokens:
        print(f" ** Warning: Not enough sample data in {filename}")

    all_tokens = all_tokens.flatten()[:min_tokens]
    all_tokens = all_tokens.view((num_rows, length))

    num_print_tokens = 50
    data_sample = all_tokens[0, :num_print_tokens]
    print(f" -- First {num_print_tokens} tokens of dataset:")
    print(f"    {repr(tokenizer.decode(data_sample))}")
    data_sample = all_tokens[-1, -num_print_tokens:]
    print(f" -- Last {num_print_tokens} tokens of dataset:")
    print(f"    {repr(tokenizer.decode(data_sample))}")

    return all_tokens


def tokenize(job, save_fn, tokenizer, measure = False):

    print_stage(job, "Tokenizing (1)" if measure else "Tokenizing (2)", 0, 1)

    cal_ds = job["cal_dataset"]

    if cal_ds is not None:
        rows = job["measurement_rows"] if measure else job["dataset_rows"]
        length = job["measurement_length"] if measure else job["length"]
        cal_tokens = get_tokens(rows, length, cal_ds, tokenizer)
    else:
        cal_tokens = get_standard_calibration(job, measure, tokenizer)
        if measure:
            job["measurement_rows"] = cal_tokens.shape[0]
        else:
            job["dataset_rows"] = cal_tokens.shape[0]

    cal_filename = os.path.join(job["out_dir"], "cal_data.safetensors")
    cal_dict = { "input_ids": cal_tokens }
    save_file(cal_dict, cal_filename)
    job["cal_filename"] = cal_filename

    print_stage(job, "Tokenizing (1)" if measure else "Tokenizing (2)", 1, 1)


def get_standard_calibration(job, measure, tokenizer):

    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "standard_cal_data")
    file_c4 =os.path.join(data_dir, "c4.utf8")
    file_code =os.path.join(data_dir, "code.utf8")
    file_multilingual =os.path.join(data_dir, "multilingual.utf8")
    file_technical =os.path.join(data_dir, "technical.utf8")
    file_wiki = os.path.join(data_dir, "wiki.utf8")
    file_tiny = os.path.join(data_dir, "tiny.utf8")

    rows = []
    rows_c4 = 2 if measure else 10
    rows_wiki = 4 if measure else 48
    rows_code = 3 if measure else 15
    rows_tiny = 2 if measure else 10
    rows_multilingual = 3 if measure else 15
    rows_multilingual_s = 1 if measure else 5
    rows_technical = 2 if measure else 10
    rows_random = 2

    ctx = min(2048, job["measurement_length"] if measure else job["length"])

    # C4: 10 rows

    with open(file_c4, "r", encoding="utf8") as f:
        lines = f.readlines()

    text = "\n\n".join(lines)
    tokens = tokenizer.encode(text)
    tokens = tokens[:, : tokens.shape[-1] - (tokens.shape[-1] % ctx)]
    tokenized_rows = tokens.view(-1, ctx)

    for i in range(rows_c4):
        rows.append(tokenized_rows[i:i+1])

    # Wiki: 24 aligned rows + 24 aligned rows with BOS

    with open(file_wiki, "r", encoding="utf8") as f:
        text = f.read()

    articles = [a[a.find("\n") + 1:] for a in text.split("</doc>\n")]
    tokenized_articles = [tokenizer.encode(a, add_bos = True, add_eos = True) for a in articles]

    idx = 0
    for r in range(rows_wiki):
        length = 0
        idx0 = idx
        while length < ctx + 1:
            length += tokenized_articles[idx].shape[-1]
            idx += 1
        row = torch.cat(tokenized_articles[idx0 : idx], dim = -1)
        if r < rows_wiki // 2: row = row[:, 1:ctx+1]
        else: row = row[:, :ctx]
        rows.append(row)

    # Code: 15 rows

    with open(file_code, "r", encoding="utf8") as f:
        text = f.read()

    tokens = tokenizer.encode(text)
    tokens = tokens[:, : tokens.shape[-1] - (tokens.shape[-1] % ctx)]
    tokenized_rows = tokens.view(-1, ctx)

    for i in range(rows_code):
        rows.append(tokenized_rows[i:i+1])

    # Tinystories: 5 aligned rows + 5 aligned rows with BOS

    with open(file_tiny, "r", encoding="utf8") as f:
        text = f.read()

    articles = text.split("<|endoftext|>")
    tokenized_articles = [tokenizer.encode(a.strip(), add_bos = True, add_eos = True) for a in articles]

    idx = 0
    for r in range(rows_tiny):
        length = 0
        idx0 = idx
        while length < ctx + 1:
            length += tokenized_articles[idx].shape[-1]
            idx += 1
        row = torch.cat(tokenized_articles[idx0 : idx], dim = -1)
        if r < rows_tiny // 2: row = row[:, 1:ctx+1]
        else: row = row[:, :ctx]
        rows.append(row)

    # Multilingual: 15 rows + 5 shuffled rows

    with open(file_multilingual, "r", encoding="utf8") as f:
        text = f.read()

    tokens = tokenizer.encode(text)
    tokens = tokens[:, : tokens.shape[-1] - (tokens.shape[-1] % ctx)]
    tokenized_rows = tokens.view(-1, ctx)

    for i in range(rows_multilingual):
        rows.append(tokenized_rows[i:i+1])

    tokenized_rows = tokens.view(-1, 128)
    random.seed(69420)
    for i in range(rows_multilingual_s):
        row = []
        for j in range(ctx // 128):
            k = random.randint(0, tokenized_rows.shape[0] - 1)
            row.append(tokenized_rows[k].unsqueeze(0))
        rows.append(torch.cat(row, dim = -1))

    # Randomized: 2 rows

    vocab_size = tokenizer.get_vocab_size()
    random.seed(69420)
    for i in range(rows_random):
        row = torch.randint(0, vocab_size, (1, ctx), dtype = torch.long)
        rows.append(row)

    # Technical: 10 rows

    with open(file_technical, "r", encoding="utf8") as f:
        text = f.read()

    tokens = tokenizer.encode(text)
    tokens = tokens[:, : tokens.shape[-1] - (tokens.shape[-1] % ctx)]
    tokenized_rows = tokens.view(-1, ctx)

    for i in range(rows_technical):
        rows.append(tokenized_rows[i:i+1])

    # for idx, r in enumerate(rows):
    #     print("------------------------------------------------------------------------------")
    #     print(idx)
    #     print("--------")
    #     print(tokenizer.decode(r))

    return torch.cat(rows, dim = 0)













