import torch
import pandas, fastparquet
import os
from safetensors.torch import save_file


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

    cal_ds = job["cal_dataset"]
    # eval_ds = job["eval_dataset"]
    # one_file = cal_ds == eval_ds
    rows = job["measurement_rows"] if measure else job["dataset_rows"]
    length = job["measurement_length"] if measure else job["length"]

    # if one_file:
    #
    #     all_tokens = get_tokens(rows * 2, length, cal_ds, tokenizer)
    #     cal_tokens = all_tokens[:rows, :]
    #     eval_tokens = all_tokens[rows:, :]
    #
    # else:
    #
    cal_tokens = get_tokens(rows, length, cal_ds, tokenizer)
    # eval_tokens = get_tokens(rows, length, eval_ds, tokenizer)

    cal_filename = os.path.join(job["out_dir"], "cal_data.safetensors")
    # eval_filename = os.path.join(job["out_dir"], "eval_data.safetensors")
    cal_dict = { "input_ids": cal_tokens }
    # eval_dict = { "input_ids": eval_tokens }
    save_file(cal_dict, cal_filename)
    # save_file(eval_dict, eval_filename)
    job["cal_filename"] = cal_filename
    # job["eval_filename"] = eval_filename