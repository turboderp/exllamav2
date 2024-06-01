from datasets import load_dataset
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn
import os, json

# Rich progress bar format

def get_progress():

    return Progress(
        TextColumn("[bold blue]{task.fields[name]}", justify = "left"),
        BarColumn(bar_width = None),
        "[progress.percentage]{task.percentage:>3.0f}%",
        TextColumn("{task.completed: 4} of {task.total: 4}", justify = "right"),
        TimeRemainingColumn()
    )

# Cached dataset loader

def get_dataset(ds_name, category, split):

    cpath = os.path.dirname(os.path.abspath(__file__))
    cpath = os.path.join(cpath, "dataset_cache")
    if not os.path.exists(cpath):
        os.mkdir(cpath)

    filename = ds_name + "-" + category + "-" + split + ".jsonl"
    filename = filename.replace("/", "_")
    filename = os.path.join(cpath, filename)

    if os.path.exists(filename):
        print(f" -- Loading dataset: {ds_name}/{category}/{split} (cached)...")
        with open(filename, "r") as f:
            return json.load(f)
    else:
        print(f" -- Loading dataset: {ds_name}/{category}/{split}...")
        dataset = load_dataset(ds_name, category, split = split)
        rows = [example for example in dataset]
        with open(filename, "w") as f:
            f.write(json.dumps(rows, indent = 4))
        return rows
