import torch
import argparse, os
from safetensors.torch import save_file

parser = argparse.ArgumentParser(description="Convert .bin/.pt files to .safetensors")
parser.add_argument("input_files", nargs='+', type=str, help="Input file(s)")
args = parser.parse_args()

for file in args.input_files:
    print(f" -- Loading {file}...")
    state_dict = torch.load(file, map_location="cpu")

    out_file = os.path.splitext(file)[0] + ".safetensors"
    print(f" -- Saving {out_file}...")
    save_file(state_dict, out_file, metadata = {'format': 'pt'})