import torch
import argparse, os, glob, sys
from safetensors.torch import save_file

parser = argparse.ArgumentParser(description = "Convert .bin/.pt files to .safetensors")
parser.add_argument("input_files", type = str, help = "Input file(s)")
args = parser.parse_args()

tensor_file_pattern = args.input_files
tensor_files = glob.glob(tensor_file_pattern)

if len(tensor_files) == 0:
    print(f" ## No files matching {tensor_file_pattern}")
    sys.exit()

for file in tensor_files:
    print(f" -- Loading {file}...")
    state_dict = torch.load(file, map_location = "cpu")

    out_file = os.path.splitext(file)[0] + ".safetensors"
    print(f" -- Saving {out_file}...")
    save_file(state_dict, out_file)








