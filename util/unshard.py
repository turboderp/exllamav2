import argparse, json, math, os, glob
from safetensors import safe_open
from safetensors.torch import save_file

parser = argparse.ArgumentParser(description = "Combine sharded .safetensors files")
parser.add_argument("output_file", type = str, help = "Path to output file")
args = parser.parse_args()

output_file = args.output_file
output_base, _ = os.path.splitext(output_file)

# Combine

output_dict = {}
input_files = glob.glob(output_base + "-*.safetensors")

for input_file in input_files:
    print(f" -- Scanning tensors in {input_file}")
    with safe_open(input_file, framework = "pt", device = "cpu") as f:
        for key in f.keys():
            print(f" -- Reading: {key}")
            output_dict[key] = f.get_tensor(key)

# Write output

print(f" -- Writing: {output_file}")
save_file(output_dict, output_file)

# Done

print(f" -- Done")

