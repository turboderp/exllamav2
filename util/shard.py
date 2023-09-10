import argparse, json, math, os
from safetensors import safe_open
from safetensors.torch import save_file

parser = argparse.ArgumentParser(description = "Split .safetensors file into shards")
parser.add_argument("input_file", type = str, help = "Path to input file")
parser.add_argument("shard_size", type = int, help = "Shard size in megabytes")
args = parser.parse_args()

input_file = args.input_file
input_base, _ = os.path.splitext(input_file)
shard_size = args.shard_size * 1024**2

# Create tensor map

def _tsize(st, key):

    tslice = st.get_slice(key)
    shape = tslice.get_shape()
    numel = 1
    for x in shape: numel *= x
    dtype = tslice.get_dtype()
    del tslice
    if dtype == "I32": return numel * 4
    elif dtype == "I16": return numel * 2
    elif dtype == "F16": return numel * 2
    elif dtype == "F32": return numel * 4
    else: raise ValueError("Unexpected datatype: " + key)

num_files = 0
current_size = shard_size + 1
total_size = 0
tensor_map = []

print(f" -- Scanning tensors in {input_file}")

with safe_open(input_file, framework = "pt", device = "cpu") as f:

    for key in f.keys():

        tensor_size = _tsize(f, key)
        total_size += tensor_size

        if current_size + tensor_size > shard_size:

            num_files += 1
            current_size = 0
            current_list = []
            tensor_map.append(current_list)

        current_size += tensor_size
        current_list.append(key)

# Split into output files

weight_map = {}

for file_index, keys in enumerate(tensor_map):

    shard = {}
    shard_filename = f"{input_base}-{file_index + 1:05}-of-{num_files:05}.safetensors"

    with safe_open(input_file, framework = "pt", device = "cpu") as f:
        for key in keys:
            print(f" -- Reading: {key}")
            shard[key] = f.get_tensor(key)
            weight_map[key] = shard_filename

    print(f" -- Writing: {shard_filename}")
    save_file(shard, shard_filename)

# Compile index

index = { "metadata": { "total_size": total_size }, "weight_map": weight_map }
index_filename = f"{input_file}.index.json"

print(f" -- Writing: {index_filename}")

with open(index_filename, 'w') as f:
    json.dump(index, f, indent = 2)

# Done

print(f" -- Done")
