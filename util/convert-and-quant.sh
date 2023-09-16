#!/bin/bash
#
# Convert .bin model files to .safetensor and run exllamav2 quant
#
# Steps:
# * Download fp16/hf model
# * Edit the paths below to point to your model
# * By default, the script will create the new exllamav2 model in the same parent directory as your model
#

set -eEuo pipefail

##
## Edit these values below
##
model_dir="/aiml/models/jondurbin_airoboros-l2-70b-gpt4-1.4.1"
bits="4.65"
quant_dir="${model_dir}-${bits}bpw"
ppl_file="0007.parquet"
##
## End edit section
##

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ex_dir=$(dirname "$SCRIPT_DIR")

# Save measurements.json to this file for subsequent reuse
measurement_file="$ex_dir/measurement-$(basename $model_dir).json"

cd "$model_dir"
for f in *.bin; do
  basename=$(echo $f | cut -d. -f1)
  if [ -f "${basename}.safetensors" ]; then
    echo "Skipping conversion of $f as the safetensor file already exists"
    continue
  fi
  python "$ex_dir/util/convert_safetensors.py" "$f"
done

if [ -d "$quant_dir" ]; then
  echo "Target output directory already exists, will continue previous run if possible: $quant_dir"
else
  mkdir -p "$quant_dir"
fi

cd "$ex_dir"
if [ ! -f "$ppl_file" ]; then
  echo "Downloading ThePile parquet file for quantizing perplexity calculation"
  wget https://huggingface.co/datasets/EleutherAI/the_pile_deduplicated/resolve/refs%2Fconvert%2Fparquet/default/train/$ppl_file
fi

if [ -f "$measurement_file" ]; then
  echo "Using previous measurement.json file: $measurement_file"
  measurement_arg="-m $measurement_file"
else
  measurement_arg=""
fi

# Run the conversion
python ./convert.py -i "$model_dir" -o "$quant_dir" -c "./$ppl_file" -b $bits $measurement_arg

if [ ! -f "$measurement_file" ]; then
  echo "Copying measurement file (reuse on quants of the same model for different bit targets): "
  cp -p "$quant_dir/measurement.json" "$measurement_file"
fi

echo "Copying original model's model.* and *.json files to new quant directory: $quant_dir"
cp -p "$model_dir"/*.json  "$model_dir"/tokenizer.* "$quant_dir/"

