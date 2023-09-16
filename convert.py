from exllamav2 import ExLlamaV2, ExLlamaV2Config, ExLlamaV2Tokenizer
import argparse, os
import sys
import json
from conversion.tokenize import tokenize
from conversion.quantize import embeddings, measure_quant, quant
from conversion.optimize import optimize
from conversion.compile import compile_model
from conversion.qparams import qparams_headoptions

# import tracemalloc
# tracemalloc.start()

parser = argparse.ArgumentParser(description = "Convert model to ExLlamaV2")
parser.add_argument("-i", "--in_dir", type = str, help = "Input directory", default = "")
parser.add_argument("-o", "--out_dir", type = str, help = "Output directory")
parser.add_argument("-c", "--cal_dataset", type = str, help = "Calibration dataset (.parquet file)", default = "")
parser.add_argument("-r", "--dataset_rows", type = int, default = 100, help = "Number of rows to apply from dataset")
parser.add_argument("-mr", "--measurement_rows", type = int, default = 16, help = "Number of rows to apply from dataset when measuring")
parser.add_argument("-gr", "--gpu_rows", type = int, default = 0, help = "Threshold for paging hidden state to CPU")
parser.add_argument("-l", "--length", type = int, default = 2048, help = "Max no. tokens per sample")
parser.add_argument("-ml", "--measurement_length", type = int, default = 2048, help = "Max no. tokens per sample when measuring")
parser.add_argument("-b", "--bits", type = float, default = 4.156, help = "Target bits per weight")
parser.add_argument("-hb", "--head_bits", type = int, default = 6, help = "Target bits per weight (head layer)")
parser.add_argument("-m", "--measurement", type = str, help = "Reuse previous measurement")
parser.add_argument("-ss", "--shard_size", type = str, help = "Max shard size in MB (default: 8192)", default = 8192)

args = parser.parse_args()

# Check some args

if not args.in_dir:
    print(" ## Please specify input model directory (-i, --in_dir)")
    sys.exit()

if not args.out_dir:
    print(" ## Please specify output/working directory (-o, --out_dir)")
    sys.exit()

if not args.cal_dataset:
    print(" ## Please specify dataset Parquet file (-c, --cal_dataset)")
    sys.exit()

if args.length > 2048 or args.measurement_length > 2048:
    print(" !! Warning: calibration rows > 2048 tokens may result in excessive VRAM use")

if not args.head_bits in qparams_headoptions:
    print(f" ## Error: {args.head_bits} is not a supported option for head layer bitrate")
    sys.exit()

if args.bits < 2 or args.bits > 8:
    print(f" !! Warning: target bitrate {args.bits} will likely not be attainable")

# Arguments

in_dir = None if args.in_dir == "" else os.path.abspath(args.in_dir)
out_dir = os.path.abspath(args.out_dir)
cal_dataset = None if args.cal_dataset == "" else os.path.abspath(args.cal_dataset)
dataset_rows = args.dataset_rows
measurement_rows = args.measurement_rows
gpu_rows = args.gpu_rows
length = args.length
measurement_length = args.measurement_length
bits = args.bits
head_bits = args.head_bits
reuse_measurement = args.measurement
shard_size = args.shard_size if args.shard_size > 0 else 1024 ** 3  # 1 PB = unlimited

if not os.path.exists(out_dir):
    print(f" ## Error: Directory not found: {out_dir}")
    sys.exit()

# Create model without loading weights

config = ExLlamaV2Config()
config.model_dir = in_dir
config.prepare()

model = ExLlamaV2(config)
model.load(lazy = True)

tokenizer = ExLlamaV2Tokenizer(config)

# Job file

job_file = os.path.join(out_dir, "job.json")

# Create new job

def save_job():
    global job_file, job
    with open(job_file, "w") as f:
        f.write(json.dumps(job, indent = 4))

if not os.path.exists(job_file):

    print(f" -- Beginning new job")

    if len(os.listdir(out_dir)) != 0:
        print(f" !! Warning: Output directory is not empty: {out_dir}")

    if in_dir is None:
        print(f" ## Error: No input directory specified")
        sys.exit()

    if cal_dataset is None:
        print(f" ## Error: No calibration dataset specified")
        sys.exit()

    job = { "in_dir": in_dir,
            "out_dir": out_dir,
            "cal_dataset": cal_dataset,
            "dataset_rows": dataset_rows,
            "measurement_rows": measurement_rows,
            "gpu_rows": gpu_rows,
            "length": length,
            "measurement_length": measurement_length,
            "bits": bits,
            "head_bits": head_bits,
            "progress": "begin",
            "shard_size": shard_size
            }

    if reuse_measurement is not None:

        with open(reuse_measurement, "r") as f:

            imp_measurement = json.load(f)
            job["measurement"] = imp_measurement["measurement"]
            job["last_module_idx"] = imp_measurement["last_module_idx"]
            job["base_perplexity"] = imp_measurement["base_perplexity"]
            job["reuse_measurement"] = reuse_measurement

    save_job()

# Resume existing job

else:

    print(f" -- Resuming job")
    print(f" !! Note: Overriding options with settings from existing job")

    with open(job_file, "r") as f:
        job = json.load(f)

    if "invalid" in job:
        print(" ** Error: Corrupted job")
        sys.exit()

    if "shard_size" not in job: job["shard_size"] = shard_size

    job["out_dir"] = out_dir

# Feedback

print(f" -- Input: {job['in_dir']}")
print(f" -- Output: {out_dir}")
print(f" -- Calibration dataset: {job['cal_dataset']}, {job['dataset_rows']} / {job['measurement_rows']} ({job['gpu_rows']}) rows, {job['length']} tokens per sample")
print(f" -- Target bits per weight: {job['bits']} (decoder), {job['head_bits']} (head)")
print(f" -- Max shard size: {job['shard_size']} MB")

# Make sure subfolders exist

out_tensor_dir = os.path.join(job["out_dir"], "out_tensor")
if not os.path.exists(out_tensor_dir):
    os.makedirs(out_tensor_dir)

# Do the things

while True:

    progress = job["progress"]

    if progress == "begin":

        if "reuse_measurement" in job:

            print(f" -- Reusing measurement: {job['reuse_measurement']}")
            job["progress"] = "optimize"
            save_job()

        else:

            print(f" -- Tokenizing samples (measurement)...")
            tokenize(job, save_job, tokenizer, measure = True)
            job["progress"] = "initial_embeddings"
            save_job()

    if progress == "initial_embeddings":

        print(f" -- Token embeddings (measurement)...")
        embeddings(job, save_job, model)
        job["progress"] = "measure_quant"
        save_job()

    if progress == "measure_quant":

        print(f" -- Measuring quantization impact...")
        measure_quant(job, save_job, model)
        job["progress"] = "optimize"
        save_job()

    if progress == "optimize":

        print(f" -- Optimizing...")
        optimize(job, save_job)
        job["progress"] = "tokens_cal"
        save_job()

    if progress == "tokens_cal":

        print(f" -- Tokenizing samples...")
        tokenize(job, save_job, tokenizer)
        job["progress"] = "embeddings"
        save_job()

    if progress == "embeddings":
        print(f" -- Token embeddings again...")
        embeddings(job, save_job, model)
        job["progress"] = "quant"
        save_job()

    if progress == "quant":

        print(f" -- Quantizing...")
        quant(job, save_job, model)
        job["progress"] = "compile"
        save_job()

    if progress == "compile":

        print(f" -- Compiling output file...")
        compile_model(job, save_job, model)
        job["progress"] = "finished"
        save_job()

    if progress == "finished": break

print(f" -- Finished")