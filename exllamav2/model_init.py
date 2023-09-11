
import argparse, sys, os, glob

from exllamav2 import(
    ExLlamaV2,
    ExLlamaV2Config,
    ExLlamaV2Tokenizer
)

def add_args(parser):

    parser.add_argument("-m", "--model_dir", type = str, help = "Path to model directory")
    parser.add_argument("-gs", "--gpu_split", type = str, help = "VRAM allocation per GPU in GB")
    parser.add_argument("-l", "--length", type = int, help = "Maximum sequence length")
    parser.add_argument("-rs", "--rope_scale", type = float, default = 1.0, help = "RoPE scaling factor")
    parser.add_argument("-ra", "--rope_alpha", type = float, default = 1.0, help = "RoPE alpha value (NTK)")


def print_options(args):

    print(f" -- Model: {args.model_dir}")

    print_opts = []
    if args.gpu_split: print_opts += [f"gpu_split: {args.gpu_split}"]
    if args.length: print_opts += [f"length: {args.length}"]
    print_opts += [f"rope_scale {args.rope_scale}"]
    print_opts += [f"rope_alpha {args.rope_alpha}"]
    print(f" -- Options: {print_opts}")


def check_args(args):

    if not args.model_dir:
        print(" ## Error: No model directory specified")
        sys.exit()

    if not os.path.exists(args.model_dir):
        print(f" ## Error: Can't find model directory: {args.model_dir}")
        sys.exit()

    required_files = ["config.json",
                      "tokenizer.model",
                      "*.safetensors"]

    for filename in required_files:

        path = os.path.join(args.model_dir, filename)
        matches = glob.glob(path)
        if len(matches) == 0:
            print(f" ## Error: Cannot find {filename} in {args.model_dir}")
            sys.exit()


def init(args, quiet = False):

    # Create config

    config = ExLlamaV2Config()
    config.model_dir = args.model_dir
    config.prepare()

    # Set config options

    if args.length: config.max_seq_len = args.length
    config.rope_scale = args.rope_scale
    config.rope_alpha = args.rope_alpha

    # Load model

    if not quiet: print(" -- Loading model...")

    model = ExLlamaV2(config)

    split = None
    if args.gpu_split: split = [float(alloc) for alloc in args.gpu_split.split(",")]
    model.load(split)

    # Load tokenizer

    if not quiet: print(" -- Loading tokenizer...")

    tokenizer = ExLlamaV2Tokenizer(config)

    return model, tokenizer