import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from exllamav2 import (
    ExLlamaV2,
    ExLlamaV2Config,
    ExLlamaV2Cache,
    ExLlamaV2Tokenizer,
    ExLlamaV2VisionTower,
)

from exllamav2.generator import (
    ExLlamaV2DynamicGenerator,
    ExLlamaV2DynamicJob,
    ExLlamaV2Sampler,
)

from PIL import Image
import requests, glob

import torch
torch.set_printoptions(precision = 5, sci_mode = False, linewidth=200)

# Model used:
#
# Qwen2-VL:
#   https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct
#   https://huggingface.co/turboderp/Qwen2-VL-7B-Instruct-exl2

streaming = True
greedy = True

model_directory = "/mnt/str/models/qwen2-vl-7b-instruct-exl2/6.0bpw"
images_mask = os.path.join(os.path.dirname(os.path.abspath(__file__)), "media/test_video_*.png")

frames = [
    {"file": f}
    for f in sorted(glob.glob(images_mask))
]

instruction = "Describe this video."

# Initialize model

config = ExLlamaV2Config(model_directory)
config.max_seq_len = 16384  # Pixtral default is 1M

# Load vision model and multimodal projector and initialize preprocessor

vision_model = ExLlamaV2VisionTower(config)
vision_model.load(progress = True)

# Load EXL2 model

model = ExLlamaV2(config)
cache = ExLlamaV2Cache(model, lazy = True, max_seq_len = 16384)
model.load_autosplit(cache, progress = True)
tokenizer = ExLlamaV2Tokenizer(config)

# Create generator

generator = ExLlamaV2DynamicGenerator(
    model = model,
    cache = cache,
    tokenizer = tokenizer,
)

# Util function to get a PIL image from a URL or from a file in the script's directory

def get_image(file = None, url = None):
    assert (file or url) and not (file and url)
    if file:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(script_dir, file)
        return Image.open(file_path)
    elif url:
        return Image.open(requests.get(url, stream = True).raw)

# Convert video to embeddings. Aliases can be given explicitly with the text_alias argument, but here we
# use automatically assigned unique identifiers, then concatenate them into a string

video_embedding = vision_model.get_video_embeddings(
    model = model,
    tokenizer = tokenizer,
    video = [get_image(**img_args) for img_args in frames],
)
video_embeddings = [video_embedding]

# Define prompt

prompt = (
        "<|im_start|>system\n" +
        "You are a helpful assistant.<|im_end|>\n" +
        "<|im_start|>user\n" +
        video_embedding.text_alias +
        # "\n" +
        instruction +
        "<|im_end|>\n" +
        "<|im_start|>assistant\n"
)

# Generate

if streaming:

    input_ids = tokenizer.encode(
        prompt,
        # add_bos = True,
        encode_special_tokens = True,
        embeddings = video_embeddings,
    )

    job = ExLlamaV2DynamicJob(
        input_ids = input_ids,
        max_new_tokens = 500,
        decode_special_tokens = True,
        stop_conditions = [tokenizer.eos_token_id],
        gen_settings = ExLlamaV2Sampler.Settings.greedy() if greedy else None,
        embeddings = video_embeddings,
    )

    generator.enqueue(job)

    print()
    print(prompt, end = ""); sys.stdout.flush()

    eos = False
    while generator.num_remaining_jobs():
        results = generator.iterate()
        for result in results:
            text = result.get("text", "")
            print(text, end = ""); sys.stdout.flush()

    print()

else:

    output = generator.generate(
        prompt = prompt,
        max_new_tokens = 500,
        add_bos = True,
        encode_special_tokens = True,
        decode_special_tokens = True,
        stop_conditions = [tokenizer.eos_token_id],
        gen_settings = ExLlamaV2Sampler.Settings.greedy() if greedy else None,
        embeddings = video_embeddings,
    )

    print(output)