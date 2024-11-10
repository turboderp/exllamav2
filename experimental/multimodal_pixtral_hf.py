import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from exllamav2 import (
    ExLlamaV2,
    ExLlamaV2Config,
    ExLlamaV2Cache,
    ExLlamaV2Tokenizer,
    ExLlamaV2VisionTower,
)

from exllamav2.generator import (
    ExLlamaV2DynamicGenerator,
    ExLlamaV2Sampler,
)

from PIL import Image
import requests

# Unquantized model used for experiment:
#
# https://huggingface.co/mistral-community/pixtral-12b/

model_directory = "/mnt/str/models/pixtral-12b"
config = ExLlamaV2Config(model_directory)
config.max_seq_len = 16384  # default is 1M

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
    tokenizer = tokenizer
)

# Create an MMEmbedding for the image features and a prompt containing the placeholder string

image_embeddings = [
    vision_model.get_image_embeddings(
        model = model,
        tokenizer = tokenizer,
        image = img,
        text_alias = alias,
    )
    for (alias, img) in [
        ("{{IMAGE_1}}", Image.open("test_image_1.jpg")),
        ("{{IMAGE_2}}", Image.open("test_image_2.jpg")),
    ]
]

prompt = "[INST]{{IMAGE_1}}{{IMAGE_2}}\n" + \
         "What are the similarities and differences between these two experiments?[/INST]"

# Run prompt through generator, with embeddings. The tokenizer will insert preepared image tokens in place
# of the aliases

output = generator.generate(
    prompt = prompt,
    max_new_tokens = 500,
    add_bos = True,
    encode_special_tokens = True,
    decode_special_tokens = True,
    stop_conditions = [tokenizer.eos_token_id],
    gen_settings = ExLlamaV2Sampler.Settings.greedy(),
    embeddings = image_embeddings,
)

print(output)