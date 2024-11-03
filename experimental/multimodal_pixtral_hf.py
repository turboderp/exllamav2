import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    PixtralImageProcessor,
    PixtralVisionModel,
)
from PIL import Image
import requests
import safetensors

from exllamav2 import (
    ExLlamaV2,
    ExLlamaV2Config,
    ExLlamaV2Cache,
    ExLlamaV2Tokenizer,
    ExLlamaV2MultimodalProjector
)

from exllamav2.generator import (
    ExLlamaV2DynamicGenerator,
    ExLlamaV2Sampler,
    ExLlamaV2MMEmbedding
)

# Unquantized model used for this experiment:
#
# https://huggingface.co/mistral-community/pixtral-12b/

model_directory = "/mnt/str/models/pixtral-12b"
config = ExLlamaV2Config(model_directory)

# PixtralVisionModel expects vision tower keys to be prefixed with "vision_encoder", but the checkpoint prefixes
# them with "vision_tower". Patch the model implementation to allow the model to load with from_pretrained.

PixtralVisionModel.base_model_prefix = "vision_tower"

# Load multimodal projector

multimodal_projector = ExLlamaV2MultimodalProjector(config)
multimodal_projector.load()

with torch.inference_mode():

    # Initialize preprocessor, vision model and multimodal projector

    image_processor = PixtralImageProcessor.from_pretrained(model_directory, device_map = "cuda:0")
    vision_model = PixtralVisionModel.from_pretrained(
        model_directory,
        device_map = "cuda:0",
        hidden_act = "silu"
    )

    # multimodal_projector = ExLlamaV2MultimodalProjector()
    # safetensors.torch.load_model(
    #     multimodal_projector,
    #     os.path.join(model_directory, "model-00001-of-00006.safetensors"),
    #     strict = False,
    # )
    # multimodal_projector.half().to("cuda:0")

    # Get an input image and process it

    # url = "https://i.imgur.com/JMDz9pC.jpeg"
    # image = Image.open(requests.get(url, stream = True).raw)
    image_path = "car2.jpg"
    image = Image.open(image_path)

    inputs = image_processor(image, return_tensors = "pt")
    pixel_values = [inputs["pixel_values"][0][0].to("cuda:0", torch.half)]
    image_features = vision_model(pixel_values)
    image_features = multimodal_projector.forward(image_features.hidden_states[0].half())
    image_features = image_features[0]
    image_size = inputs["image_sizes"][0][0]

# Load EXL2 model

model = ExLlamaV2(config)
cache = ExLlamaV2Cache(model, lazy = True, max_seq_len = 16384)
model.load_autosplit(cache, progress = True)
tokenizer = ExLlamaV2Tokenizer(config)

# Insert [IMG_BREAK] and [IMG_END] tokens.

features_x = image_size[1] // 16
features_y = image_size[0] // 16
assert image_size == (features_y * 16, features_x * 16)  # Image should be padded in preprocessing

id_break = tokenizer.single_id("[IMG_BREAK]")
id_end = tokenizer.single_id("[IMG_END]")
img_break = model.modules[0].forward(torch.tensor([id_break], dtype = torch.long)).to("cuda:0")
img_end = model.modules[0].forward(torch.tensor([id_end], dtype = torch.long)).to("cuda:0")

dim = image_features.shape[-1]
image_features = image_features.view((features_y, features_x, dim))
break_col = img_break.expand(features_y, -1, -1)
image_features = torch.cat((image_features, break_col), dim = 1)
image_features = image_features.view((features_y * (features_x + 1)), dim)
image_features = torch.cat((image_features, img_end), dim = 0)

# Create generator

generator = ExLlamaV2DynamicGenerator(
    model = model,
    cache = cache,
    tokenizer = tokenizer
)

# Create an MMEmbedding for the image features and a prompt containing the placeholder string

image_tokens = ExLlamaV2MMEmbedding(
    model = model,
    embeddings = image_features,
    text_alias = "{{EMBED_HERE}}"
)

prompt = "[INST] {{EMBED_HERE}}\nDescribe the image. [/INST]"

# Pass embeddings to generator

output = generator.generate(
    prompt = prompt,
    max_new_tokens = 200,
    add_bos = True,
    encode_special_tokens = True,
    decode_special_tokens = True,
    stop_conditions = [tokenizer.eos_token_id],
    # gen_settings = ExLlamaV2Sampler.Settings.greedy(),
    embeddings = [image_tokens],
)

print(output)
