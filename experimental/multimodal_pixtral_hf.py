import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from exllamav2 import (
    ExLlamaV2,
    ExLlamaV2Config,
    ExLlamaV2Cache,
    ExLlamaV2Tokenizer,
    ExLlamaV2MultimodalProjector,
    ExLlamaV2VisionTower
)

from exllamav2.generator import (
    ExLlamaV2DynamicGenerator,
    ExLlamaV2Sampler,
    ExLlamaV2MMEmbedding
)

from PIL import Image
import requests

# Get an input image

url = "https://pbs.twimg.com/media/BAeuBsnCIAAUITV.jpg:large"
image = Image.open(requests.get(url, stream = True).raw)

# Unquantized model used for experiment:
#
# https://huggingface.co/mistral-community/pixtral-12b/

model_directory = "/mnt/str/models/pixtral-12b"
config = ExLlamaV2Config(model_directory)
config.max_seq_len = 32768  # default is 1M

# Load multimodal projector

multimodal_projector = ExLlamaV2MultimodalProjector(config)
multimodal_projector.load()

# Load vision tower and preprocessor

vision_tower = ExLlamaV2VisionTower(config)
vision_tower.load(progress = True)

# Preprocess

image_tensor = vision_tower.preprocess(image)
image_tensor = image_tensor.cuda()
image_size = tuple(image_tensor.shape[1:])

# Produce embeddings

embeddings = vision_tower.process(image_tensor)
embeddings = multimodal_projector.forward(embeddings)[0]

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

dim = embeddings.shape[-1]
embeddings = embeddings.view((features_y, features_x, dim))
break_col = img_break.expand(features_y, -1, -1)
embeddings = torch.cat((embeddings, break_col), dim = 1)
embeddings = embeddings.view((features_y * (features_x + 1)), dim)
embeddings = torch.cat((embeddings, img_end), dim = 0)

# Create generator

generator = ExLlamaV2DynamicGenerator(
    model = model,
    cache = cache,
    tokenizer = tokenizer
)

# Create an MMEmbedding for the image features and a prompt containing the placeholder string

image_tokens_a = ExLlamaV2MMEmbedding(
    model = model,
    embeddings = embeddings,
    text_alias = "{{EMBED_A}}"
)

prompt = "[INST]{{EMBED_A}}\nDescribe the image.[/INST]"

# Pass embeddings to generator

output = generator.generate(
    prompt = prompt,
    max_new_tokens = 500,
    add_bos = True,
    encode_special_tokens = True,
    decode_special_tokens = True,
    stop_conditions = [tokenizer.eos_token_id],
    gen_settings = ExLlamaV2Sampler.Settings.greedy(),
    embeddings = [image_tokens_a],
)

print(output)