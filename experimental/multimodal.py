import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import (
    LlavaNextImageProcessor,
    LlavaNextConfig,
    CLIPVisionModel,
)
from transformers.models.llava_next.modeling_llava_next import (
    LlavaNextMultiModalProjector,
    get_anyres_image_grid_shape,
    unpad_image
)
from PIL import Image
import requests
import safetensors

from exllamav2 import ExLlamaV2, ExLlamaV2Config, ExLlamaV2Cache, ExLlamaV2Tokenizer
from exllamav2.generator import ExLlamaV2DynamicGenerator, ExLlamaV2Sampler, ExLlamaV2MMEmbedding

# Load Llava non-text components. We only need config.json, preprocessor_config.json, the CLIP model and the
# projections from Llava.
#
# Llava (for configs): https://huggingface.co/liuhaotian/llava-v1.6-mistral-7b/tree/main
# CLIP model and projections: https://huggingface.co/panoyo9829/llava-v1.6-mistral-7b-CLIP/tree/main

image_processor = LlavaNextImageProcessor.from_pretrained("/mnt/str/models/llava-v1.6-mistral-7b-config/")

vision_tower = CLIPVisionModel.from_pretrained(
    "/mnt/str/models/clip/llava-v1.6-mistral-7b-clip/",
    torch_dtype = torch.float16,
    low_cpu_mem_usage = True
).to("cuda:0")

HF_config = LlavaNextConfig.from_pretrained("/mnt/str/models/llava-v1.6-mistral-7b-config/")
multi_modal_projector = LlavaNextMultiModalProjector(HF_config)
safetensors.torch.load_model(
    multi_modal_projector,
    "/mnt/str/models/clip/llava-v1.6-mistral-7b-clip/llava-v1.6-mistral-7b-PROJ.safetensors"
)
multi_modal_projector.half().to("cuda:0")

with safetensors.safe_open('/mnt/str/models/clip/llava-v1.6-mistral-7b-clip/newline.safetensors', framework='pt') as f:
    newline = f.get_slice('image_newline')[:]
image_newline = torch.nn.Parameter(newline, requires_grad = False).to("cuda:0")

# Get an input image and preprocess it

url = "https://preview.redd.it/dont-touch-the-cats-money-v0-hlh89a18ixz81.jpg?width=640&crop=smart&auto=webp&s=b73c22b5231abd6f2498059a33d21838033215d4"
image = Image.open(requests.get(url, stream = True).raw)

inputs = image_processor(image, return_tensors = "pt")
pixel_values = inputs["pixel_values"].to("cuda:0")
image_sizes = inputs["image_sizes"]

# Get image features

batch_size, num_patches, num_channels, height, width = pixel_values.shape
assert batch_size == 1  # Just testing
reshaped_pixel_values = pixel_values.view(batch_size * num_patches, num_channels, height, width)

image_features = vision_tower(reshaped_pixel_values, output_hidden_states = True)
image_features = image_features.hidden_states[HF_config.vision_feature_layer][:, 1:]
image_features = multi_modal_projector(image_features)

# Split up image features

split_sizes = [image.shape[0] for image in pixel_values]
image_features = torch.split(image_features, split_sizes, dim = 0)
assert len(image_features) == 1  # Just testing
height = width = HF_config.vision_config.image_size // HF_config.vision_config.patch_size

new_image_features = []
for image_idx, image_feature in enumerate(image_features):
    if image_feature.shape[0] > 1:
        base_image_feature = image_feature[0]
        image_feature = image_feature[1:]

        if height * width != base_image_feature.shape[0]:
            raise ValueError("The number of patches is not consistent with the image size.")
        num_patch_height, num_patch_width = get_anyres_image_grid_shape(
            image_sizes[image_idx],
            HF_config.image_grid_pinpoints,
            HF_config.vision_config.image_size,
        )
        image_feature = image_feature.view(num_patch_height, num_patch_width, height, width, -1)
        image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
        image_feature = image_feature.flatten(1, 2).flatten(2, 3)
        image_feature = unpad_image(image_feature, image_sizes[image_idx])
        image_feature = torch.cat(
            (
                image_feature,
                image_newline[:, None, None].expand(*image_feature.shape[:-1], 1),
            ),
            dim = -1,
        )
        image_feature = image_feature.flatten(1, 2).transpose(0, 1)
        image_feature = torch.cat((base_image_feature, image_feature), dim = 0)
    else:
        image_feature = image_feature[0]
        image_feature = torch.cat((image_feature, image_newline[None]), dim = 0)
    new_image_features.append(image_feature)

image_features = torch.stack(new_image_features, dim = 0)
assert image_features.shape[0] == 1
image_features = image_features[0]

# Load EXL2 model
#
# https://huggingface.co/LoneStriker/Mistral-7B-Instruct-v0.2-6.0bpw-h6-exl2-2

model_directory = "/mnt/str/models/mistral-7b-instruct-v0.2-exl2/6.0bpw"
config = ExLlamaV2Config(model_directory)
model = ExLlamaV2(config)
cache = ExLlamaV2Cache(model, lazy = True)
model.load_autosplit(cache)
tokenizer = ExLlamaV2Tokenizer(config)

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
