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
import requests

# Model used:
#
# Quantized: https://huggingface.co/turboderp/pixtral-12b-exl2
# Unquantized: https://huggingface.co/mistral-community/pixtral-12b/

model_directory = "/mnt/str/models/pixtral-12b-exl2/6.0bpw"
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

# Util function to get a PIL image from a URL or from a file in the script's directory

def get_image(file = None, url = None):
    assert (file or url) and not (file and url)
    if file:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(script_dir, file)
        return Image.open(file_path)
    elif url:
        return Image.open(requests.get(url, stream = True).raw)

# Convert image(s) to embeddings

image_embeddings = [
    vision_model.get_image_embeddings(
        model = model,
        tokenizer = tokenizer,
        image = img,
        text_alias = alias,
    )
    for (alias, img) in [
        ("{{IMAGE_1}}", get_image(file = "test_image_1.jpg")),
        ("{{IMAGE_2}}", get_image(file = "test_image_2.jpg")),
        # ("{{IMAGE_3}}", get_image(url = "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRSERy82bn3jpYKr1cNxMLXTyEsVvSt2wZOIQ&s")),
    ]
]

# Define a prompt using the aliases above as placeholders for image tokens. The tokenizer will replace each alias
# with a range of temporary token IDs, and the model will embed those temporary IDs from their respective sources
# rather than the model's text embedding table.
#
# The temporary IDs are unique for the lifetime of the process and persist as long as a reference is held to the
# corresponding ExLlamaV2Embedding object. This way, images can be reused between generations, or used multiple
# for multiple jobs in a batch, and the generator will be able to apply prompt caching and deduplication to image
# tokens as well as text tokens.
#
# Image token IDs are assigned sequentially, however, so two ExLlamaV2Embedding objects created from the same
# source image will not be recognized as the same image for purposes of prompt caching etc.

prompt = "[INST]{{IMAGE_1}}{{IMAGE_2}}\n" + \
         "What are the similarities and differences between these two experiments?[/INST]"

# Generate

streaming = True
greedy = True

if streaming:

    input_ids = tokenizer.encode(
        prompt,
        add_bos = True,
        encode_special_tokens = True,
        embeddings = image_embeddings,
    )

    job = ExLlamaV2DynamicJob(
        input_ids = input_ids,
        max_new_tokens = 500,
        decode_special_tokens = True,
        stop_conditions = [tokenizer.eos_token_id],
        gen_settings = ExLlamaV2Sampler.Settings.greedy() if greedy else None,
        embeddings = image_embeddings,
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
        embeddings = image_embeddings,
    )

    print(output)