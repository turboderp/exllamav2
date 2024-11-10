import torch
import numpy as np
from PIL import Image
from exllamav2.config import ExLlamaV2Config
from exllamav2.vlm.util import (
    convert_to_rgb,
    size_to_longest_edge_and_patch_size,
    normalize_image
)

def preprocess(
    config: ExLlamaV2Config,
    image: Image
) -> torch.Tensor:

    assert "longest_edge" in config.vision_size, \
        "preprocessing size must specify longest_edge"

    patch_size = tuple(config.vision_patch_size[d] for d in ["height", "width"])
    longest_edge = config.vision_size["longest_edge"]
    resample = Image.Resampling(config.vision_resample)
    image_mean = tuple(config.vision_image_mean)
    image_std = tuple(config.vision_image_std)
    rescale_factor = config.vision_rescale_factor

    # Convert to RGB and resize as necessary

    image = convert_to_rgb(image)
    old_size = image.size
    new_size = size_to_longest_edge_and_patch_size(image.size, (longest_edge, longest_edge), patch_size)
    if old_size != new_size:
        image = image.resize(new_size, resample = resample)

    # Convert to numpy array and normalize

    image = np.array(image).astype(np.float32)
    image = image * rescale_factor
    image = normalize_image(image, image_mean, image_std)

    # Convert to tensor, shape (3, resized_height, resized_width)

    image = image.transpose(2, 0, 1)
    image = torch.from_numpy(image).half()
    return image