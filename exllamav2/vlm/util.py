import torch
import numpy as np
from PIL import Image
from typing import Tuple

def convert_to_rgb(image: Image) -> Image:
    """
    Converts an image to RGB format and ensure any transparent regions are converted to white
    """
    if image.mode == "RGB":
        return image

    image = image.convert("RGBA")

    new_image = Image.new("RGBA", image.size, "WHITE")
    new_image.paste(image, (0, 0), image)
    new_image = new_image.convert("RGB")
    return new_image


def size_to_longest_edge_and_patch_size(
    input_size: tuple,
    max_size: tuple,
    patch_size: tuple,
) -> tuple:
    """
    Compute the output size for resizing an image while maintaining aspect ratio and constraining to a
    maximum bounding box while keeping each dimension a multiple of the corresponding patch dimension.
    """

    assert all(p % d == 0 for p, d in zip(max_size, patch_size)), \
        "max_size must be a multiple of patch_size"

    # Reduce to bounding box

    ratio = max(input_size[0] / max_size[0], input_size[1] / max_size[1])
    if ratio > 1:
        output_size = tuple(int(np.ceil(d / ratio)) for d in input_size)
    else:
        output_size = input_size

    # Align size to patch grid

    output_size = tuple((((d + p - 1) // p) * p) for d, p in zip(output_size, patch_size))
    return output_size

def normalize_image(
    image: np.ndarray,
    mean: tuple,
    std: tuple,
) -> np.ndarray:
    """
    Normalizes RGB image in numpy format using the mean and standard deviation specified by `mean` and `std`:
    image = (image - mean(image)) / std
    """

    assert len(mean) == 3 and len(std) == 3, \
        "mean and std arguments must be 3D"

    # Upcast image to float32 if it's not already a float type

    if not np.issubdtype(image.dtype, np.floating):
        image = image.astype(np.float32)

    mean = np.array(mean, dtype = image.dtype)
    std = np.array(std, dtype = image.dtype)
    image = (image - mean) / std
    return image


def position_ids_in_meshgrid(
    height: int,
    width: int,
    max_width: int
):
    """
    Create flat position IDs tensor for grid of patches: id(row, col) = row * max_width + col
    """

    row_indices = torch.arange(height).unsqueeze(1) * max_width
    col_indices = torch.arange(width).unsqueeze(0)
    ids = row_indices + col_indices
    return ids.flatten().unsqueeze(0)