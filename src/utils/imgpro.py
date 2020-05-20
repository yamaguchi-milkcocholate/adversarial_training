from __future__ import annotations
import numpy as np
import torch
from PIL import Image


def scale_gtsrb(images: np.ndarray):
    images = images.astype(np.uint8)
    images = images / 255. - 0.5
    return images.astype(np.float32)


def rescale_gtsrb(images: np.ndarray):
    images = (images + 0.5) * 255.
    return images.astype(np.uint8)


def resize(img: np.ndarray, size: tuple) -> np.ndarray:
    img = img.astype(np.uint8)
    resized = list()
    for i in range(len(img)):
        i_img = Image.fromarray(img[i])
        i_img = i_img.resize(size=size)
        resized.append(np.asarray(i_img))

    return np.array(resized)


def add_noise(inputs: torch.Tensor, noise: torch.Tensor, mask: torch) -> torch.Tensor:
    """
    :param inputs: (batch_size, channels, row, col)
    :param noise:  (channels, row, col)
    :param mask:   (channels, row, col)
    :return: torch.Tensor
    """
    if inputs.shape[1:] == noise.shape and inputs.shape[1:] == mask.shape:
        return inputs + (noise * mask).reshape((1, mask.shape[0], mask.shape[1], mask.shape[2]))
    else:
        raise ValueError('Invalid shape of tensors.')
