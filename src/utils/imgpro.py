from __future__ import annotations
import numpy as np
import torch


def scale_gtsrb(images: np.ndarray):
    images = images.astype(np.uint8)
    images = images / 255. - 0.5
    return images.astype(np.float32)


def rescale_gtsrb(images: np.ndarray):
    images = (images + 0.5) * 255.
    return images.astype(np.uint8)


def resize(img: torch.Tensor, size: tuple) -> torch.Tensor:
    img = torch.nn.functional.interpolate(img, size, mode='nearest')
    return img


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
