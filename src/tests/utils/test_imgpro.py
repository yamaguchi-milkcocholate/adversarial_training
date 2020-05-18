import unittest
import torch
import numpy as np
from src.utils.imgpro import *


class TestImgPro(unittest.TestCase):
    def test_resize(self):
        img = torch.tensor(np.ones((5, 3, 10, 10)))
        resized = resize(img=img, size=(5, 5))
        self.assertEqual(resized.shape, (5, 3, 5, 5))

    def test_add_noise(self):
        inputs = torch.tensor(np.zeros((5, 3, 10, 10)))
        mask = torch.tensor(np.ones((3, 10, 10)))
        noise = torch.tensor(np.ones((3, 10, 10)) * 2)
        noisy_inputs = add_noise(inputs=inputs, noise=noise, mask=mask)
        self.assertEqual(noisy_inputs.shape, (5, 3, 10, 10))


if __name__ == '__main__':
    unittest.main()
