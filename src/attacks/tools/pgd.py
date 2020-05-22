from __future__ import annotations
import torch
from torch import Tensor
from torch.nn import Module


def projected_gradient_descent(model: Module, X: Tensor, y: Tensor, input_range: tuple,
                               noise: Tensor, epsilon=1/255, alpha=1/255, randomize=False):
    if torch.max(X) > input_range[1] or torch.min(X) < input_range[0]:
        raise ValueError('scaling have not been yet done.')

    # if randomize:
    #     raise NotImplementedError()
    #     # delta = torch.rand_like(X, requires_grad=True) * (input_range[1] - input_range[0]) + input_range[0]
    #     # delta.data = delta.data * 2 * epsilon - epsilon
    #     # delta.data = (delta.data + X).clamp(input_range[0], input_range[1]) - X
    # else:
    #     delta = torch.zeros_like(X, requires_grad=True)

    inputs = X + noise
    inputs.retain_grad()
    loss = torch.nn.CrossEntropyLoss()(model(inputs), y)
    loss.backward(retain_graph=True)
    print(inputs.is_leaf, inputs.requires_grad)
    print(noise.is_leaf, noise.requires_grad)
    new_noise = (noise - alpha * inputs.grad.detach().sign()).clamp(-epsilon, epsilon)
    new_noise = (X + new_noise).clamp(input_range[0], input_range[1]) - X
    return new_noise.detach()
