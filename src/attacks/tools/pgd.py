from __future__ import annotations
import torch
from torch import Tensor
from torch.nn import Module


def multi_step_attack(model: Module, X: Tensor, y: Tensor, input_range: tuple,
                      epsilon=1, alpha=1, num_iter=20, randomize=False):
    """ Construct L_inf adversarial examples on the examples X """
    if torch.max(X) > input_range[1] or torch.min(X) < input_range[0]:
        raise ValueError('scaling have not been yet done.')
    if randomize:
        delta = torch.rand_like(X, requires_grad=True) * (input_range[1] - input_range[0]) + input_range[0]
        delta.data = delta.data * 2 * epsilon - epsilon
        delta.data = (delta.data + X).clamp(input_range[0], input_range[1]) - X
    else:
        delta = torch.zeros_like(X, requires_grad=True)
    delta.retain_grad()
    for t in range(num_iter):
        inputs = X + delta
        loss = torch.nn.CrossEntropyLoss()(model(inputs), y)
        loss.backward(retain_graph=True)

        delta.data = (delta - alpha * delta.grad.detach().sign()).clamp(-epsilon, epsilon)
        delta.data = (delta.data + X).clamp(input_range[0], input_range[1]) - X
        delta.grad.zero_()
    return delta.detach()
