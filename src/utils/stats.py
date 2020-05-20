from __future__ import annotations
import torch


def success_rate(outputs: torch.Tensor, labels: torch.Tensor) -> float:
    _, prediction = torch.max(outputs.data, 1)
    total = labels.size(0)
    correct = (prediction == labels).sum().item()
    return 100 * float(correct / total)
