import torch
from src.storage.cv import printable_colors


class PhysicalRobustCriterion:
    def __init__(self, perturbation_coefficient=0.001):
        self.cross_entropy = torch.nn.CrossEntropyLoss()
        self.perturbation_coefficient = perturbation_coefficient
        self.printable_colors = printable_colors()

    def __call__(self, outputs, labels, noise, mask):
        return self.cross_entropy(outputs, labels) + self._perturbation_regulation(noise=noise)

    def _perturbation_regulation(self, noise):
        return self.perturbation_coefficient * torch.norm(noise, 2)

    def printability_score(self, noise, mask):
        square_distance = mask * (noise - self.printable_colors) ** 2

