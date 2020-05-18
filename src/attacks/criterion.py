import torch


class PhysicalRobustCriterion:
    def __init__(self, perturbation_coefficient=0.001):
        self.cross_entropy = torch.nn.CrossEntropyLoss()
        self.perturbation_coefficient = perturbation_coefficient

    def __call__(self, outputs, labels, noise):
        return self.cross_entropy(outputs, labels) + self._perturbation_regulation(noise=noise)

    def _perturbation_regulation(self, noise):
        return self.perturbation_coefficient * torch.norm(noise, 2)

    def _printability_score(self):
        raise NotImplemented()
