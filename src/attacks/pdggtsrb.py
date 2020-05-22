from __future__ import annotations
import torch
import numpy as np
from src.attacks.tools.pgd import projected_gradient_descent
from src.attacks.gtsrb import GTSRBAttacker
from src.storage.datasetrepo import GTSRBRepository
from src.utils.imgpro import reshape_to_torch


class GTSRBPDGAttacker(GTSRBAttacker):
    METHOD = 'pdg'

    def __init__(self, target_class: int, model_filename: str, is_valid: bool = True):
        super().__init__(target_class=target_class, model_filename=model_filename)
        if is_valid:
            self.inputs, self.true_labels = self._prepare_valid_data()
            self.labels = torch.tensor(np.array([self.target_class] * len(self.inputs)), dtype=torch.long)

    def run(self, iteration: int, is_terminate_when_success=False):
        self.iteration = iteration
        self.is_terminated = False  # Todo: common run() method?
        noise = torch.zeros_like(self.inputs)
        self._evaluate(outputs=self.model(self.inputs + noise), loss=0.0, i=-1)

        for i in range(iteration):
            noise = projected_gradient_descent(
                model=self.model, X=self.inputs, y=self.labels, input_range=self.input_range,
                noise=noise, epsilon=16/255, alpha=2/255, randomize=False
            )

            self._evaluate(outputs=self.model(self.inputs + noise), loss=0.00, i=i)
            if self.is_terminated and is_terminate_when_success:
                break

        self._save_results(noise=noise, noisy_inputs=self.inputs + noise)

    @classmethod
    def _prepare_valid_data(cls):
        _, _, x_valid, y_valid, _, _ = GTSRBRepository.load_from_pickle_tf()
        x_valid = reshape_to_torch(images=x_valid)
        x_valid = x_valid[:100]
        y_valid = y_valid[:100]
        x_valid = torch.tensor(x_valid, dtype=torch.float32, requires_grad=True)
        y_valid = torch.tensor(y_valid, dtype=torch.long)
        return x_valid, y_valid


if __name__ == '__main__':
    for model in ['model', 'pdg_model_12', 'pdg_model_16']:
        attacker = GTSRBPDGAttacker(target_class=5, model_filename=model)
        attacker.run(iteration=50, is_terminate_when_success=True)
