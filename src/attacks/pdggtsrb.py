from __future__ import annotations
import torch
from src.attacks.tools.pgd import projected_gradient_descent
from src.attacks.gtsrb import GTSRBAttacker


class GTSRBPDGAttacker(GTSRBAttacker):
    METHOD = 'pdg'

    def __init__(self, target_class: int, model_filename: str):
        super().__init__(target_class=target_class, model_filename=model_filename)

    def run(self, iteration: int, is_terminate_when_success=False):
        self.iteration = iteration
        self.is_terminated = False  # Todo: common run() method?
        # self._pick_one_data()
        noise = torch.zeros_like(self.inputs)
        self._evaluate(outputs=self.model(self.inputs + noise), loss=0.0, i=-1)

        for i in range(iteration):
            noise = projected_gradient_descent(
                model=self.model, X=self.inputs, y=self.labels, input_range=self.input_range,
                noise=noise, epsilon=8/255, alpha=1/255, randomize=False
            )

            self._evaluate(outputs=self.model(self.inputs + noise), loss=0.00, i=i)
            if self.is_terminated and is_terminate_when_success:
                break

        self._save_results(noise=noise, noisy_inputs=self.inputs + noise)

    def _pick_one_data(self):
        index = 5
        self.inputs = self.inputs[index].unsqueeze(0)
        self.labels = self.labels[index].unsqueeze(0)
        self.true_labels = self.true_labels[index].unsqueeze(0)


if __name__ == '__main__':
    for model in ['model']:
        attacker = GTSRBPDGAttacker(target_class=5, model_filename=model)
        attacker.run(iteration=10, is_terminate_when_success=True)
        exit()
