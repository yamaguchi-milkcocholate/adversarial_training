from __future__ import annotations
import torch
from src.attacks.tools.pgd import multi_step_attack
from src.attacks.gtsrb import GTSRBAttacker


class GTSRBPDGAttacker(GTSRBAttacker):
    METHOD = 'pdg'

    def __init__(self, target_class: int, model_filename: str):
        super().__init__(target_class=target_class, model_filename=model_filename)

    def run(self, iteration: int, is_terminate_when_success=False):
        self.iteration = iteration
        self.is_terminated = False  # Todo: common run() method?
        self._pick_one_data()
        noise = torch.zeros_like(self.inputs)
        noisy_inputs = self.inputs + noise
        self._evaluate(outputs=self.model(noisy_inputs), loss=0.0, i=-1)

        for i in range(iteration):
            delta = multi_step_attack(
                model=self.model, X=self.inputs, y=self.labels, input_range=self.input_range,
                epsilon=8/255, alpha=2/255, num_iter=1, randomize=True
            )
            noise += delta
            noisy_inputs += delta

            self._evaluate(outputs=self.model(noisy_inputs), loss=0.00, i=i)
            if self.is_terminated and is_terminate_when_success:
                break

        self._save_results(noise=noise, noisy_inputs=noisy_inputs)

    def _pick_one_data(self):
        index = 5
        self.inputs = self.inputs[index].unsqueeze(0)
        self.labels = self.labels[index].unsqueeze(0)
        self.true_labels = self.true_labels[index].unsqueeze(0)


if __name__ == '__main__':
    for model in ['model', 'pdg_model_8', 'pdg_model_16']:
        attacker = GTSRBPDGAttacker(target_class=5, model_filename=model)
        attacker.run(iteration=40, is_terminate_when_success=True)
