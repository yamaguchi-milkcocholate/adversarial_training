from __future__ import annotations
import torch
from src.attacks.tools.pgd import multi_step_attack
from src.attacks.gtsrb import GTSRBAttacker


class GTSRBPDGAttacker(GTSRBAttacker):

    def __init__(self, target_class: int):
        super().__init__(target_class=target_class)
        self.results_filename = {
            '32': {
                'noise': 'gtsrb-pdg-noise-32',
                'noisy_inputs': 'gtsrb-pdg-noisy-inputs-32'
            },
            '256': {
                'noise': 'gtsrb-pdg-noise-256',
                'noisy_inputs': 'gtsrb-pdg-noisy-inputs-256'
            }
        }

    def run(self, iteration: int):
        self.iteration = iteration
        self.is_terminated = False  # Todo: common run() method?
        self._pick_one_data()
        noise = torch.zeros_like(self.inputs)
        noisy_inputs = self.inputs + noise
        for i in range(iteration):
            delta = multi_step_attack(
                model=self.model, X=self.inputs, y=self.labels, input_range=self.input_range,
                epsilon=8/255, alpha=2/255, num_iter=40, randomize=True
            )
            noise += delta
            # noisy_inputs += delta

            self._evaluate(outputs=self.model(noisy_inputs), loss=0.00, i=i)
            if self.is_terminated:
                break

        self._save_results(noise=noise, noisy_inputs=noisy_inputs)

    def _pick_one_data(self):
        index = 2
        self.inputs = self.inputs[index].unsqueeze(0)
        self.labels = self.labels[index].unsqueeze(0)
        self.true_labels = self.true_labels[index].unsqueeze(0)


if __name__ == '__main__':
    attacker = GTSRBPDGAttacker(target_class=5)
    attacker.run(iteration=15)
