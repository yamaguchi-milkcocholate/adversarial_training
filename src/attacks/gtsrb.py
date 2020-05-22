from __future__ import annotations
from typing import List
from src.storage.modelrepo import ModelRepository
from src.storage.datasetrepo import GTSRBRepository
from src.trains.models import GTSRBCNN
from src.utils.imgpro import *
from src.utils.stats import success_rate
from src.attacks.tools.criterion import PhysicalRobustCriterion
from src.storage.resultsrepo import ResultsRepository
import os


class GTSRBAttacker:
    METHOD = 'gtsrb'

    def __init__(self, target_class: int, model_filename: str):
        self.device = torch.device('cpu')
        self.results_folder_name = self.METHOD+'&'+model_filename
        self.model = GTSRBCNN()
        self.model = ModelRepository.load(
            filename=os.path.join('GTSRB', model_filename), device=self.device, model=self.model
        )
        self.model.eval()
        self.target_class = target_class
        self.true_class = 14
        self.input_size = (32, 32)
        self.image_size = (256, 256)
        self.input_range = (-0.5, 0.5)
        self.inputs, self.mask = self._prepare_dataset()
        self.labels = torch.tensor(np.array([self.target_class] * len(self.inputs)), dtype=torch.long)
        self.true_labels = torch.tensor(np.array([self.true_class] * len(self.inputs)), dtype=torch.long)
        self.iteration = None
        self.is_terminated = None
        self.results_filename = {
            '32': {
                'noise': os.path.join(self.results_folder_name, 'gtsrb-noise-32'),
                'noisy_inputs': os.path.join(self.results_folder_name, 'gtsrb-noisy-inputs-32')
            },
            '256': {
                'noise': os.path.join(self.results_folder_name, 'gtsrb-noise-256'),
                'noisy_inputs': os.path.join(self.results_folder_name, 'gtsrb-noisy-inputs-256')
            },
            'overview': os.path.join(self.results_folder_name, 'overview'),
            'history': os.path.join(self.results_folder_name, 'history'),
        }
        self.history = {
            'adv_loss': list(), 'adv_sr': list(), 'nor_sr': list(),
        }

    def run(self, iteration, is_terminate_when_success=False):
        self.iteration = iteration
        self.is_terminated = False
        # Todo: initial noise is 0.0 which is different from the original code.
        noise = torch.tensor(np.random.normal(0.0, 1.0, self.mask.shape) * 0.0, dtype=torch.float32, requires_grad=True)
        noisy_inputs = add_noise(inputs=self.inputs, noise=noise, mask=self.mask)
        self._evaluate(outputs=self.model(noisy_inputs), loss=0.0, i=-1)

        criterion = PhysicalRobustCriterion()
        optimizer = torch.optim.Adam([noise])  # Todo: lr & eps are different

        for i in range(self.iteration):
            optimizer.zero_grad()
            outputs = self.model(noisy_inputs)
            loss = criterion(outputs, self.labels, noise=noise, mask=self.mask)
            loss.backward()
            optimizer.step()
            noisy_inputs = add_noise(inputs=self.inputs, noise=noise, mask=self.mask)

            self._evaluate(outputs=outputs, loss=loss.item(), i=i)
            if self.is_terminated and is_terminate_when_success:
                break

        self._save_results(noise=noise, noisy_inputs=noisy_inputs)

    def _evaluate(self, outputs, loss, i):
        adv_sr = success_rate(outputs=outputs, labels=self.labels)
        nor_sr = success_rate(outputs=outputs, labels=self.true_labels)
        print('[{:d}/{:d}] adv_loss: {:.3f} adv_sr: {:.3f} nor_acc: {:.3f}'.format(
            i + 1,
            self.iteration,
            loss,
            adv_sr,
            nor_sr
        ))
        self.history['adv_loss'].append(loss)
        self.history['adv_sr'].append(adv_sr)
        self.history['nor_sr'].append(nor_sr)
        if adv_sr == 100.0:
            self.is_terminated = True

    def _prepare_dataset(self) -> List[torch.Tensor]:
        ori_data = GTSRBRepository.load_from_images(dir_name='victim-set')
        ori_data = resize(img=ori_data, size=self.input_size)
        ori_data = scale_gtsrb(images=ori_data)
        ori_data = reshape_to_torch(images=ori_data)
        ori_data_ts = torch.tensor(ori_data, dtype=torch.float, requires_grad=True)
        mask = GTSRBRepository.load_from_images(dir_name='masks')
        mask = resize(img=mask, size=self.input_size)
        mask = reshape_to_torch(images=mask)
        mask = mask.astype(np.float32)
        mask_ts = torch.tensor(mask, dtype=torch.float, requires_grad=True)
        return ori_data_ts, mask_ts

    def _save_results(self, noise: torch.Tensor, noisy_inputs: torch.Tensor):
        noise, noisy_inputs = noise.detach().numpy(), noisy_inputs.detach().numpy()
        noise, noisy_inputs = reshape_to_pil(images=noise), reshape_to_pil(images=noisy_inputs)
        noise_32, noisy_inputs_32 = rescale_gtsrb(images=noise), rescale_gtsrb(images=noisy_inputs)
        noise_256 = resize(img=noise_32, size=self.image_size).astype(np.uint8)
        noisy_inputs_256 = resize(img=noisy_inputs_32, size=self.image_size).astype(np.uint8)

        ResultsRepository.make_results_folder(folder_name=self.results_folder_name)
        ResultsRepository.save_as_pickle(filename=self.results_filename['32']['noise'], data=noise_32)
        ResultsRepository.save_as_pickle(filename=self.results_filename['32']['noisy_inputs'], data=noisy_inputs_32)
        ResultsRepository.save_as_pickle(filename=self.results_filename['256']['noise'], data=noise_256)
        ResultsRepository.save_as_pickle(filename=self.results_filename['256']['noisy_inputs'], data=noisy_inputs_256)
        ResultsRepository.save_as_pickle(filename=self.results_filename['history'], data=self.history)
        ResultsRepository.write_items(filename=self.results_filename['overview'], items=self._itemize_model_overview())

    def _itemize_model_overview(self) -> List[tuple]:
        return [
            ('iteration', self.iteration),
        ]


if __name__ == '__main__':
    for model in ['model', 'pdg_model_12', 'pdg_model_16']:
        attacker = GTSRBAttacker(target_class=5, model_filename=model)
        attacker.run(iteration=50, is_terminate_when_success=True)
