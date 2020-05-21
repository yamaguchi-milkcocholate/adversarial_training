from typing import List
from src.storage.modelrepo import ModelRepository
from src.storage.datasetrepo import GTSRBRepository
from src.trains.models import GTSRBCNN
from src.utils.imgpro import *
from src.utils.stats import success_rate
from src.attacks.criterion import PhysicalRobustCriterion
from src.storage.resultsrepo import ResultsRepository


def attack(iteration: int):
    device = torch.device('cpu')
    model = GTSRBCNN()
    model = ModelRepository.load(filename='GTSRB/model', device=device, model=model)
    model.eval()
    true_class = 14
    target_class = 5

    inputs, mask = _prepare_inputs()
    labels = torch.tensor(np.array([target_class] * len(inputs)), dtype=torch.long)
    true_labels = torch.tensor(np.array([true_class] * len(inputs)), dtype=torch.long)
    # Todo: initial noise is 0.0 which is different from the original code.
    noise = torch.tensor(np.random.normal(0.0, 1.0, mask.shape)*0.0, dtype=torch.float32, requires_grad=True)
    noisy_inputs = add_noise(inputs=inputs, noise=noise, mask=mask)

    criterion = PhysicalRobustCriterion()
    optimizer = torch.optim.Adam([noise])  # Todo: lr & eps are different

    for i in range(iteration):
        optimizer.zero_grad()
        outputs = model(noisy_inputs)
        loss = criterion(outputs, labels, noise=noise)
        loss.backward()
        optimizer.step()
        noisy_inputs = add_noise(inputs=inputs, noise=noise, mask=mask)

        atc_sr = success_rate(outputs=outputs, labels=labels)
        print('[{:d}/{:d}] loss: {:.3f} attack_sr: {:.3f} normal_acc: {:.3f}'.format(
            i+1,
            iteration,
            loss.item(),
            atc_sr,
            success_rate(outputs=outputs, labels=true_labels)
        ))
        if atc_sr == 100.0:
            break

    _save_results(noise=noise, noisy_inputs=noisy_inputs)


def _prepare_inputs() -> List[torch.Tensor]:
    ori_data = GTSRBRepository.load_from_images(dir_name='victim-set')
    ori_data = resize(img=ori_data, size=(32, 32))
    ori_data = scale_gtsrb(images=ori_data)
    ori_data = reshape_to_torch(images=ori_data)
    ori_data_ts = torch.tensor(ori_data, dtype=torch.float, requires_grad=True)
    mask = GTSRBRepository.load_from_images(dir_name='masks')
    mask = resize(img=mask, size=(32, 32))
    mask = reshape_to_torch(images=mask)
    mask = mask.astype(np.float32)
    mask_ts = torch.tensor(mask, dtype=torch.float, requires_grad=True)
    return ori_data_ts, mask_ts


def _save_results(noise: torch.Tensor, noisy_inputs: torch.Tensor):
    noise, noisy_inputs = noise.detach().numpy(), noisy_inputs.detach().numpy()
    noise, noisy_inputs = reshape_to_pil(images=noise), reshape_to_pil(images=noisy_inputs)

    ResultsRepository.save_as_pickle(filename='gtsrb-noise-input-32', data=noise)
    ResultsRepository.save_as_pickle(filename='gtsrb-noisy-inputs-32', data=noisy_inputs)
    noise = rescale_gtsrb(images=noise)
    noisy_inputs = rescale_gtsrb(images=noisy_inputs)
    noise = resize(img=noise, size=(256, 256)).astype(np.uint8)
    noisy_inputs = resize(img=noisy_inputs, size=(256, 256)).astype(np.uint8)
    ResultsRepository.save_as_pickle(filename='gtsrb-noise-input-256', data=noise)
    ResultsRepository.save_as_pickle(filename='gtsrb-noisy-inputs-256', data=noisy_inputs)


if __name__ == '__main__':
    attack(iteration=15)
