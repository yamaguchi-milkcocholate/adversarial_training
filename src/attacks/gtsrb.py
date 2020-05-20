from typing import List
from src.storage.modelrepo import ModelRepository
from src.storage.datasetrepo import GTSRBRepository
from src.trains.models import GTSRBCNN
from src.utils.imgpro import *
from src.utils.stats import success_rate
from src.attacks.criterion import PhysicalRobustCriterion


def attack(iteration: int):
    device = torch.device('cpu')
    model = GTSRBCNN()
    model = ModelRepository.load(filename='GTSRB/model', device=device, model=model)
    model.eval()
    true_class = 14
    target_class = 5
    input_size = (32, 32)
    inputs, mask = _prepare_inputs()
    labels = torch.tensor(np.array([target_class] * len(inputs)), dtype=torch.long)
    true_labels = torch.tensor(np.array([true_class] * len(inputs)), dtype=torch.long)
    # Todo: initial noise is 0.0 which is different from the original code.
    noise = torch.tensor(np.random.normal(0.0, 1.0, inputs.shape[1:])*0.0, dtype=torch.float32, requires_grad=True)

    criterion = PhysicalRobustCriterion()
    optimizer = torch.optim.Adam([noise])  # Todo: lr & eps are different

    for i in range(iteration):
        optimizer.zero_grad()
        noisy_inputs = add_noise(inputs=inputs, noise=noise, mask=mask)
        outputs = model(resize(img=noisy_inputs, size=input_size))
        loss = criterion(outputs, labels, noise=noise)
        loss.backward()
        optimizer.step()

        print('[{:d}/{:d}] loss: {:.3f} attack_sr: {:.3f} normal_acc: {:.3f}'.format(
            i+1,
            iteration,
            loss.item(),
            success_rate(outputs=outputs, labels=labels),
            success_rate(outputs=outputs, labels=true_labels)
        ))


def _prepare_inputs() -> List[torch.Tensor]:
    ori_data = GTSRBRepository.load_from_images(dir_name='victim-set')
    ori_data = ori_data.reshape((ori_data.shape[0], ori_data.shape[3], ori_data.shape[1], ori_data.shape[2]))
    mask = GTSRBRepository.load_from_images(dir_name='masks')[0]
    mask = mask.reshape(mask.shape[2], mask.shape[0], mask.shape[1])

    ori_data = scale_gtsrb(images=ori_data)
    mask = mask.astype(np.float32)

    ori_data_ts = torch.tensor(ori_data, dtype=torch.float, requires_grad=True)
    mask_ts = torch.tensor(mask, dtype=torch.float, requires_grad=True)
    return ori_data_ts, mask_ts


if __name__ == '__main__':
    attack(iteration=10)
