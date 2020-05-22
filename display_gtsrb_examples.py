import matplotlib.pylab as plt
import os
import numpy as np
from src.storage.datasetrepo import GTSRBRepository
from src.storage.resultsrepo import ResultsRepository
from src.utils.imgpro import *


def _plot_stop(data, col, row):
    _, ax_stop = plt.subplots(row, col)
    for i in range(len(data)):
        ax_stop[int(i / col), i % col].imshow(data[i])
        ax_stop[int(i / col), i % col].set_title(str(i + 1))
        ax_stop[int(i / col), i % col].axis('off')


def display(method: str):
    x_stop = GTSRBRepository.load_from_images(dir_name='victim-set')
    row = 4
    col = 10

    # Original Images
    _plot_stop(data=x_stop, col=col, row=row)

    # Input Data
    x_stop = scale_gtsrb(images=resize(img=x_stop, size=(32, 32)))
    x_stop += 0.5
    _plot_stop(data=x_stop, col=col, row=row)

    for model in ['model']:
        # 32 x 32 adversarial examples
        noisy_inputs_32 = ResultsRepository.load_as_pickle(
            filename=os.path.join(method+'&'+model, 'gtsrb-noisy-inputs-32'))
        _plot_stop(data=noisy_inputs_32, col=col, row=row)

        # 256 x 256 adversarial examples
        noisy_inputs_256 = ResultsRepository.load_as_pickle(
            filename=os.path.join(method+'&'+model, 'gtsrb-noisy-inputs-256'))
        _plot_stop(data=noisy_inputs_256, col=col, row=row)

    plt.show()


def plot_success_rate(method: str):
    for model in ['model', 'pdg_model_8', 'pdg_model_16']:
        history = ResultsRepository.load_as_pickle(filename=os.path.join(method+'&'+model, 'history'))
        plt.plot(np.arange(len(history['adv_sr'])), history['adv_sr'], label=model)
    plt.legend()
    plt.xlabel('#Iteration')
    plt.ylabel('Adv. success rate(%)')
    plt.show()


if __name__ == '__main__':
    # display(method='gtsrb')
    display(method='pdg')
    # plot_success_rate(method='gtsrb')
    # plot_success_rate(method='pdg')
