import matplotlib.pylab as plt
from src.storage.datasetrepo import GTSRBRepository
from src.storage.resultsrepo import ResultsRepository
from src.utils.imgpro import *

_, _, _, _, x_test, y_test = GTSRBRepository.load_from_pickle_tf()

i_label = 0
class_num = len(np.unique(y_test))
col = 10
row = int(class_num / col) if class_num % col == 0 else int(class_num / col) + 1
_, ax_test = plt.subplots(row, col)

for i in range(len(y_test)):
    if i_label == y_test[i]:
        ax_test[int(i_label / col), i_label % col].imshow(rescale_gtsrb(images=x_test[i]))
        ax_test[int(i_label / col), i_label % col].set_title(str(i_label + 1))
        ax_test[int(i_label / col), i_label % col].axis('off')
        i_label += 1

for i in range(class_num % col, col):
    ax_test[row-1][i].imshow(np.ones((32, 32, 3)))
    ax_test[row-1][i].axis('off')


def _plot_stop(data, col, row):
    _, ax_stop = plt.subplots(row, col)
    for i in range(len(data)):
        ax_stop[int(i / col), i % col].imshow(data[i])
        ax_stop[int(i / col), i % col].set_title(str(i + 1))
        ax_stop[int(i / col), i % col].axis('off')


x_test = x_test[y_test == 14][:40]
x_test = rescale_gtsrb(images=x_test)
row = int(len(x_test) / col) if len(x_test) % col == 0 else int(len(x_test) / col) + 1
_plot_stop(data=x_test, col=col, row=row)

x_stop = GTSRBRepository.load_from_images(dir_name='victim-set')
row = 4

# Original Images
_plot_stop(data=x_stop, col=col, row=row)

# Input Data
x_stop = scale_gtsrb(images=resize(img=x_stop, size=(32, 32)))
x_stop += 0.5
_plot_stop(data=x_stop, col=col, row=row)

# 32 x 32 adversarial examples
noisy_inputs_32 = ResultsRepository.load_as_pickle(filename='gtsrb-noisy-inputs-32')
_plot_stop(data=noisy_inputs_32, col=col, row=row)

# 256 x 256 adversarial examples
noisy_inputs_256 = ResultsRepository.load_as_pickle(filename='gtsrb-noisy-inputs-256')
_plot_stop(data=noisy_inputs_256, col=col, row=row)

plt.show()

