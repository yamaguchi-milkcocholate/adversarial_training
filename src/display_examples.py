import numpy as np
import matplotlib.pylab as plt
from src.storage.datasetrepo import GTSRBRepository
from src.utils.imgpro import rescale_gtsrb


_, _, _, _, x_test, y_test = GTSRBRepository.load_from_pickle_tf()

i_label = 0
class_num = len(np.unique(y_test))
col = 10
row = int(class_num / col) if class_num % col == 0 else int(class_num / col) + 1
f, ax_arr = plt.subplots(row, col)

for i in range(len(y_test)):
    if i_label == y_test[i]:
        ax_arr[int(i_label / col), i_label % col].imshow(rescale_gtsrb(images=x_test[i]))
        ax_arr[int(i_label / col), i_label % col].set_title(str(i_label + 1))
        ax_arr[int(i_label / col), i_label % col].axis('off')
        i_label += 1

for i in range(class_num % col, col):
    ax_arr[row-1][i].imshow(np.ones((32, 32, 3)))
    ax_arr[row-1][i].axis('off')

plt.show()
