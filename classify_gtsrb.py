from src.storage.datasetrepo import GTSRBRepository
from src.utils.imgpro import *
from src.storage.modelrepo import ModelRepository
from src.trains.models import GTSRBCNN
from src.utils.stats import success_rate
import random
import torch


x_stop = GTSRBRepository.load_from_images(dir_name='victim-set')
x_stop = resize(img=x_stop, size=(32, 32))
x_stop = scale_gtsrb(images=x_stop)
x_stop = x_stop.reshape((x_stop.shape[0], x_stop.shape[3], x_stop.shape[1], x_stop.shape[2]))
x_stop, y_stop = torch.tensor(x_stop, dtype=torch.float32), torch.tensor([14]*len(x_stop), dtype=torch.long)

_, _, _, _, x_test, y_test = GTSRBRepository.load_from_pickle_tf()
x_test = x_test.reshape((x_test.shape[0], x_test.shape[3], x_test.shape[1], x_test.shape[2]))
x_test, y_test = torch.tensor(x_test, dtype=torch.float), torch.tensor(y_test, dtype=torch.long)
# rand_index = random.sample(range(len(y_test)), 100)
# x_test, y_test = x_test[rand_index], y_test[rand_index]
x_test, y_test = x_test[y_test == 14], y_test[y_test == 14]
print(x_stop[0])
print(x_test[0])

device = torch.device('cpu')
model = GTSRBCNN()
model = ModelRepository.load(filename='GTSRB/model', device=device, model=model)
model.eval()

print('Stop: size={:d} accuracy={:.3f}'.format(len(x_stop), success_rate(outputs=model(x_stop), labels=y_stop)))
print('Test: size={:d} accuracy={:.3f}'.format(len(x_test), success_rate(outputs=model(x_test), labels=y_test)))
print(torch.max(model(x_stop).data, 1))
