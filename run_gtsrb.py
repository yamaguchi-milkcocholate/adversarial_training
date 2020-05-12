from src.trains.gtsrb import train
import sys

args = sys.argv
print('Batch Size: {0} | Epoch: {1}'.format(args[0], args[1]))
train(batch_size=int(args[0]), epochs=int(args[1]))
