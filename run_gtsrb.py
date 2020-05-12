from src.trains.gtsrb import train
import sys

args = sys.argv
print('Batch Size: {0} | Epoch: {1}'.format(args[1], args[2]))
train(batch_size=int(args[1]), epochs=int(args[2]))
