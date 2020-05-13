from src.trains.gtsrb import train
import sys

args = sys.argv
print('Batch Size: {0} | Epoch: {1} | LR: {2} WD: {3}'.format(args[1], args[2], args[3], args[4]))
train(batch_size=int(args[1]), epochs=int(args[2]), lr=float(args[3]), wd=float(args[4]))
