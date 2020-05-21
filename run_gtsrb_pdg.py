from src.defenses.gtsrb import GTSRBAdversarialTraining
import sys

args = sys.argv
print('Batch Size: {0} | Epoch: {1} | LR: {2} WD: {3} PDG Iter: {4}'.format(args[1], args[2], args[3], args[4], args[5]))
training = GTSRBAdversarialTraining(
    batch_size=int(args[1]),
    lr=float(args[3]),
    wd=float(args[4]),
)
training.run(epochs=int(args[2]), pdg_iteration=int(args[5]))
