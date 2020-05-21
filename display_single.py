from src.storage.resultsrepo import ResultsRepository
from PIL import Image

noise_32 = ResultsRepository.load_as_pickle(filename='gtsrb-pdg-noise-32')[0]
noisy_inputs_32 = ResultsRepository.load_as_pickle(filename='gtsrb-pdg-noisy-inputs-32')[0]
noise_256 = ResultsRepository.load_as_pickle(filename='gtsrb-pdg-noise-256')[0]
noisy_inputs_256 = ResultsRepository.load_as_pickle(filename='gtsrb-pdg-noisy-inputs-256')[0]

noise_32 = Image.fromarray(noise_32)
noisy_inputs_32 = Image.fromarray(noisy_inputs_32)
noise_256 = Image.fromarray(noise_256)
noisy_inputs_256 = Image.fromarray(noisy_inputs_256)

noise_32.show()
noisy_inputs_32.show()
noise_256.show()
noisy_inputs_256.show()
