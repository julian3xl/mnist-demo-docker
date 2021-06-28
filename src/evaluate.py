import os
from functions import *

if __name__ == "__main__":
  n_train = os.environ.get('MNIST_N_SAMPLES_TRAINING', None)
  n_test = os.environ.get('MNIST_N_SAMPLES_TESTING', None)

  # load dataset
  trainX, trainY, testX, testY = load_dataset(n_train, n_test)

  # evaluate final model
  model = load_model()

  # evaluate model on test dataset
  _, acc = model.evaluate(testX, testY, verbose=0)
  print('\naccuracy (using testing dataset): %.3f' % (acc * 100.0))
