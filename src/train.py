import os
from functions import *

if __name__ == "__main__":
  n_train = os.environ.get('MNIST_N_SAMPLES_TRAINING', None)
  n_test = os.environ.get('MNIST_N_SAMPLES_TESTING', None)
  n_folds = int(os.environ.get('MNIST_N_FOLDS', 5))
  epochs = int(os.environ.get('MNIST_EPOCHS', 10))
  batch_size = int(os.environ.get('MNIST_BATCH_SIZE', 32))

  # load dataset
  trainX, trainY, testX, testY = load_dataset(n_train, n_test)

  # prepare pixel data
  trainX, testX = prep_pixels(trainX, testX)

  # evaluate model
  model, scores, histories = evaluate_model(trainX, trainY, n_folds, epochs, batch_size)

  # learning curves
  summarize_diagnostics(histories)

  # summarize estimated performance
  summarize_performance(scores)

  # save model
  save_model(model)
