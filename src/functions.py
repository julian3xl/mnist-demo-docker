import requests
from keras.datasets import mnist
from keras.layers import Conv2D, Dense, Flatten, MaxPooling2D
from keras.models import Sequential, load_model as keras_load_model
from keras.optimizers import SGD
from keras.preprocessing.image import img_to_array, load_img
from keras.utils import to_categorical
from matplotlib import pyplot
from numpy import mean, std
from sklearn.model_selection import KFold


DATA_FOLDER = '/data'

# load train and test dataset
def load_dataset(n_train=None, n_test=None):
  # load dataset
  (trainX, trainY), (testX, testY) = mnist.load_data()
  n_trainX = n_train if n_train else len(trainX)
  n_trainY = n_train if n_train else len(trainY)
  n_testX = n_test if n_test else len(testX)
  n_testY = n_test if n_test else len(testY)

  # reshape dataset to have a single channel
  trainX = trainX.reshape((trainX.shape[0], 28, 28, 1))
  testX = testX.reshape((testX.shape[0], 28, 28, 1))

  # one hot encode target values
  trainY = to_categorical(trainY)
  testY = to_categorical(testY)

  return trainX[0:n_trainX], trainY[0:n_trainY], testX[0:n_testX], testY[0:n_testY]

# scale pixels and return normalized images
def prep_pixels(train, test):
  # convert from integers to floats
  train_norm = train.astype('float32')
  test_norm = test.astype('float32')

  # normalize to range 0-1
  train_norm = train_norm / 255.0
  test_norm = test_norm / 255.0

  return train_norm, test_norm

# define cnn model
def define_model():
  model = Sequential()
  model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(28, 28, 1)))
  model.add(MaxPooling2D((2, 2)))
  model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform'))
  model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform'))
  model.add(MaxPooling2D((2, 2)))
  model.add(Flatten())
  model.add(Dense(100, activation='relu', kernel_initializer='he_uniform'))
  model.add(Dense(10, activation='softmax'))

  # compile model
  opt = SGD(lr=0.01, momentum=0.9)
  model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

  return model

# evaluate a model using k-fold cross-validation
def evaluate_model(dataX, dataY, n_folds=2, epochs=1, batch_size=32):
  scores, histories = list(), list()

  # prepare cross validation
  kfold = KFold(n_folds, shuffle=True, random_state=1)

  # enumerate splits
  for train_ix, test_ix in kfold.split(dataX):
    # define model
    model = define_model()

    # select rows for train and test
    trainX, trainY, testX, testY = dataX[train_ix], dataY[train_ix], dataX[test_ix], dataY[test_ix]

    # fit model
    history = model.fit(trainX, trainY, epochs, batch_size, validation_data=(testX, testY), verbose=0)

    # evaluate model
    _, acc = model.evaluate(testX, testY, verbose=0)

    # stores scores
    scores.append(acc)
    histories.append(history)

  return model, scores, histories

# plot diagnostic learning curves
def summarize_diagnostics(histories):
  for i in range(len(histories)):
    # plot loss
    pyplot.subplot(2, 1, 1)
    pyplot.title('Cross Entropy Loss')
    pyplot.plot(histories[i].history['loss'], color='blue', label='train')
    pyplot.plot(histories[i].history['val_loss'], color='orange', label='test')

    # plot accuracy
    pyplot.subplot(2, 1, 2)
    pyplot.title('Classification Accuracy')
    pyplot.plot(histories[i].history['accuracy'], color='blue', label='train')
    pyplot.plot(histories[i].history['val_accuracy'], color='orange', label='test')

  pyplot.show()
  pyplot.savefig('%s/summary_diagnostics.png' % DATA_FOLDER)

# summarize model performance
def summarize_performance(scores):
  # print summary
  print('\naccuracy: mean=%.3f std=%.3f, n=%d' % (mean(scores)*100, std(scores)*100, len(scores)))

  # box and whisker plots of results
  pyplot.boxplot(scores)
  pyplot.show()
  pyplot.savefig('%s/summary_performance.png' % DATA_FOLDER)

# download sample image
def download_sample_image(uri='https://machinelearningmastery.com/wp-content/uploads/2019/02/sample_image.png', filename='sample.png'):
  print('\ndownloading %s...' % uri)

  image_path = '%s/%s' % (DATA_FOLDER, filename)

  with open(image_path, 'wb') as f:
    r = requests.get(uri, allow_redirects=True)
    f.write(r.content)

  return image_path

# load and prepare the image
def load_image(filename):
  # load the image
  img = load_img(filename, target_size=(28, 28))

  # convert to array
  img = img_to_array(img)

  # convert image to grayscale
  img = img[:,:,0]

  # reshape into a single sample with 1 channel
  img = img.reshape(1, 28, 28, 1)

  # prepare pixel data
  img = img.astype('float32')
  img = img / 255.0

  return img

# load model
def load_model(filename='model.h5'):
  return keras_load_model('%s/%s' % (DATA_FOLDER, filename))

# save model
def save_model(model, filename='model.h5'):
  model.save('%s/%s' % (DATA_FOLDER, filename))
