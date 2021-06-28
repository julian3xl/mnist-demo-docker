import numpy as np
import os
from functions import *

if __name__ == "__main__":
  image_uri = os.environ.get('MNIST_IMAGE_URI', 'https://machinelearningmastery.com/wp-content/uploads/2019/02/sample_image.png')

  # download sample image
  image_path = download_sample_image(image_uri)

  # load the image
  img = load_image(image_path)

  # load model
  model = load_model()

  # predict the class
  digit = np.argmax(model.predict(img), axis=-1)

  print('\ninference prediction: %s' % digit[0])
