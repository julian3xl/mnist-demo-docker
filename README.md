# MNIST demo docker

The aim of this repo is provide a deep learning demo docker to try the basis: training, evaluation and inference using the MNIST dataset.

## MNIST dataset

The MNIST database is available at http://yann.lecun.com/exdb/mnist/

The MNIST database is a dataset of handwritten digits. It has 60,000 training samples, and 10,000 test samples. Each image is represented by 28x28 pixels, each containing a value 0 - 255 with its grayscale value.

It is a subset of a larger set available from NIST.
The digits have been size-normalized and centered in a fixed-size image.

It is a good database for people who want to try learning techniques and pattern recognition
methods on real-world data while spending minimal efforts on preprocessing and formatting.

## Training

Running the command below you will get a trained model into the data folder.
```
$ docker run -ti -v $(pwd)/data:/data julian3xl/mnist-demo-docker:latest train.py
$ ls -l data/
```

You can tweak a little bit the training with the following env variables (-e docker's arg, Example: docker run -e MNIST_EPOCHS=1 ...):
```
MNIST_N_SAMPLES_TRAINING (defaults to 1000): number of training samples to be used during the training

MNIST_N_SAMPLES_TESTING (defaults to 1000): number of validation samples to be used during the training

MNIST_N_FOLDS (defaults to 2): number of splits to perform cross-validation

MNIST_EPOCHS (defaults to 1): number of epochs to run

MNIST_BATCH_SIZE (defaults to 32): number of samples to train on every batch
```

## Evaluation

Running the command below you will get the accuracy calculated using the testing dataset.
```
$ docker run -ti -v $(pwd)/data:/data julian3xl/mnist-demo-docker:latest evaluate.py
```

You can tweak a little bit the training with the following env variables:
```
MNIST_N_SAMPLES_TESTING (defaults to 1000): number of validation samples to be used during the training
```

## Inference

Running the command below you will get a prediction for the image provided.
```
 $ docker run -ti -v $(pwd)/data:/data julian3xl/mnist-demo-docker:latest infer.py
```

You can tweak a little bit the training with the following env variables:
```
MNIST_IMAGE_URI (defaults to a 7 image): uri of a image to be inferenced
```

## Docker build
To build the docker image in your local run the command below.
```
$ docker build -t julian3xl/mnist-demo-docker:latest .
```
