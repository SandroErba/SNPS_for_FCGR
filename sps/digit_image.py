from keras.src.datasets import mnist
import numpy as np
from sps.config import Config

def get_mnist_data():
    (train_data, train_label), (test_data, test_label) = mnist.load_data()
    train_data = train_data[:Config.TRAIN_SIZE]
    train_label = train_label[:Config.TRAIN_SIZE]
    test_data = test_data[:Config.TEST_SIZE]
    test_label = test_label[:Config.TEST_SIZE]
    train_q = train_data.astype(np.uint8)
    test_q = test_data.astype(np.uint8)
    if Config.INVERT:
        train_q = 255 - train_q
        test_q = 255 - test_q
    return train_q, train_label, test_q, test_label