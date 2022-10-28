import numpy as np
from keras.utils import get_file

def load_mnist( ):
    # download MNIST
    path = get_file("mnist.npz", cache_subdir="/home/ubuntu/jupyter/BackdoorDemo/dataset", origin="https://s3.amazonaws.com/img-datasets/mnist.npz",)

    # Load
    dict_data = np.load(path)
    x_train = dict_data["x_train"]
    y_train = dict_data["y_train"]
    x_test = dict_data["x_test"]
    y_test = dict_data["y_test"]
    dict_data.close()

    min_, max_ = 0.0, 255.0
    return (x_train, y_train), (x_test, y_test), min_, max_



