from random import Random
from sklearn.model_selection import train_test_split
import tensorflow as tf
import keras
import numpy as np
import os
from keras import backend as K
from keras.datasets import mnist
import itertools as itr
from functools import reduce
import operator
import re
from PIL import Image
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.models import Sequential
from keras.datasets import cifar10, mnist, fashion_mnist

# padding mnist 28x28->32x32
def mnist_reshape_28(_batch):
 
    batch = np.reshape(_batch,[-1,32,32])
    num = batch.shape[0]
    batch_28 = np.array(np.random.rand(num,28,28),dtype=np.float32)

    for i in range(num):
        for j in range(28):
            for k in range(28):
                batch_28[i][j][k] = batch[i][j+2][k+2]

    return batch_28

def get_cifar10():
    num_classes = 10            # 数据类型，10类不同物体或动物
    img_rows, img_cols = 32, 32
    # 数据，切分为训练和测试集。
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    if K.image_data_format() == 'channels_first':
        x_train = x_train.reshape(x_train.shape[0], 3, img_rows, img_cols)
        x_test = x_test.reshape(x_test.shape[0], 3, img_rows, img_cols)
    else:
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 3)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 3)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    # 将类向量转换为二进制类矩阵。
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    return x_train, y_train, x_test, y_test


def get_fashion_mnist():
    """
    Return the mnist data, scaled to [0,1].
    """
    num_classes = 10
    img_rows, img_cols = 28, 28
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

    if K.image_data_format() == 'channels_first':
        x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
        x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    else:
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    return x_train, y_train, x_test, y_test


def get_adv_mnist():
    """
    Return the mnist data, scaled to [0,1].
    """
    num_classes = 10
    img_rows, img_cols = 32, 32
    X = np.load('MNIST_adv_samples/DeepFool_mnist_image.npy')
    y = np.load('MNIST_adv_samples/DeepFool_mnist_label.npy')

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=3)
    
    # (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

    if K.image_data_format() == 'channels_first':
        x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
        x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    else:
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)

    x_train = mnist_reshape_28(x_train)
    x_test = mnist_reshape_28(x_test)
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    return x_train, y_train, x_test, y_test
