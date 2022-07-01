# -*- coding: utf-8 -*-
# @Time    : 2018/12
# @Author  : wengfutian
# @Email   : wengfutian@csu.edu.cn
from __future__ import print_function


import numpy as np
from keras.utils import plot_model
from keras.models import Model
from keras.layers import Input, Dense, BatchNormalization, Conv2D, MaxPooling2D, AveragePooling2D, ZeroPadding2D, Dropout
from keras.layers import add, Flatten

seed = 7
np.random.seed(seed)

def Conv2d_BN(x, nb_filter, kernel_size, strides=(1, 1), padding='same', name=None):
    if name is not None:
        bn_name = name + '_bn'
        conv_name = name + '_conv'
    else:
        bn_name = None
        conv_name = None

    x = Conv2D(nb_filter, kernel_size, padding=padding, strides=strides, activation='relu', name=conv_name)(x)
    x = BatchNormalization(axis=3, name=bn_name)(x)
    return x

def Conv_Block(inpt, nb_filter, kernel_size, strides=(1, 1), with_conv_shortcut=False):
    x = Conv2d_BN(inpt, nb_filter=nb_filter, kernel_size=kernel_size, strides=strides, padding='same')
    x = Conv2d_BN(x, nb_filter=nb_filter, kernel_size=kernel_size, padding='same')
    if with_conv_shortcut:
        shortcut = Conv2d_BN(inpt, nb_filter=nb_filter, strides=strides, kernel_size=kernel_size)
        x = add([x, shortcut])
        return x
    else:
        x = add([x, inpt])
        return x

class Resnet34:
    @staticmethod
    def build(width, height, depth, NB_CLASS):
        inpt = Input(shape=(height, width, depth))
        x = ZeroPadding2D((3, 3))(inpt)
        x = Conv2d_BN(x, nb_filter=64, kernel_size=(7, 7), strides=(2, 2), padding='valid')
        x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)
        x = Dropout(0.25)(x, training=True)
        # (56,56,64)
        x = Conv_Block(x, nb_filter=64, kernel_size=(3, 3))
        x = Conv_Block(x, nb_filter=64, kernel_size=(3, 3))
        x = Conv_Block(x, nb_filter=64, kernel_size=(3, 3))
        # (28,28,128)
        x = Conv_Block(x, nb_filter=128, kernel_size=(3, 3), strides=(2, 2), with_conv_shortcut=True)
        x = Conv_Block(x, nb_filter=128, kernel_size=(3, 3))
        x = Conv_Block(x, nb_filter=128, kernel_size=(3, 3))
        x = Conv_Block(x, nb_filter=128, kernel_size=(3, 3))
        # (14,14,256)
        x = Conv_Block(x, nb_filter=256, kernel_size=(3, 3), strides=(2, 2), with_conv_shortcut=True)
        x = Conv_Block(x, nb_filter=256, kernel_size=(3, 3))
        x = Conv_Block(x, nb_filter=256, kernel_size=(3, 3))
        x = Conv_Block(x, nb_filter=256, kernel_size=(3, 3))
        x = Conv_Block(x, nb_filter=256, kernel_size=(3, 3))
        x = Conv_Block(x, nb_filter=256, kernel_size=(3, 3))
        # (7,7,512)
        x = Conv_Block(x, nb_filter=512, kernel_size=(3, 3), strides=(2, 2), with_conv_shortcut=True)
        x = Conv_Block(x, nb_filter=512, kernel_size=(3, 3))
        x = Conv_Block(x, nb_filter=512, kernel_size=(3, 3))
        # x = AveragePooling2D(pool_size=(7, 7))(x)
        x = AveragePooling2D(pool_size=(1, 1))(x)
        x = Dropout(0.5)(x, training=True)
        x = Flatten()(x)
        x = Dense(NB_CLASS, activation='softmax')(x)

        # Create a Keras Model
        model = Model(inputs=inpt, outputs=x)
        model.summary()
        # Save a PNG of the Model Build
        #plot_model(model, to_file='../imgs/Resnet34.png')
        # return the constructed network architecture
        return model