'''Trains a simple convnet on the MNIST dataset.

Gets to 99.25% test accuracy after 12 epochs
(there is still a lot of margin for parameter tuning).
16 seconds per epoch on a GRID K520 GPU.


Note that this file is based on the KERAS mnist example
https://github.com/keras-team/keras/blob/master/examples/mnist_cnn.py
which is also MIT licensed. THere are only minor changes
'''

from __future__ import print_function
from re import A
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from numpy import deprecate_with_doc

import src.utilities as U
import src.utilities2 as U2
import src.alenet as Alenet
import src.googlenet as Gonet
import src.lenet as Lenet
import src.lenet_dropout as Lenet_D
import src.resnet34 as Resnet
import src.resnet34_dropout as Resnet_D
import src.vgg16 as Vgg


batch_size = 128        # 批量，一次训练所选取的样本数
num_classes = 10        # 数据类型0-9，共10种
epochs = 30             # 训练轮数,ResNet和AlexNet为30，LeNet为100

# input image dimensions，28x28的图片
img_rows, img_cols = 28, 28

# the data, shuffled and split between train and test sets，划分训练集和测试集
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# mnist.load_data函数返回的四个参数形状如下所示：
# x_train.shape == (60000, 28, 28)      x_train是训练集数据
# x_test.shape == (10000, 28, 28)       x_test是测试集数据
# y_train.shape == (60000,)             y_train是训练集标签
# y_test.shape == (10000,)              y_test是测试集标签

# channel_first代表数据通道维在前面，channel_last通道维度在后面
# 注意reshape函数中，第二(或四)个参数为1，即代表MNIST数据集为单通道黑白图象。
if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)      # 这里reshape多了1，就是多了个括号而已。
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)                                   # 代表28x28的灰度图,[1,28,28]
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)                                   # 代表28x28的灰度图,[28,28,1],本代码用的这个，属于channel_last.

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)          # 输出结果为：(60000, 28, 28, 1)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices ，将类别向量转为二进制的矩阵表示
# 将y_train和y_test按照10个类别(数字0-9)来进行转换
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# 从src中直接导入模型结构，其中AlexNet自带Dropout层，不需要额外添加。
# ResNet和LeNet中不含Dropout层，需要增加Dropout层。
# 使用LeNet时，epoch建议设置为100
# model = Resnet.Resnet34.build(img_rows, img_cols, 1, num_classes)
# model = Resnet_D.Resnet34.build(img_rows, img_cols, 1, num_classes)
# model = Alenet.AlexNet.build(img_rows, img_cols, 1, num_classes)
# model = Lenet.LeNet.build(img_rows, img_cols, 1, num_classes)
model = Lenet_D.LeNet.build(img_rows, img_cols, 1, num_classes)

# 在模型训练之前，需要通过compile对模型进行编译
model.compile(loss=keras.losses.categorical_crossentropy,   # 目标函数，这里是多类的对数损失函数
              optimizer=keras.optimizers.Adadelta(1),        # 优化器，该优化器学习率默认为1，限于LeNet和ResNet
            #   optimizer=keras.optimizers.Adam(0.001),           # 优化器，仅限于使用AlexNet时
              metrics=['accuracy'])                         # 列表，包含评估模型在训练和测试时的性能的指标


# 使用fit对模型进行训练
model.fit(x_train, y_train,
          batch_size=batch_size,                    # 每次训练的样本个数，本代码中每次训练128个，每轮60000个
          epochs=epochs,                            # 训练总轮数
          verbose=1,                                # 日志显示，0为不在标准输出流输出日志信息，1为输出进度条记录，2为每个epoch输出一行记录
          validation_data=(x_test, y_test))         # 形式为（X，y）的tuple，是指定的验证集。此参数将覆盖validation_spilt。
score = model.evaluate(x_test, y_test, verbose=0)   # 本函数按batch计算在某些输入数据上模型的误差，本函数返回一个测试误差的标量值（如果模型没有其他评价指标），或一个标量的list（如果模型还有其他的评价指标）
print('Test loss:', score[0])
print('Test accuracy:', score[1])
# fname = U.gen_save_name('save/mnist_cnn_run_lenet_dropout.h5')
# model.save(fname)
