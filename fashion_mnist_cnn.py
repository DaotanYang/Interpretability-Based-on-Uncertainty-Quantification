'''Trains a simple convnet on the MNIST dataset.

Gets to 99.25% test accuracy after 12 epochs
(there is still a lot of margin for parameter tuning).
16 seconds per epoch on a GRID K520 GPU.


Note that this file is based on the KERAS mnist example
https://github.com/keras-team/keras/blob/master/examples/mnist_cnn.py
which is also MIT licensed. THere are only minor changes
'''

from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

import src.utilities as U
import src.utilities2 as U2

batch_size = 128        # 批量，一次训练所选取的样本数
num_classes = 10        # 数据类型0-9，共10种
epochs = 50             # 训练轮数

# input image dimensions，28x28的图片
img_rows, img_cols = 28, 28

x_train, y_train, x_test, y_test = U2.get_fashion_mnist()
input_shape = (img_rows, img_cols, 1)

# 构建CNN
model = Sequential()                                    # Sequential模型，顺序模型，Sequential模型的核心操作是添加layers
model.add(Conv2D(32, kernel_size=(3, 3),                # 卷积层，Conv2D创建二维卷积层，filters: 整数，输出空间的维度 （即卷积中滤波器的数量）。
                 # 这里filters=32意味着输出后的数据由[28,28,1](input_shape)变为[26,26,32]。这个参数只影响了1变为32.
                 # kernel_size:一个整数或2个整数的元组/列表，指定二维卷积窗口的高度和宽度。即每9个映射成1个，上面的26从这里来的、26=28-3+1.
                 activation='relu',                     # activation激活函数，这里用的relu激活函数。
                 input_shape=input_shape))              # 此层作为第一层需要input_shape参数。
model.add(Conv2D(64, (3, 3), activation='relu'))        # 卷积层，输出为[24,24,64]
model.add(MaxPooling2D(pool_size=(2, 2)))               # 最大池化层。pool_size：长为2的整数tuple，代表在两个方向（竖直，水平）上的下采样因子，如取（2，2）将使图片在两个维度上均变为原长的一半,此步输出为[12,12,64]
model.add(Dropout(0.25))                                # Dropout层，为输入数据施加Dropout。Dropout将在训练过程中每次更新参数时随机断开一定百分比（p）的输入神经元连接，Dropout层用于防止过拟合。
model.add(Flatten())                                    # 展平层，Flatten层用来将输入“压平”，即把多维的输入一维化，常用在从卷积层到全连接层的过渡。此步输出为[12x12x64]
model.add(Dense(128, activation='relu'))                # 全连接层，output = activation(dot(input, kernel)+bias)，此步输出为[128]
# activation是逐元素计算的激活函数，kernel是本层的权值矩阵，bias为偏置向量，只有当use_bias=True才会添加。
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))     # 全连接层，num_classes=10，此步输出为[10]

# 在模型训练之前，需要通过compile对模型进行编译
model.compile(loss=keras.losses.categorical_crossentropy,   # 目标函数，这里是多类的对数损失函数
              optimizer=keras.optimizers.Adadelta(),        # 优化器，该优化器学习率为1
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
fname = U.gen_save_name('save/fashion_mnist_cnn_run.h5')
model.save(fname)
