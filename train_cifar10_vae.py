import keras
from tensorflow import optimizers
from keras import backend as K
from keras.models import Model
from keras.layers import Dense, Flatten, Lambda, Reshape, Conv2D, Conv2DTranspose
import tensorflow as tf
import src.utilities as U
import src.utilities2 as U2

tf.compat.v1.disable_eager_execution()

BATCH_SIZE = 256
img_rows, img_cols, img_chns = 32, 32, 3


def define_VAE(optim='adagrad', latent_dim=2):
    inputs = keras.layers.Input(shape=(32, 32, 3))  # encoder_input
    enc_1 = Conv2D(filters=3  , kernel_size=2, activation='elu', padding='same')(inputs)         # encoder，encoding过程
    enc_2 = Conv2D(filters=32 , kernel_size=2, activation='elu', strides=2, padding='same')(enc_1)
    enc_3 = Conv2D(filters=32 , kernel_size=3, activation='elu', padding='same')(enc_2)
    enc_4 = Conv2D(filters=32 , kernel_size=3, activation='elu', padding='same')(enc_3)
    enc_5 = Flatten()(enc_4)
    enc_6 = Dense(128, activation='elu')(enc_5)

    z_mu = Dense(latent_dim)(enc_6)             # z代表隐变量
    z_logsigma = Dense(latent_dim)(enc_6)       # g1=h1,此处mu与sigma的计算共享参数

    encoder = Model(inputs=inputs, outputs=z_mu)  # represent the latent space by the mean

    # 定义隐空间z的分布函数，z服从normal(mu,sigma)
    def sample_z(args):
        mu, logsigma = args
        # 这里使用了重参数化，σ*ζ+μ,其中ζ服从标准正态分布，在这一步就对z进行采样了
        # random_normal函数从服从标准正态分布的的序列中随机取shape形状的值
        return 0.5 * K.exp(logsigma / 2) * K.random_normal(shape=(K.shape(mu)[0], latent_dim)) + mu

    # 将表达式用Lambda函数封装为layer对象，封装的函数为sample_z，预计输出为(2,)，输入为[z_mu, z_logsigma]
    # 在这一步将encoder算出的μ和σ传入隐空间z的采样中。
    z = Lambda(sample_z, output_shape=(latent_dim,))([z_mu, z_logsigma])

    dec_input = keras.layers.Input(shape=(latent_dim,))             # decoder_input，decoding过程
    dec_1 = Dense(128, activation='elu')(dec_input)                 # decoder
    dec_2 = Dense(32 * img_rows / 2 * img_cols / 2, activation='relu')(dec_1)

    if K.image_data_format() == 'channels_first':
        output_shape = (BATCH_SIZE, 32, 16, 16)
    else:
        output_shape = (BATCH_SIZE, 16, 16, 32)

    dec_3 = Reshape(output_shape[1:])(dec_2)
    dec_4 = Conv2DTranspose(filters=32 , kernel_size=3, activation='elu', padding='same')(dec_3)
    dec_5 = Conv2DTranspose(filters=32 , kernel_size=3, activation='elu', padding='same')(dec_4)
    dec_6 = Conv2DTranspose(filters=32 , kernel_size=2, activation='elu', strides=2, padding='same')(dec_5)
    dec_output = Conv2DTranspose(filters=3 , kernel_size=2, padding='same')(dec_6)

    dec_reshaped = Reshape((32, 32, 3))(dec_output)
    decoder = Model(inputs=dec_input, outputs=dec_reshaped)

    reconstruction = decoder(z)         # 由隐空间z进行解码重构

    VAE = Model(inputs=inputs, outputs=reconstruction)

    # 损失函数由交叉熵和KL散度组成
    def vae_loss(inputs, reconstruction):
        x = K.flatten(inputs)
        rec = K.flatten(reconstruction)
        # 计算交叉熵与KL散度
        x_ent = keras.metrics.binary_crossentropy(x, rec)
        kl_div = 0.5 * K.sum(K.exp(z_logsigma) + K.square(z_mu) - z_logsigma - 1, axis=-1)     # axis=-1,表示将最后一个轴上的数据相加
        return img_rows * img_cols * x_ent + kl_div

    VAE.compile(optimizer=optim, loss=vae_loss)

    return VAE, encoder, decoder


if __name__ == '__main__':
    latent_dim = 2
    x_train, y_train, x_test, y_test = U2.get_cifar10()

    VAE, encoder, decoder = define_VAE(
        optim=optimizers.RMSprop(),
        latent_dim=latent_dim,
    )

    VAE.fit(x_train, x_train,
            epochs=100,
            batch_size=BATCH_SIZE,
            validation_data=(x_test, x_test))

    encoder.save_weights('cifar10_enc_weights_latent_dim_' + str(latent_dim) + '.h5')
    decoder.save_weights('cifar10_dec_weights_latent_dim_' + str(latent_dim) + '.h5')
