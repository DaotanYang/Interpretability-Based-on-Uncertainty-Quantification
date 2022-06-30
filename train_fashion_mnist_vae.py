import keras
from keras import backend as K
from keras.layers import Dense, Flatten, Lambda, Reshape
from keras.models import Model
import tensorflow as tf
import src.utilities as U
import src.utilities2 as U2

tf.compat.v1.disable_eager_execution()

BATCH_SIZE = 128


def define_VAE(optim='adagrad', latent_dim=2):
    inputs = keras.layers.Input(shape=(28, 28, 1))  # encoder_input
    x = Flatten()(inputs)       # 展平成28x28=784个输出的一维数组
    enc_1 = Dense(400, activation='elu')(x)         # encoder，encoding过程
    enc_2 = Dense(256, activation='elu')(enc_1)

    z_mu = Dense(latent_dim)(enc_2)             # z代表隐变量
    z_logsigma = Dense(latent_dim)(enc_2)       # g1=h1,此处mu与sigma的计算共享参数

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
    dec_1 = Dense(256, activation='elu')(dec_input)                 # decoder
    dec_2 = Dense(400, activation='elu')(dec_1)
    dec_output = Dense(784, activation='sigmoid')(dec_2)            # 784=28x28

    dec_reshaped = Reshape((28, 28, 1))(dec_output)
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
        return 28 * 28 * x_ent + kl_div

    VAE.compile(optimizer=optim, loss=vae_loss)

    return VAE, encoder, decoder


if __name__ == '__main__':
    latent_dim = 2
    x_train, y_train, x_test, y_test = U2.get_fashion_mnist()

    VAE, encoder, decoder = define_VAE(
        optim=keras.optimizers.Adam(),
        latent_dim=latent_dim,
    )

    VAE.fit(x_train, x_train,
            epochs=50,
            batch_size=BATCH_SIZE,
            validation_data=(x_test, x_test))

    encoder.save_weights('fashion_mnist_enc_weights_latent_dim_' + str(latent_dim) + '.h5')
    decoder.save_weights('fashion_mnist_dec_weights_latent_dim_' + str(latent_dim) + '.h5')
