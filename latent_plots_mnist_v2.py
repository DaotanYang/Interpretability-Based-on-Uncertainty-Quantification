from cProfile import label
import os
import pickle
import cycler
import keras
import numpy as np
from keras import backend as K
from keras.layers import Conv2D, Dense, Flatten, MaxPooling2D
from keras.models import Sequential, load_model
from matplotlib import pyplot as plt
from matplotlib import gridspec
from scipy.stats import norm

import src.utilities as U
import src.utilities2 as U2

from train_mnist_vae import define_VAE
plt.rcParams['figure.figsize'] = 8, 8
#use true type fonts only
plt.rcParams['pdf.fonttype'] = 42 
plt.rcParams['ps.fonttype'] = 42 

def H(x):
    return - np.sum( x * np.log(x + 1e-8), axis=-1)

def visualise_latent_space(decoder, n_grid=10):
    grid = norm.ppf(np.linspace(0.01,0.99, n_grid))

    xx, yy = np.meshgrid(grid, grid)

    X = np.concatenate([xx.reshape(-1,1), yy.reshape(-1,1)], axis=1)
    Z = decoder.predict(X)

    Z = Z.reshape(n_grid, n_grid, 28,28)

    imgrid = np.concatenate(
        [np.concatenate([Z[i,j] for i in range(n_grid)], axis=1)
         for j in range(n_grid)], axis=0)
    plt.imshow(imgrid, cmap='gray_r')


def get_uncertainty_samples(mc_model,encoder, decoder, extent, n_grid=100):

    z_min, z_max = extent
    grid = np.linspace(z_min,z_max,n_grid)          # 产生z_min到z_max之间的n_grid个数。保含z_minz和z_max

    xx, yy = np.meshgrid(grid, grid)
    Z = np.concatenate([xx.reshape(-1,1), yy.reshape(-1,1)], axis=1) #将xx和yy两个数组合并

    # 在潜在空间中的这一点上采样图像，并获得BALD（BALD算的就是互信息）
    X = decoder.predict(Z) # 为潜在空间网格生成相应的图像
    preds,entropy, bald = mc_model.get_results(X)
    return preds, entropy.reshape(xx.shape), bald.reshape(xx.shape) 


# 获取VAE编码器和解码器权重，获取CNN训练结果
def get_models():
    _, encoder, decoder = define_VAE()
    encoder.load_weights('save/my_enc_weights_latent_dim_2.h5')
    decoder.load_weights('save/my_dec_weights_latent_dim_2.h5')

    # encoder.load_weights('save/fashionmnist&mnist_enc_weights_latent_dim_2.h5')
    # decoder.load_weights('save/fashionmnist&mnist_dec_weights_latent_dim_2.h5')

    # encoder.load_weights('save/advmnist&mnist_enc_weights_latent_dim_2.h5')
    # decoder.load_weights('save/advmnist&mnist_dec_weights_latent_dim_2.h5')

    K.set_learning_phase(True)
    model = keras.models.load_model('save/mnist_cnn_run_lenet_dropout.h5')
    mc_model = U.MCModel(model, model.input, n_mc=50)           # MC-Monte-Carlo
    #we have been using more mc samples elsewhere, but save time for now
    return mc_model, encoder, decoder

def get_ML_ensemble(n_mc=10):
    
    _, encoder, decoder = define_VAE()
    encoder.load_weights('save/my_enc_weights_latent_dim_2.h5')
    decoder.load_weights('save/my_dec_weights_latent_dim_2.h5')

    # encoder.load_weights('save/fashionmnist&mnist_enc_weights_latent_dim_2.h5')
    # decoder.load_weights('save/fashionmnist&mnist_dec_weights_latent_dim_2.h5')

    # encoder.load_weights('save/advmnist&mnist_enc_weights_latent_dim_2.h5')
    # decoder.load_weights('save/advmnist&mnist_dec_weights_latent_dim_2.h5')

    K.set_learning_phase(False)
    ms = []
    for name in filter(lambda x: 'mnist_cnn_run_lenet_dropout' in x, os.listdir('save')):
        print('loading model {}'.format(name))
        model = load_model('save/' + name)
        ms.append(model)

    model = U.MCEnsembleWrapper(ms, n_mc=10)
    return model, encoder, decoder 


def make_plot(proj_x1,
              proj_y1,
            #   proj_x2,
            #   proj_y2,
              extent,
              plot_bg,
              decoder,
              title="",
              bgcmap='gray',
              bgalpha=0.9,
              sccmap='seismic'):
    
    f, ax1 = plt.subplots()

    # label1 = [proj_y1.argmax(axis=1)]
    # label2 = [proj_y2.argmax(axis=1)]
  
    ax1.scatter(proj_x1[:,0],
                proj_x1[:,1],
                c = proj_y1.argmax(axis=1),
                marker=',',
                s=1,
                cmap='Reds',        # Reds,tab10,sesmics
                label = str(proj_y1.argmax(axis=1)),
               alpha=0.1
    )

    # ax1.scatter(proj_x2[:,0],
    #             proj_x2[:,1],
    #             c = proj_y2.argmax(axis=1),
    #             marker=',',
    #             s=1,
    #             cmap='PuBu',
    #             label = str(proj_y2.argmax(axis=1)),
    #            alpha=0.1
    # )

    ax1.imshow(plot_bg,
                 cmap=bgcmap,
                 origin='lower',
                 alpha=bgalpha,
                 extent=extent,
    )
    ax1.set_xlabel('First Latent Dimension')
    ax1.set_ylabel('Second Latent Dimension')
    ax1.set_title(title) 

    ax1.legend()

    return

if __name__ == '__main__':

    model, encoder, decoder = get_models()
    # model, encoder, decoder = get_ML_ensemble(n_mc=20)

    m_x_train, m_y_train, m_x_test, m_y_test = U.get_mnist()
    # fm_x_train, fm_y_train, fm_x_test, fm_y_test = U2.get_fashion_mnist()
    # advm_x_train, advm_y_train, advm_x_test, advm_y_test = U2.get_adv_mnist()

    # Fashion_MNIST和MNIST数据集
    # x_train = np.concatenate([m_x_train, fm_x_train])
    # x_test  = np.concatenate([m_x_test , fm_x_test ])
    # y_train = np.concatenate([m_y_train, fm_y_train])
    # y_test  = np.concatenate([m_y_test , fm_y_test ])

    # adversarial samples和MNIST数据集
    # x_train = np.concatenate([m_x_train, advm_x_train])
    # x_test  = np.concatenate([m_x_test , advm_x_test ])
    # y_train = np.concatenate([m_y_train, advm_y_train])
    # y_test  = np.concatenate([m_y_test , advm_y_test ])

    proj_m_x_train = encoder.predict(m_x_train)
    # proj_fm_x_train = encoder.predict(fm_x_train)
    # proj_advm_x_train = encoder.predict(advm_x_train)

    zmin, zmax = -10,10
    n_grid = 100
    preds, plot_ent, plot_bald = get_uncertainty_samples(model,
                                                         encoder,
                                                         decoder,
                                                         [zmin, zmax],
                                                         n_grid=n_grid)
    
    make_plot(proj_m_x_train,
              m_y_train,
            #   proj_fm_x_train,
            #   fm_y_train,
              [zmin, zmax, zmin, zmax],
              plot_bald,
              decoder
    )
    print('done')              
    plt.savefig('Result_picture/lenet5_test')
    plt.show()


