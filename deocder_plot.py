import os
import pickle
from pyexpat import model
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

def get_decoder(decoder_url):
    _, encoder, decoder = define_VAE()
    decoder.load_weights(decoder_url)
    return decoder

def get_model(method_index, net_index):
    if method_index == 'MC':
        if net_index == 'LeNet5':
            K.set_learning_phase(True)
            model = keras.models.load_model('save/mnist_cnn_run_lenet_dropout.h5')
            mc_model = U.MCModel(model, model.input, n_mc=50)           # MC-Monte-Carlo
            #we have been using more mc samples elsewhere, but save time for now
            return mc_model
        elif net_index == 'AlexNet':
            K.set_learning_phase(True)
            model = keras.models.load_model('save/mnist_cnn_run_alexnet.h5')
            mc_model = U.MCModel(model, model.input, n_mc=50)           # MC-Monte-Carlo
            #we have been using more mc samples elsewhere, but save time for now
            return mc_model
        else :
            K.set_learning_phase(True)
            model = keras.models.load_model('save/mnist_cnn_run_resnet34_dropout.h5')
            mc_model = U.MCModel(model, model.input, n_mc=50)           # MC-Monte-Carlo
            #we have been using more mc samples elsewhere, but save time for now
            return mc_model
    else :
        if net_index == 'LeNet5':
            K.set_learning_phase(False)
            ms = []
            for name in filter(lambda x: 'mnist_cnn_run_lenet_dropout' in x, os.listdir('save')):
                print('loading model {}'.format(name))
                model = load_model('save/' + name)
                ms.append(model)
            model = U.MCEnsembleWrapper(ms, n_mc=10)
            return model
        elif net_index == 'AlexNet':
            K.set_learning_phase(False)
            ms = []
            for name in filter(lambda x: 'mnist_cnn_run_alexnet' in x, os.listdir('save')):
                print('loading model {}'.format(name))
                model = load_model('save/' + name)
                ms.append(model)
            model = U.MCEnsembleWrapper(ms, n_mc=10)
            return model
        else :
            K.set_learning_phase(False)
            ms = []
            for name in filter(lambda x: 'mnist_cnn_run_resnet34_dropout' in x, os.listdir('save')):
                print('loading model {}'.format(name))
                model = load_model('save/' + name)
                ms.append(model)
            model = U.MCEnsembleWrapper(ms, n_mc=10)
            return model

def make_decoder_plot(latent_z1,
                      latent_z2,
                      decoder):
    f, ax = plt.subplots()
    ax.imshow(decoder.predict(np.array([[latent_z1, latent_z2]])).squeeze(), cmap='gray_r') 

    ax.legend()

    return

def caculate_MI_Ent(z1,
                    z2,
                    decoder,
                    model):
    dream = decoder.predict(np.array([[z1, z2]]))
    pred,entropy,bald = model.get_results(dream)
    pre_class = pred.argmax(axis=1)
    pre_prob = pred.max(axis=1)
    MI_score = entropy[0]
    Ent = bald[0]

    return pre_class, pre_prob, MI_score, Ent

def get_lantent_picture(z1,
                        z2,
                        decoder_index,
                        method_index,
                        net_index):

    if decoder_index == 'MNIST':
        decoder_url = 'save/my_dec_weights_latent_dim_2.h5'
    elif decoder_index == 'FashionMNIST':
        decoder_url = 'save/fashionmnist&mnist_dec_weights_latent_dim_2.h5'
    elif decoder_index == 'AdvMNIST':
        decoder_url = 'save/advmnist&mnist_dec_weights_latent_dim_2.h5'
    else :
        decoder_url = ''
    
    model = get_model(method_index, net_index)

    decoder = get_decoder(decoder_url)
    latent_z1 = z1
    latent_z2 = z2

    make_decoder_plot(
        latent_z1,
        latent_z2,
        decoder
    )

    pre_class, pre_prob, MI_score, Ent = caculate_MI_Ent(latent_z1,
                                                   latent_z2,
                                                   decoder,
                                                   model)

    # print('done')              
    plt.savefig('static/latent_picture/latent')
    # plt.show()

    return pre_class, pre_prob, MI_score, Ent