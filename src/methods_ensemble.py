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

import colorlover as cl
import plotly.graph_objs as go
import plotly.express as px
import numpy as np
from sklearn import metrics

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
    # encoder.load_weights('save/my_enc_weights_latent_dim_2.h5')
    # decoder.load_weights('save/my_dec_weights_latent_dim_2.h5')

    encoder.load_weights('save/fashionmnist&mnist_enc_weights_latent_dim_2.h5')
    decoder.load_weights('save/fashionmnist&mnist_dec_weights_latent_dim_2.h5')

    # encoder.load_weights('save/advmnist&mnist_enc_weights_latent_dim_2.h5')
    # decoder.load_weights('save/advmnist&mnist_dec_weights_latent_dim_2.h5')

    K.set_learning_phase(True)
    model = keras.models.load_model('save/mnist_cnn_run_lenet_dropout.h5')
    mc_model = U.MCModel(model, model.input, n_mc=50)           # MC-Monte-Carlo
    #we have been using more mc samples elsewhere, but save time for now
    return mc_model, encoder, decoder

def get_ML_ensemble(n_mc=10):
    
    _, encoder, decoder = define_VAE()
    # encoder.load_weights('save/my_enc_weights_latent_dim_2.h5')
    # decoder.load_weights('save/my_dec_weights_latent_dim_2.h5')

    encoder.load_weights('save/fashionmnist&mnist_enc_weights_latent_dim_2.h5')
    decoder.load_weights('save/fashionmnist&mnist_dec_weights_latent_dim_2.h5')

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

def generate_data():
    model, encoder, decoder = get_models()
    # model, encoder, decoder = get_ML_ensemble(n_mc=20)

    m_x_train, m_y_train, m_x_test, m_y_test = U.get_mnist()
    fm_x_train, fm_y_train, fm_x_test, fm_y_test = U2.get_fashion_mnist()
    # advm_x_train, advm_y_train, advm_x_test, advm_y_test = U2.get_adv_mnist()

    # Fashion_MNIST和MNIST数据集
    x_train = np.concatenate([m_x_train, fm_x_train])
    x_test  = np.concatenate([m_x_test , fm_x_test ])
    y_train = np.concatenate([m_y_train, fm_y_train])
    y_test  = np.concatenate([m_y_test , fm_y_test ])

    # adversarial samples和MNIST数据集
    # x_train = np.concatenate([m_x_train, advm_x_train])
    # x_test  = np.concatenate([m_x_test , advm_x_test ])
    # y_train = np.concatenate([m_y_train, advm_y_train])
    # y_test  = np.concatenate([m_y_test , advm_y_test ])

    proj_m_x_train = encoder.predict(m_x_train)
    proj_fm_x_train = encoder.predict(fm_x_train)
    # proj_advm_x_train = encoder.predict(advm_x_train)

    zmin, zmax = -10,10
    n_grid = 100
    preds, plot_ent, plot_bald = get_uncertainty_samples(model,
                                                         encoder,
                                                         decoder,
                                                         [zmin, zmax],
                                                         n_grid=n_grid)
    
    return proj_m_x_train, m_y_train, proj_fm_x_train, fm_y_train, [zmin, zmax, zmin, zmax], plot_ent, decoder

def serve_lantent_plot(
    proj_x1,
    proj_y1,
    proj_x2,
    proj_y2,
    extent,
    plot_bg,
    decoder
):
    # # Get train and test score from model
    # y_pred_train = (model.decision_function(X_train) > threshold).astype(int)
    # y_pred_test = (model.decision_function(X_test) > threshold).astype(int)
    # train_score = metrics.accuracy_score(y_true=y_train, y_pred=y_pred_train)
    # test_score = metrics.accuracy_score(y_true=y_test, y_pred=y_pred_test)

    # # Compute threshold
    # scaled_threshold = threshold * (Z.max() - Z.min()) + Z.min()
    # range = max(abs(scaled_threshold - Z.min()), abs(scaled_threshold - Z.max()))

    # Colorscale
    # bright_cscale = [[0, "#ff3700"], [1, "#0b8bff"]]
    # cscale = [
    #     [0.0000000, "#ff744c"],
    #     [0.1428571, "#ff916d"],
    #     [0.2857143, "#ffc0a8"],
    #     [0.4285714, "#ffe7dc"],
    #     [0.5714286, "#e5fcff"],
    #     [0.7142857, "#c8feff"],
    #     [0.8571429, "#9af8ff"],
    #     [1.0000000, "#20e6ff"],
    # ]

    # Create the plot
    # # Plot the prediction contour of the SVM
    # trace0 = go.Contour(
    #     x=np.arange(xx.min(), xx.max(), mesh_step),
    #     y=np.arange(yy.min(), yy.max(), mesh_step),
    #     z=Z.reshape(xx.shape),
    #     zmin=scaled_threshold - range,
    #     zmax=scaled_threshold + range,
    #     hoverinfo="none",
    #     showscale=False,
    #     contours=dict(showlines=False),
    #     colorscale=cscale,
    #     opacity=0.9,
    # )

    # # Plot the threshold
    # trace1 = go.Contour(
    #     x=np.arange(xx.min(), xx.max(), mesh_step),
    #     y=np.arange(yy.min(), yy.max(), mesh_step),
    #     z=Z.reshape(xx.shape),
    #     showscale=False,
    #     hoverinfo="none",
    #     contours=dict(
    #         showlines=False, type="constraint", operation="=", value=scaled_threshold
    #     ),
    #     name=f"Threshold ({scaled_threshold:.3f})",
    #     line=dict(color="#708090"),
    # )

    # Plot Training Data
    trace2 = px.scatter(
        x=proj_x1[:, 0],
        y=proj_y1[:, 1],
        mode="markers",
        marker=dict(size=10),
    )

    # Plot Test Data
    trace3 = px.scatter(
        x=proj_x2[:, 0],
        y=proj_y2[:, 1],
        mode="markers",
        marker=dict(size=10),
    ),

    trace4 = px.imshow(
        plot_bg,
        color_continuous_scale='gray',
        cmap='gray',
        origin='lower',
        extent=extent,
    )
    


    layout = go.Layout(
        xaxis=dict(ticks="", showticklabels=False, showgrid=False, zeroline=False),
        yaxis=dict(ticks="", showticklabels=False, showgrid=False, zeroline=False),
        hovermode="closest",
        legend=dict(x=0, y=-0.01, orientation="h"),
        margin=dict(l=0, r=0, t=0, b=0),
        plot_bgcolor=plot_bg,
        paper_bgcolor="#282b38",
        font={"color": "#a5b1cd"},
    )

    data = [trace2, trace3, trace4]
    figure = go.Figure(data=data, layout=layout)

    return figure

def serve_decoder_plot():

    return 

def serve_roc_curve(model, X_test, y_test):
    decision_test = model.decision_function(X_test)
    fpr, tpr, threshold = metrics.roc_curve(y_test, decision_test)

    # AUC Score
    auc_score = metrics.roc_auc_score(y_true=y_test, y_score=decision_test)

    trace0 = go.Scatter(
        x=fpr, y=tpr, mode="lines", name="Test Data", marker={"color": "#13c6e9"}
    )

    layout = go.Layout(
        title=f"ROC Curve (AUC = {auc_score:.3f})",
        xaxis=dict(title="False Positive Rate", gridcolor="#2f3445"),
        yaxis=dict(title="True Positive Rate", gridcolor="#2f3445"),
        legend=dict(x=0, y=1.05, orientation="h"),
        margin=dict(l=100, r=10, t=25, b=40),
        plot_bgcolor="#282b38",
        paper_bgcolor="#282b38",
        font={"color": "#a5b1cd"},
    )

    data = [trace0]
    figure = go.Figure(data=data, layout=layout)

    return figure


def serve_pie_confusion_matrix(model, X_test, y_test, Z, threshold):
    # Compute threshold
    scaled_threshold = threshold * (Z.max() - Z.min()) + Z.min()
    y_pred_test = (model.decision_function(X_test) > scaled_threshold).astype(int)

    matrix = metrics.confusion_matrix(y_true=y_test, y_pred=y_pred_test)
    tn, fp, fn, tp = matrix.ravel()

    values = [tp, fn, fp, tn]
    label_text = ["True Positive", "False Negative", "False Positive", "True Negative"]
    labels = ["TP", "FN", "FP", "TN"]
    blue = cl.flipper()["seq"]["9"]["Blues"]
    red = cl.flipper()["seq"]["9"]["Reds"]
    colors = ["#13c6e9", blue[1], "#ff916d", "#ff744c"]

    trace0 = go.Pie(
        labels=label_text,
        values=values,
        hoverinfo="label+value+percent",
        textinfo="text+value",
        text=labels,
        sort=False,
        marker=dict(colors=colors),
        insidetextfont={"color": "white"},
        rotation=90,
    )

    layout = go.Layout(
        title="Confusion Matrix",
        margin=dict(l=50, r=50, t=100, b=10),
        legend=dict(bgcolor="#282b38", font={"color": "#a5b1cd"}, orientation="h"),
        plot_bgcolor="#282b38",
        paper_bgcolor="#282b38",
        font={"color": "#a5b1cd"},
    )

    data = [trace0]
    figure = go.Figure(data=data, layout=layout)

    return figure