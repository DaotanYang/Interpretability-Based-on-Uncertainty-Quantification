# Interpretability-Based-on-Uncertainty-Quantification
Research and Application of Deep Learning Model Interpretability Based on Uncertainty Quantification<br>
This project is based on the paper:Understanding Measures of Uncertainty for Adversarial Example Detection-Lewis Smith and Yarin Gal. https://github.com/lsgos/uncertainty-adversarial-paper. We made some changes based on their code.<br>
LeNet5, ResNet-34 and AlexNet models are based on https://github.com/rookiedata1/Keras-CNN-mnist-classification. We made some changes based on the code.

****

Experiment Process
-----------
The experimental process is shown in the following figure：
![数据流图3](https://user-images.githubusercontent.com/58934786/176865621-1c63d404-7086-49f7-b162-e645d9cd040b.jpg)


Models, Datasets and Metrics
-----------
|Models|Datasets|Metrics|
|--|--|--|
|LeNst5|MNIST|Entorpy|
|AlexNet|FashionMNIST|Mutual Information|
|ResNet-34|Adversarial Samples of MNIST|softmax output|


Example: Uncertainty Quantification on MNIST Dataset
-----------
The results on the MNIST dataset are shown below<br>
![MNIST_LeNet5_MC_Ent](https://user-images.githubusercontent.com/58934786/176863116-7652b3bf-bf34-4b78-85f4-4803a48e8552.png)
<br>
In the above figure, the colored areas represent numbers 0 to 9, and the black and white areas represent uncertainty. The darker the color, the higher the uncertainty.


Example: Uncertainty Quantification on FashionMNIST & MNIST Dataset
-----------
The results on the MNIST & FashionMNIST dataset are shown below<br>
![FashionMNIST_LeNet5_MC_Ent](https://user-images.githubusercontent.com/58934786/176870123-879d10da-0bcd-4a19-b8ad-4fcf0be06b1e.png)
<br>
In the above figure, the red areas represent MNIST, and the blue areas represent FashionMNIST. The black and white areas represent uncertainty.

Example: Uncertainty Quantification on Adversarial Samples & MNIST Dataset
-----------
The results on the adversarial samples & MNIST dataset are shown below<br>
![AdvMNIST_LeNet5_MC_Ent](https://user-images.githubusercontent.com/58934786/176893226-2aaf29b5-3e87-4dc2-b146-d72fc2712423.png)
<br>
In the above figure, the red areas represent MNIST, and the blue areas(point areas in the center) represent FashionMNIST. The black and white areas represent uncertainty.


Results
-----------
For the MNIST dataset, through interpretability work, we determined that the appropriate data uncertainty measure is entropy, and the measure method is the Deep Ensemble method. We also explore the latent space distribution of Fashion MNIST dataset and adversarial samples of MNIST dataset. For the MNIST dataset itself, in the experiment, it can be found that under the ResNet34 network, when entropy is used as an indicator, the uncertainty of most out-of-domain areas is high, but there is always an exception. Decoding this place can find that the picture corresponding to this part of the space is the number 1, but It is not included in the MNIST dataset. In this regard, we consider the MNIST dataset to be an "incomplete dataset", and the missing part is the low distribution uncertainty area (black area) outside the distribution. Based on this, a dataset can be expanded and generated, and the low-distribution uncertainty regions outside the distribution can be restored through the decoder, and the restored images can be expanded into a training set to deal with scenarios where training data is missing or insufficient.

















