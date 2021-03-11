"""This file is to reproduce the result of deepMIML with tensorflow.keras and python 3.8
This code is based on DeepMIML.demo.demo_train_miml_vgg.py
"""
#%% load package
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D

import scipy.io as sio
import sklearn.metrics as metrics

# from DeepMIML.lib.cocodemo import COCODataset, COCODataLayer
from Deep_MIML_repr.deepmiml import DeepMIML
from DeepMIML.lib.deepmiml.utils import save_keras_model
from Deep_MIML_repr.vgg_16 import VGG_16

"limit the gpu memory usage"
gpu = tf.config.experimental.list_physical_devices('GPU')
for i in range(len(gpu)):
    tf.config.experimental.set_memory_growth(gpu[i], True)
# tf.keras.applications.VGG16() # to load the pre-trained weights, for first use

#%%
loss = "binary_crossentropy"
nb_epoch = 100
batch_size = 32
L = 10
K = 5
model_name = "miml_vgg_16"

vgg_model_path = "../data/vgg16_weights_tf_dim_ordering_tf_kernels.h5"
base_model = VGG_16(vgg_model_path)
model = Sequential(layers=ZeroPadding2D((1,1),input_shape=(100, 500, 1)))
model.add(Convolution2D(3, 1, activation="relu"))
model.add(Sequential(base_model.layers[1: -7]))
model.add(Convolution2D(512, 1, activation="relu"))
model.add(Dropout(0.5))

deepmiml = DeepMIML(L=L, K=K, base_model=model)
deepmiml.model.summary()

"load training data"
mat = sio.loadmat('../data/ESC10/esc10_tr.mat')
x, y, yy = mat['X'], mat['Y'], mat['yy']

#%%
print("Compiling Deep MIML Model...")
deepmiml.model.compile(optimizer="adadelta", loss=loss, metrics=["accuracy"])

print("Start Training...")
deepmiml.model.fit(x=x[...,None], y=y, epochs=nb_epoch, validation_split=0.25)

# save_keras_model(deepmiml.model, "outputs/{}/{}".format(dataset.name, model_name))


#%% test part
model = deepmiml.model
print("Compiling Deep MIML Model...")
model.compile(optimizer="adadelta", loss="binary_crossentropy")

# # crate data layer
# dataset = COCODataset("data/coco", "val", "2014")
# data_layer = COCODataLayer(dataset, batch_size=batch_size)

print("Start Predicting...")
num_images = dataset.num_images
y_pred = np.zeros((num_images, dataset.num_classes))
y_gt = np.zeros((num_images, dataset.num_classes))
for i in range(0, num_images, batch_size):
    if i // batch_size % 10 == 0:
        print("[progress] ({}/{})".format(i, num_images))
    x_val_mini, y_val_mini = data_layer.get_data(i, i + batch_size)
    y_pred_mini = model.predict(x_val_mini)
    y_pred[i: i + batch_size] = y_pred_mini
    y_gt[i: i + batch_size] = y_val_mini
    metrics.f1_score(y_gt, y_pred, threshold_value=0.5)


#%%
with open('/home/chenhao1/Matlab/WSCDL/you_raich.txt', 'r') as f:
    res = f.readlines()

count = 0
for i, s in enumerate(res):
    if s == 'rec =\n':
        count += 1
        print(res[i+2])
