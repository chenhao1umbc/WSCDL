"""This file is to reproduce the result of deepMIML with tensorflow.keras and python 3.8
This code is based on DeepMIML.demo.demo_train_miml_vgg.py
"""
#%% load package
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout, Reshape
from tensorflow.keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D

import scipy.io as sio
import sklearn.metrics as metrics
import matplotlib.pyplot as plt

# from DeepMIML.lib.cocodemo import COCODataset, COCODataLayer
from Deep_MIML_repr.deepmiml import DeepMIML
# from DeepMIML.lib.deepmiml.utils import save_keras_model
from Deep_MIML_repr.vgg_16 import VGG_16

"limit the gpu memory usage"
gpu = tf.config.experimental.list_physical_devices('GPU')
for i in range(len(gpu)):
    tf.config.experimental.set_memory_growth(gpu[i], True)
# tf.keras.applications.VGG16() # to load the pre-trained weights, for first use

#%%
loss = "binary_crossentropy"
nb_epoch = 100
batch_size = 64
L = 10
K = 5
model_name = "miml_vgg_16"

# vgg_model_path = "../data/vgg16_weights_tf_dim_ordering_tf_kernels.h5"
# base_model = VGG_16(vgg_model_path)
# original model is too large easily overfits, validation loss increase but train loss descrease
model = Sequential(layers=ZeroPadding2D((1,1),input_shape=(100, 500, 1)))
model.add(Convolution2D(4, 3, activation='relu'))
model.add(Convolution2D(4, 3, activation='relu'))
model.add(MaxPooling2D((5,10), strides=(4,5)))
model.add(Dropout(0.5))

deepmiml = DeepMIML(L=L, K=K, base_model=model)
deepmiml.model.summary()

"load training data"
mat = sio.loadmat('../data/ESC10/esc10_tr.mat')
x, y, yy = mat['X'], mat['Y'], mat['yy']

#%%
print("Compiling Deep MIML Model...")
deepmiml.model.compile(optimizer="adam", loss=loss, metrics=["accuracy"])

print("Start Training...")
history = deepmiml.model.fit(x=x[...,None], y=y, \
    batch_size = batch_size, epochs=nb_epoch, validation_split=0.25)
# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# save_keras_model(deepmiml.model, "outputs/{}/{}".format(dataset.name, model_name))


#%% test part
model = deepmiml.model
mat = sio.loadmat('../data/ESC10/esc10_tr.mat')
xt, yt, yyt = mat['X'], mat['Y'], mat['yy']

print("Start Predicting...")
y_pred = model.predict(xt[...,None])
thr = 0.23
y_pred[y_pred>=thr] = 1
y_pred[y_pred<thr] = 0
metrics.f1_score(yt.flatten(), y_pred.flatten())


#%%
import torch
route = '/home/chenhao1/Matlab/WSCDL/'
with open(route+'you_raich_0.txt') as f:
    data =f.readlines()
rec = []
prec = []
count = 0
for i, d in enumerate(data):
    if d == 'rec =\n':
        rec.append(float(data[i+2][4:10]))
        count += 1

    if d == 'prec =\n':
        prec.append(float(data[i+2][4:10]))
        if count == 10:
            print(rec[-1])
            print(prec[-1])

rec, prec = torch.tensor(rec), torch.tensor(prec)
f1 = 2/(1/rec+1/prec)
v, i = f1.sort()
# best lamb, winzize, N
# 10, 30, 200
# 10, 100, 10
# 10, 50, 50

with open(route+'you_raich_k2.txt') as f:
    data =f.readlines()
f1 = []
for i, d in enumerate(data):
    if d == 'f1 =\n':
        f1.append(float(data[i+2][4:10]))

f1 = torch.tensor(f1)
v, i = f1.sort()
# best 
# 39,  57,  23
# 0.5555, 0.5559, 0.5579
count = 0
for i, d in enumerate(data):
    if d == 'f1 =\n':
        if count== 38 or count == 56 or count == 22:
            print('lamb', data[i+7])
            print('winsize', data[i+12])
            print('N', data[i+17])
        count += 1
# best lamb, winzize, N, k=2
# 1, 150, 200
# 0.01, 50, 50
# 0.1, 100, 50
# %%
