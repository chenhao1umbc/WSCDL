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
mat = sio.loadmat('/home/chenhao1/Hpython/data/ESC10/esc10_tr.mat')
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
mat = sio.loadmat('/home/chenhao1/Hpython/data/ESC10/esc10_te.mat')
xt, yt, yyt = mat['X'], mat['Y'], mat['yy']

print("Start Predicting...")
y_pred = model.predict(xt[...,None])
r = metrics.roc_curve(yt.flatten(), y_pred.flatten())
plt.plot(r[0], r[1])
print('auc' , metrics.auc(r[0], r[1]))

thr = 0.1001
y_pred[y_pred>thr] = 1
y_pred[y_pred<=thr] = 0 
print('recall', metrics.recall_score(yt.flatten(), y_pred.flatten()))
print('f1', metrics.f1_score(yt.flatten(), y_pred.flatten()))


# %%
