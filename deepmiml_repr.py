"""This file is to reproduce the result of deepMIML with tensorflow.keras and python 3.8
This code is based on DeepMIML.demo.demo_train_miml_vgg.py
"""
#%% load package
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout
from tensorflow. keras.layers import Convolution2D
import sys
sys.path.insert(0, "lib")

# from DeepMIML.lib.cocodemo import COCODataset, COCODataLayer
from Deep_MIML_repr.deepmiml import DeepMIML
from DeepMIML.lib.deepmiml.utils import save_keras_model
from Deep_MIML_repr.vgg_16 import VGG_16

#%%
loss = "binary_crossentropy"
nb_epoch = 10
batch_size = 32
L = 80
K = 20
model_name = "miml_vgg_16"

vgg_model_path = "/home/chenhao1/.keras/models/vgg16_weights_tf_dim_ordering_tf_kernels.h5"
base_model = VGG_16(vgg_model_path)
base_model = Sequential(layers=base_model.layers[: -7])
base_model.add(Convolution2D(512, 1, activation="relu"))
base_model.add(Dropout(0.5))

deepmiml = DeepMIML(L=L, K=K, base_model=base_model)
# deepmiml.model.summary()

#%%
print("Compiling Deep MIML Model...")
deepmiml.model.compile(optimizer="adadelta", loss=loss, metrics=["accuracy"])

print("Start Training...")
samples_per_epoch = data_layer.num_images
deepmiml.model.fit_generator(data_layer.generate(),
        samples_per_epoch=samples_per_epoch,
        nb_epoch=nb_epoch)

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
(y_gt, y_pred, threshold_value=0.5)