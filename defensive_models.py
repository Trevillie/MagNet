## defensive_models.py -- defines several flavors of autoencoders for defense
##
## Copyright (C) 2017, Dongyu Meng <zbshfmmm@gmail.com>.
##
## This program is licenced under the BSD 2-Clause licence,
## contained in the LICENCE file in this directory.

import os
import numpy as np
from keras.layers.core import Lambda
from keras.layers.merge import Average, add
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, AveragePooling2D
from keras.models import Model
import keras.regularizers as regs


class DenoisingAutoEncoder:
    def __init__(self, image_shape,
                 structure,
                 v_noise=0.0,
                 activation="relu",
                 model_dir="./defensive_models/",
                 reg_strength=0.0):
        """
        Denoising autoencoder.

        image_shape: Shape of input image. e.g. 28, 28, 1.
        structure: Structure of autoencoder.
        v_noise: Volume of noise while training.
        activation: What activation function to use.
        model_dir: Where to save / load model from.
        reg_strength: Strength of L2 regularization.
        """
        h, w, c = image_shape
        self.image_shape = image_shape
        self.model_dir = model_dir
        self.v_noise = v_noise

        input_img = Input(shape=self.image_shape)
        x = input_img

        for layer in structure:
            if isinstance(layer, int):
                x = Conv2D(layer, (3, 3), activation=activation, padding="same",
                           activity_regularizer=regs.l2(reg_strength))(x)
            elif layer == "max":
                x = MaxPooling2D((2, 2), padding="same")(x)
            elif layer == "average":
                x = AveragePooling2D((2, 2), padding="same")(x)
            else:
                print(layer, "is not recognized!")
                exit(0)

        for layer in reversed(structure):
            if isinstance(layer, int):
                x = Conv2D(layer, (3, 3), activation=activation, padding="same",
                           activity_regularizer=regs.l2(reg_strength))(x)
            elif layer == "max" or layer == "average":
                x = UpSampling2D((2, 2))(x)

        decoded = Conv2D(c, (3, 3), activation='sigmoid', padding='same',
                         activity_regularizer=regs.l2(reg_strength))(x)
        self.model = Model(input_img, decoded)

    def train(self, data, archive_name, num_epochs=100, batch_size=256,
              if_save=True):
        self.model.compile(loss='mean_squared_error',
                           metrics=['mean_squared_error'],
                           optimizer='adam')

        noise = self.v_noise * np.random.normal(size=np.shape(data.train_data))
        noisy_train_data = data.train_data + noise
        noisy_train_data = np.clip(noisy_train_data, 0.0, 1.0)

        self.model.fit(noisy_train_data, data.train_data,
                       batch_size=batch_size,
                       validation_data=(data.validation_data, data.validation_data),
                       epochs=num_epochs,
                       shuffle=True)

        if if_save: self.model.save(os.path.join(self.model_dir, archive_name))

    def load(self, archive_name, model_dir=None):
        if model_dir is None: model_dir = self.model_dir
        self.model.load_weights(os.path.join(model_dir, archive_name))


class PackedAutoEncoder:
    def __init__(self, image_shape, structure, data,
                 v_noise=0.1, n_pack=2, pre_epochs=3, activation="relu",
                 model_dir="./defensive_models/"):
        """
        Train different autoencoders.
        Demo code for graybox scenario.

        pre_epochs: How many epochs do we train before fine-tuning.
        n_pack: Number of autoencoders we want to train at once.
        """
        self.v_noise = v_noise
        self.n_pack = n_pack
        self.model_dir = model_dir
        pack = []

        for i in range(n_pack):
            dae = DenoisingAutoEncoder(image_shape, structure, v_noise=v_noise,
                                       activation=activation, model_dir=model_dir)
            dae.train(data, "", if_save=False, num_epochs=pre_epochs)
            pack.append(dae.model)

        shared_input = Input(shape=image_shape, name="shared_input")
        outputs = [dae(shared_input) for dae in pack]
        avg_output = Average()(outputs)
        delta_outputs = [add([avg_output, Lambda(lambda x: -x)(output)])
                         for output in outputs]

        self.model = Model(inputs=shared_input, outputs=outputs+delta_outputs)

    def train(self, data, archive_name, alpha, num_epochs=10, batch_size=128):
        noise = self.v_noise * np.random.normal(size=np.shape(data.train_data))
        noisy_train_data = data.train_data + noise
        noisy_train_data = np.clip(noisy_train_data, 0.0, 1.0)

        train_zeros = [np.zeros_like(data.train_data)] * self.n_pack
        val_zeros = [np.zeros_like(data.validation_data)] * self.n_pack

        self.model.compile(loss="mean_squared_error", optimizer="adam",
                           loss_weights=[1.0]*self.n_pack + [-alpha]*self.n_pack)

        self.model.fit(noisy_train_data,
                       [data.train_data]*self.n_pack + train_zeros,
                       batch_size=batch_size,
                       validation_data=(data.validation_data,
                            [data.validation_data]*self.n_pack+val_zeros),
                       epochs=num_epochs,
                       shuffle=True)

        for i in range(self.n_pack):
            model = Model(self.model.input, self.model.outputs[i])
            model.save(os.path.join(self.model_dir, archive_name+"_"+str(i)))

    def load(self, archive_name, model_dir=None):
        if model_dir is None: model_dir = self.model_dir
        self.model.load_weights(os.path.join(model_dir, archive_name))

