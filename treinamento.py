import tensorflow
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.inception_v3 import preprocess_input as preprocessInputInceptionV3
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as preprocessMobileNetV2
from tensorflow.keras.applications.vgg16 import preprocess_input as preprocessVGG6
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout, Flatten, Input, Add, GlobalAveragePooling2D
from tensorflow.keras.preprocessing import image
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint as Checkpoint
from tensorflow.keras.regularizers import l1
from IPython.display import clear_output

from tensorflow.keras.models import load_model

# Descomentar linhas abaixo caso tenha problema com treinamento com GPU
# from tensorflow.compat.v1 import ConfigProto
# from tensorflow.compat.v1 import InteractiveSession
# config = ConfigProto()
# config.gpu_options.allow_growth = True
# session = InteractiveSession(config=config)

import os
import sys
import shutil
import time
from PIL import Image
import cv2
import glob
import numpy as np
import random
import matplotlib.pyplot as plt

import Config

class LoadExtractor:

    def __init__(self):
        self._model_name = ''
        self._out_size = 0
        self._target_size = 0
        self._model = None
        self._preprocess_input = None

    def run(self, model_name):
        self._model_name = model_name
        if self._model_name.upper() == 'VGG16':
            print('inicializando VGG16')
            model = VGG16(weights="imagenet", include_top=False)
            preprocess_input = preprocessVGG6
            target_size = (224, 224)
            out_size = 7 * 7 * 512
        elif self._model_name.upper() == 'INCEPTIONV3':
            print('inicializando INCEPTIONV3')
            model = InceptionV3(weights="imagenet", include_top=False)
            preprocess_input = preprocessInputInceptionV3
            target_size = (299, 299)
            out_size = 5 * 5 * 2048
        else:
            model = MobileNetV2(weights="imagenet", include_top=False)
            print('inicializando MobileNetV2')
            preprocess_input = preprocessMobileNetV2
            target_size = (224, 224)
            out_size = 7 * 7 * 1280
        self._out_size = out_size
        self._target_size = target_size
        self._preprocess_input = preprocess_input
        self._model = model

    @property
    def model_name(self):
        if self._model is None:
            print('modelo nao definido')
        else:
            return self._model_name

    @property
    def model(self):
        if self._model is None:
            print('modelo nao definido')
        else:
            return self._model

    @property
    def out_size(self):
        if self._model is None:
            print('modelo nao definido')
        else:
            return self._out_size

    @property
    def preprocess_input(self):
        if self._model is None:
            print('modelo nao definido')
        else:
            return self._preprocess_input

    @property
    def target_size(self):
        if self._model is None:
            print('modelo nao definido')
        else:
            return self._target_size
    @model.setter
    def model(self, new_model):
        self._model = new_model

class LoadImages:

    def __init__(self, extrator):
        self._target_size = extrator.target_size
        self._preprocess_input = extrator.preprocess_input

    def run(self, folder_name, numero_amostras=0, format='jpg', batch_size=32, label=0):
        if sys.platform == 'linux':
            folder_name = folder_name if folder_name[-1] is '/' else f'{folder_name}/'
        else:
            folder_name = folder_name if folder_name[-1] is '\\' else f'{folder_name}\\'
        paths = glob.glob(f'{folder_name}*.{format}')
        if numero_amostras != 0:
            random.shuffle(paths)
            paths = paths[0:numero_amostras]
        images = []
        batchPaths = []
        for i in range(len(paths)):
            batchPaths.append(paths[i])

            if ((i + 1) % batch_size == 0) or (i == (len(paths) - 1)):
                for path in batchPaths:
                    img = image.load_img(path, target_size=self._target_size)
                    plt.imshow(img)
                    plt.show()
                    img = image.img_to_array(img)
                    img = np.expand_dims(img, axis=0)
                    img = self._preprocess_input(img)
                    images.append(img)
                batchPaths = []
        images = np.vstack(images)
        y = np.zeros((images.shape[0])) + label
        return images, y, paths

    def run_test_examples(self, folder_name, numero_amostras=0, format='jpg', label=0):
        if sys.platform == 'linux':
            folder_name = folder_name if folder_name[-1] is '/' else f'{folder_name}/'
        else:
            folder_name = folder_name if folder_name[-1] is '\\' else f'{folder_name}\\'
        paths = glob.glob(f'{folder_name}*.{format}')
        if numero_amostras != 0:
            random.shuffle(paths)
            paths = paths[0:numero_amostras]
        images = []
        images_array = []
        for path in paths:
            img_array = image.load_img(path)
            img = image.load_img(path, target_size=self._target_size)
            img = image.img_to_array(img)
            img = np.expand_dims(img_array, axis=0)
            img = self._preprocess_input(img)
            images.append(img)
            images_array.append(img_array)
        y = np.zeros((len(images))) + label
        return images, images_array, y

class PlotLosses(tensorflow.keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.i = 0
        self.x = []
        self.losses = []
        self.val_losses = []

        self.fig = plt.figure()

        self.logs = []

    def on_epoch_end(self, epoch, logs={}):
        self.logs.append(logs)
        self.x.append(self.i)
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.i += 1

        clear_output(wait=True)
        plt.plot(self.x, self.losses, label="loss")
        plt.plot(self.x, self.val_losses, label="val_loss")
        plt.legend()
        plt.show();


if __name__ == '__main__':

    ARMAS_FOLDER = Config.ARMAS_FOLDER
    NAO_ARMAS_FOLDER = Config.NAO_ARMAS_FOLDER
    ARMAS_TESTE_FOLDER = Config.ARMAS_TESTE_FOLDER
    ARMAS_TESTE_NAO_FOLDER = Config.ARMAS_TESTE_NAO_FOLDER
    RESULTADOS_FOLDER = Config.RESULTADOS_FOLDER
    MODELO_SAVE_FOLDER = Config.MODELO_SAVE_FOLDER

    FOLDERS = [ARMAS_FOLDER, NAO_ARMAS_FOLDER,ARMAS_TESTE_FOLDER,ARMAS_TESTE_NAO_FOLDER]

    AMOSTRAS_POR_CLASSE = Config.AMOSTRAS_POR_CLASSE
    EPOCAS = Config.EPOCAS
    BATCH_SIZE = Config.BATCH_SIZE
    label_nome = Config.label_nome
    MODELO = Config.MODELO

    paths_folder_resultados = glob.glob(f'{RESULTADOS_FOLDER}*.jpg')
    [os.remove(path) for path in paths_folder_resultados]
    # Carregando modelos e adicionando camada fully connected

    extrator = LoadExtractor()
    extrator.run(MODELO)

    # freezando layers
    for layer in extrator.model.layers:
        layer.trainable = False

    # class_weight = {0: 1,
    #                 1: 3}

    out_extractor = extrator.model.output
    fully_connected = GlobalAveragePooling2D()(out_extractor)
    fully_connected = Dense(256, activation='relu')(fully_connected)
    fully_connected = Dense(128, activation='relu')(fully_connected)
    fully_connected = Dense(64, activation='relu')(fully_connected)
    prediction = Dense(2, activation='softmax')(fully_connected)

    extrator.model = Model(inputs=extrator.model.input, outputs=prediction)

    extrator.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])

    # Carregando imagens, normalizadas de acordo com o modelo de rede neural convolucional escolhida para extratora de
    # features

    images_armas, y_armas, paths_armas = LoadImages(extrator).run(ARMAS_FOLDER, AMOSTRAS_POR_CLASSE, label=1)
    images_nao_armas, y_nao_armas, paths_nao_armas = LoadImages(extrator).run(NAO_ARMAS_FOLDER, AMOSTRAS_POR_CLASSE, label=0)

    # Concatenando dados

    X = np.vstack((images_armas,images_nao_armas))
    y = np.concatenate((y_armas, y_nao_armas))
    y = to_categorical(y)

    # call backs. Acoes tomadas entre cada época
    plot = PlotLosses()
    checkpoint = Checkpoint(MODELO_SAVE_FOLDER + '_' + MODELO + 'weights.{epoch:02d}-{val_loss:.2f}_dropout.h5', save_weights_only=False)

    # treinando modelo
    history = extrator.model.fit(X, y, validation_split=0.2, epochs=EPOCAS, batch_size=BATCH_SIZE, callbacks=[plot,checkpoint])

    plt.plot(history.history['loss'], label="train_loss")
    plt.plot(history.history['val_loss'], label="val_loss")
    plt.plot(history.history['categorical_accuracy'], label="train_acc")
    plt.plot(history.history['val_categorical_accuracy'], label="val_acc")

    plt.title('model accuracy')
    plt.ylabel('Loss and Accuracy', fontsize=20)
    plt.xlabel('Epoch', fontsize=20)
    plt.legend(['train', 'val'], loc='upper left')

    plt.title('model loss')

    plt.tick_params(labelsize=20)
    plt.ylim(0, 1)

    plt.legend(prop={'size': 16},framealpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{RESULTADOS_FOLDER}{MODELO}_final.png')
    print(f'salvo em {RESULTADOS_FOLDER}{MODELO}_final.png')
