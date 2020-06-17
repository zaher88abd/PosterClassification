import os
import cv2
import image_functions
from tensorflow.keras.utils import Sequence
from tensorflow import keras
import numpy as np


class DataGenerator(Sequence):
    def __int__(self, posters_list, genre, n_class=5, batch_size=32, dim=(32, 32, 32), n_channel=3, shuffle=True):
        self.posters_list = posters_list
        self.genre = genre
        self.batch_size = batch_size
        self.dim = dim
        self.n_channel = n_channel
        self.shuffle = shuffle
        self.n_class = n_class
        self.on_epoch_end()

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.posters_list))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_temp):
        X = np.empty((self.batch_size, *self.dim, self.n_channel))
        y = np.empty((self.batch_size, self.n_class))

        for i, f in enumerate(list_temp):
            X[i,] = cv2.imread("/home/zaher/DataFolder/Movie_Poster_Dataset/images/" + f + ".jpg")
            y[i] = self.genre[f]

        return X, y

    def __len__(self):
        return int(np.floor(len(self.posters_list) / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        posters_list_temp = [self.posters_list[k] for k in indexes]

        X, y = self.__data_generation(posters_list_temp)

        return X, y
