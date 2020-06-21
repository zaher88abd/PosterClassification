import os
import cv2
import image_functions
from tensorflow.keras.utils import Sequence
from tensorflow.keras.preprocessing import image
import numpy as np


class DataGenerator(Sequence):
    def __init__(self, posters_list, genre_list, n_class=5, batch_size=32, dim=(32, 32), n_channel=3, shuffle=True):
        self.posters_list = posters_list
        self.genre_list = genre_list
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

    def __get_label(self, genre):
        y = np.zeros((self.n_class), dtype="uint8")

        for i in genre:
            y[i] = 1
        return y

    def __read_image(self, f):
        f = "/home/zaher/DataFolder/posters/" + str(f) + ".jpg"
        img = image.load_img(f, target_size=(self.dim[0], self.dim[1]))
        img = image.img_to_array(img)
        img = img.astype(np.float32)
        img = img / 255
        rand = np.random.randn()
        if rand < 0.25:
            img = image_functions.flip(img, vflip=True)
        elif 0.50 > rand > 0.25:
            img = image_functions.flip(img, hflip=True)
        return img

    def __data_generation(self, list_temp):
        X = np.empty((self.batch_size, *self.dim, self.n_channel))
        y = np.empty((self.batch_size, self.n_class), dtype='uint8')

        for i, f in enumerate(list_temp):
            X[i, :] = self.__read_image(f)

            y[i] = self.__get_label(self.genre_list[i])
        # print(X.shape,y.shape)
        return X, y

    def __len__(self):
        return int(np.floor(len(self.posters_list) / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        posters_list_temp = [self.posters_list[k] for k in indexes]

        X, y = self.__data_generation(posters_list_temp)

        return X, y


class FunGen():
    def __init__(self, posters_list, genre_list, n_class=5, batch_size=32, dim=(32, 32), n_channel=3, shuffle=True):
        self.posters_list = posters_list
        self.genre_list = genre_list
        self.batch_size = batch_size
        self.dim = dim
        self.n_channel = n_channel
        self.shuffle = shuffle
        self.n_class = n_class

    def __get_label(self, genre):
        y = np.zeros((self.n_class), dtype=int)
        for i in genre:
            y[i] = 1
        return y

    def __read_image(self, f):
        f = "/home/zaher/DataFolder/posters/" + str(f) + ".jpg"
        img = image.load_img(f, target_size=(self.dim[0], self.dim[1]))
        img = image.img_to_array(img)
        if img is None:
            print("rrrerer")
        img = img.astype(np.float32)
        img = img / 255.0
        rand = np.random.randn()
        if rand < 0.25:
            img = image_functions.flip(img, vflip=True)
        elif 0.50 > rand > 0.25:
            img = image_functions.flip(img, hflip=True)
        return img

    def fun_gen(self):
        self.indexes = np.arange(len(self.posters_list))
        if self.shuffle:
            # print("Randomize data")
            np.random.shuffle(self.indexes)
        counter = 0
        while True:

            X = np.empty((self.batch_size, *self.dim, self.n_channel))
            y = np.empty((self.batch_size, self.n_class), dtype="uint8")
            z = []

            d = []
            for i in range(self.batch_size):
                X[i, :] = self.__read_image(self.posters_list[self.indexes[counter]])
                y[i] = self.__get_label(self.genre_list[self.indexes[counter]])
                z.append(self.genre_list[self.indexes[counter]])
                d.append(self.posters_list[self.indexes[counter]])
                counter += 1
                if counter == len(self.indexes):
                    counter = 0

            yield X, y
