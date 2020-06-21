import cv2
import os
from os.path import join as pjoin
from bson.json_util import loads, dumps
import pandas as pd
from pandas import json_normalize
from sklearn.model_selection import train_test_split
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, BatchNormalization, Activation
from tensorflow.keras.optimizers import Adam

from util import FunGen, DataGenerator
import tensorflow as tf
import tensorflow.keras as k
import numpy as np
from tensorflow.keras.applications import vgg19


def get_basic_model(input_shape, first_layer=6):
    vgg = vgg19.VGG19(weights="imagenet",
                      include_top=False, input_shape=input_shape)
    # or if we want to set the first 20 layers of the network to be non-trainable
    index = 1
    while index <= first_layer:
        layer = vgg.layers[index]
        if "conv" in layer.name:
            layer.trainable = False
            index += 1
        else:
            index += 1
            first_layer += 1
        print(layer.name, layer.trainable)

    return vgg


main_folder = "/home/zaher/DataFolder/"
csv_file = "cleanData.csv"
images = "posters"

df = pd.read_csv(pjoin(main_folder, csv_file), encoding="ISO-8859-1")

genre = []
genre_cats = []
genre_cats_dict = dict()
counter = 0
for i, r in df.iterrows():
    gs = []
    gs_cat = []
    f = str(r['imdbId'])
    f = "/home/zaher/DataFolder/Movie_Poster_Dataset/images/" + f + ".jpg"

    for g in str(r[4]).split('|'):
        if g not in genre_cats_dict.keys():
            genre_cats_dict[g] = counter
            counter += 1

        gs_cat.append(genre_cats_dict[g])
        gs.append(g)

    genre.append(gs)
    genre_cats.append(gs_cat)

print(genre_cats_dict)
# df.to_csv(pjoin(main_folder, "cleanData.csv"), index=False)
df['Genres'] = genre
df['Genres_cat'] = genre_cats
df = df[:25000]

split_ratio_testing = 30

split = int(len(df) * split_ratio_testing / 100)

X_train, X_test = df[split:], df[:split]

split_ratio_val = 25

split = int(len(df) * split_ratio_val / 100)

X_train, X_val = X_train[split:], X_train[:split]

print("Training Size ", len(X_train))
print("Val Size ", len(X_val))
print("Test Size", len(X_test))

X_train_img_id, X_train_genres = X_train['imdbId'], X_train['Genres_cat']
X_val_img_id, X_val_genres = X_val['imdbId'], X_val['Genres_cat']
X_test_img_id, X_test_genres = X_test['imdbId'], X_test['Genres_cat']

n_class = 29
batch_size = 32
params = {'dim': (150, 101),
          'batch_size': batch_size,
          'n_class': n_class,
          'n_channel': 3,
          'shuffle': True}

train_gen = FunGen(X_train_img_id.values, X_train_genres.values, **params).fun_gen()
val_gen = FunGen(X_val_img_id.values, X_val_genres.values, **params).fun_gen()
test_gen = FunGen(X_test_img_id.values, X_test_genres.values, **params).fun_gen()

chanDim = -1

model = Sequential()
model.add(Conv2D(32, (3, 3), padding="same", input_shape=(150, 101, 3)))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=chanDim))
model.add(MaxPooling2D(pool_size=(3, 3)))
model.add(Dropout(0.25))

# (CONV => RELU) * 2 => POOL
model.add(Conv2D(64, (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=chanDim))
model.add(Conv2D(64, (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=chanDim))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
# (CONV => RELU) * 2 => POOL
model.add(Conv2D(128, (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=chanDim))
model.add(Conv2D(128, (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=chanDim))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

# first (and only) set of FC => RELU layers
model.add(Flatten())
model.add(Dense(1024))
model.add(Activation("relu"))
model.add(BatchNormalization())
model.add(Dropout(0.5))

model.add(Dense(n_class, activation='sigmoid'))

INIT_LR = 1e-3
EPOCHS = 75
model.compile(Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS), loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

model.fit(x=train_gen, epochs=EPOCHS, validation_data=val_gen, verbose=1,
          steps_per_epoch=len(X_train_img_id) // batch_size,
          validation_steps=len(X_val_img_id) // batch_size)

output = model.evaluate(test_gen, steps=len(X_test) // batch_size)
print(np.floor(output))
