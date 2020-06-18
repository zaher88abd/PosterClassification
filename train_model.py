import os
from os.path import join as pjoin
from bson.json_util import loads, dumps
import pandas as pd
from pandas import json_normalize
from sklearn.model_selection import train_test_split
from util import DataGenerator
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


main_folder = "/home/zaher/DataFolder/Movie_Poster_Dataset"
csv_file = "IMDB_DB.csv"
images = "images"

df = pd.read_csv(pjoin(main_folder, csv_file))

genre = []
genre_cats = []
genre_cats_dict = dict()
counter = 0
for x, i in df.iterrows():
    gs = []
    gs_cat = []
    f = str(i[7])
    f = "/home/zaher/DataFolder/Movie_Poster_Dataset/images/" + f + ".jpg"

    if not os.path.exists(f):
        gs = None
        gs_cat = None
        genre.append(gs)
        genre_cats.append(gs_cat)
        continue

    for g in str(i[5]).split(','):
        s = g
        s = s.lstrip()
        s = s.rstrip()
        if s == 'nan':
            gs = None
            gs_cat = None
            break
        if not s in genre_cats_dict.keys():
            genre_cats_dict[s] = counter
            counter += 1

        gs_cat.append(genre_cats_dict[s])
        gs.append(s)

    genre.append(gs)
    genre_cats.append(gs_cat)

print(genre_cats_dict)

df['Genres'] = genre
df['Genres_cat'] = genre_cats

df = df.dropna(subset=['Genres_cat'])

# df.to_csv("test.csv",header=True)
split_ratio = 10

split = int(len(df) * split_ratio / 100)

X_train, X_test = df[split:], df[:split]

split_ratio = 10

split = int(len(df) * split_ratio / 100)

X_train, X_val = X_train[split:], X_train[:split]

print(len(X_test), len(X_train), len(X_val))

X_train_img_id, X_train_genres = X_train['imdbID'], X_train['Genres_cat']
X_val_img_id, X_val_genres = X_val['imdbID'], X_val['Genres_cat']
X_test_img_id, X_test_genres = X_test['imdbID'], X_test['Genres_cat']

params = {'dim': (255, 255),
          'batch_size': 32,
          'n_class': 27,
          'n_channel': 3,
          'shuffle': True}

train_gen = DataGenerator(X_train_img_id.values, X_train_genres.values, **params)
val_gen = DataGenerator(X_val_img_id.values, X_val_genres.values, **params)
test_gen = DataGenerator(X_test_img_id.values, X_test_genres.values, **params)

model = k.Sequential()
model.add(k.layers.Conv2D(filters=16, kernel_size=(5, 5), activation="relu", input_shape=(255,255, 3)))
model.add(k.layers.Conv2D(filters=16, kernel_size=(5, 5), activation="relu"))
model.add(k.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(k.layers.Dropout(0.25))
model.add(k.layers.Conv2D(filters=32, kernel_size=(5, 5), activation='relu'))
model.add(k.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(k.layers.Dropout(0.25))
model.add(k.layers.Conv2D(filters=64, kernel_size=(5, 5), activation="relu"))
model.add(k.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(k.layers.Dropout(0.25))
model.add(k.layers.Conv2D(filters=64, kernel_size=(5, 5), activation='relu'))
model.add(k.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(k.layers.Dropout(0.25))
model.add(k.layers.Flatten())
model.add(k.layers.Dense(128, activation='relu'))
model.add(k.layers.Dropout(0.5))
model.add(k.layers.Dense(64, activation='relu'))
model.add(k.layers.Dropout(0.5))
model.add(k.layers.Dense(27, activation='sigmoid'))

model.compile(tf.optimizers.Adam(lr=0.01), loss='binary_crossentropy', metrics=['accuracy'])
# model.summary()
model.fit(x=train_gen, epochs=50, validation_data=val_gen, verbose=1)
