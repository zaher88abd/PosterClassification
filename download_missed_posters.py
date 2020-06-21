import os
import pandas as pd
import urllib.request
import cv2
from PIL import Image

main_folder = "/home/zaher/DataFolder"
posters = "posters"
posters_new = "new_posters"
csv_file = "MovieGenre.csv"

df = pd.read_csv(os.path.join(main_folder, csv_file), encoding="ISO-8859-1")
indexes = []
# /home/zaher/DataFolder/posters/114709.jpg
for i, r in df.iterrows():
    img_name = os.path.join(main_folder, posters, str(r['imdbId']) + ".jpg")

    if cv2.imread(img_name) is None:
        indexes.append(i)

print(len(df), len(indexes))
df = df.drop(indexes)
print(len(df))

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

df['Genres'] = genre
df['Genres_cat'] = genre_cats

df = df.dropna(subset=['Genres_cat'])

df.to_csv(os.path.join(main_folder, "cleanData.csv"), header=True, index=False)
