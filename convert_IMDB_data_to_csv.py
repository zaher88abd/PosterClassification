import os
import sys
from os.path import join as pjoin
from bson.json_util import loads, dumps
import pandas as pd
from pandas import json_normalize
import io

main_folder = "/home/zaher/DataFolder"
meta_data = "./groundtruth"
poster_data = "Movie_Poster_Dataset"


def readingJsonFile(file):
    lines = ""
    try:
        with io.open(file, 'r', encoding='utf-16') as f:
            for line in f.readlines():

                if line is None:
                    break
                if not line.strip().startswith("\"_id\""):
                    lines += line + "\n"
    except:
        with io.open(file, 'r', encoding='utf-8') as f:
            for line in f.readlines():

                if line is None:
                    break
                if not line.strip().startswith("\"_id\""):
                    lines += line + "\n"

    contents = lines.replace("}", "},").rstrip()[:-1]

    contents = "[" + contents + "]"
    # print(contents)
    # contents=dumps(contents)
    contents = loads(contents)
    # print(type(contents))

    df = pd.DataFrame(json_normalize(contents))
    # print(type(df))
    return df


def joinDFs(mainDF, df):
    if mainDF is None:
        return df
    mainDF = pd.concat([mainDF, df])
    return mainDF


def main(arag):
    mainDF = None
    count = 1
    for f in os.listdir(pjoin(meta_data)):
        print(f, count)
        mainDF = joinDFs(mainDF, readingJsonFile(pjoin(meta_data, f)))
        # print(len(mainDF))
        count += 1

    mainDF.to_csv("./" + arag[0], index=False)


if __name__ == '__main__':
    main(arag=sys.argv)
