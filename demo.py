import os
import sys
import pickle
import gc
import xgboost as xgb
import numpy as np
import re
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

max_num_features = 10
pad_size = 1
boundary_letter = -1
space_letter = 0
max_data_size = 10000000


def context_window_transform(data, pad_size):
    pre = np.zeros(max_num_features)
    pre = [pre for x in np.arange(pad_size)]
    data = pre + data +	pre
    neo_data = []
    for i in tqdm(np.arange(len(data) - pad_size * 2)):
        row = []
        for x in data[i: i + pad_size * 2 + 1]:
            row.append([boundary_letter])
            row.append(x)
        row.append([boundary_letter])
        neo_data.append([int(x) for y in row for x in y])
    return neo_data


def main():
    sentence = input('Enter a sentence:\n')
    out_path = r'.'
    f = open('output.csv','w')
    sentence = sentence.strip()
    words = sentence.split()
    f.write('"before"\n')
    for word in words:
        f.write(word+"\n")
    f.close()
    df = pd.read_csv(r'output.csv',sep='\t')
    x_data = []
    gc.collect()

    for x in tqdm(df['before'].values):
        x_row = np.ones(max_num_features, dtype=int) * space_letter
        for xi, i in zip(list(str(x)), np.arange(max_num_features)):
            x_row[i] = ord(xi)
        x_data.append(x_row)




    x_data = x_data[:max_data_size]
    x_data = np.array(context_window_transform(x_data, pad_size))
    gc.collect()

    x_test = x_data
    gc.collect()

    gc.collect()
    dtest = xgb.DMatrix(x_test)
    bst = xgb.Booster({'nthread': 4})  # init model
    model = bst.load_model('../models/xgb_model.h5')
    gc.collect()
    labels = [u'PLAIN', u'PUNCT', u'DATE', u'LETTERS', u'CARDINAL', u'VERBATIM',
              u'DECIMAL', u'MEASURE', u'MONEY', u'ORDINAL', u'TIME', u'ELECTRONIC',
              u'DIGIT', u'FRACTION', u'TELEPHONE', u'ADDRESS']

    pred = bst.predict(dtest)
    pred = [labels[int(x)] for x in pred]
    pred_binary = []
    for item in pred:
        if item =='PLAIN':
            pred_binary.append('PLAIN')
        elif item == 'PUNCT':
            pred_binary.append('PUNCT')
        else:
            pred_binary.append('TN-token')
    print (list(zip(words,pred_binary)))
    sys.exit()

if __name__ == '__main__':
    main()
from datetime import datetime

timestamps = [
    datetime(2022, 12, 30),
    datetime(2023, 1, 31),
    datetime(2023, 4, 28),
    datetime(2023, 5, 31),
    datetime(2023, 6, 30)
]

ranges = []
start_date = timestamps[0]

for i in range(len(timestamps) - 1):
    if (timestamps[i + 1] - timestamps[i]).days > 31:
        ranges.append((start_date.strftime('%Y-%m-%d'), timestamps[i].strftime('%Y-%m-%d')))
        start_date = timestamps[i + 1]

# Adding the last range
ranges.append((start_date.strftime('%Y-%m-%d'), timestamps[-1].strftime('%Y-%m-%d')))

print(ranges)


[('2022-12-30', '2023-01-31'), ('2023-04-28', '2023-06-30')]

