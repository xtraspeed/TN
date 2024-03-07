nonoimport os
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

import pandas as pd

# Sample left dataframe
left_data = {
    'key': ['A', 'B', 'C', 'D', 'E'],
    'date': ['2024-01-01', '2024-02-15', '2024-03-20', '2024-04-10', '2024-05-25']
}
df_left = pd.DataFrame(left_data)

# Sample right dataframe
right_data = {
    'key': ['A', 'B', 'C', 'D', 'E'],
    'start_date': ['2024-01-01', '2024-02-01', '2024-03-15', '2024-04-05', '2024-05-20'],
    'end_date': ['2024-01-31', '2024-02-28', '2024-03-31', '2024-04-30', '2024-05-31']
}
df_right = pd.DataFrame(right_data)

# Define chunk size
chunk_size = 2

# Initialize previous_chunk_state
previous_chunk_state = None

# Iterate over chunks of left dataframe
for left_chunk in pd.read_csv("left_dataframe.csv", chunksize=chunk_size):
    # Initialize current_chunk_state
    current_chunk_state = None
    # Iterate over chunks of right dataframe
    for right_chunk in pd.read_csv("right_dataframe.csv", chunksize=chunk_size):
        # If there's a previous chunk's state, merge it with the current right chunk
        if previous_chunk_state is not None:
            right_chunk = pd.concat([previous_chunk_state, right_chunk], ignore_index=True)
        # Perform inner join operation on the current chunks
        merged_chunk = pd.merge(left_chunk, right_chunk, on='key', how='inner')
        # Filter merged_chunk based on date condition
        filtered_chunk = merged_chunk[(merged_chunk['date'] >= merged_chunk['start_date']) & 
                                      (merged_chunk['date'] <= merged_chunk['end_date'])]
        # Store the end state of the current right chunk for the next iteration
        current_chunk_state = right_chunk.iloc[filtered_chunk.index[-1] + 1:]
        # Process the filtered_chunk as needed
        print(filtered_chunk)
    # Update previous_chunk_state for the next iteration
    previous_chunk_state = current_chunk_state
        
