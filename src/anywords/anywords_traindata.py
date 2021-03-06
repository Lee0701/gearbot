import sys
sys.path.insert(0, '../')

import numpy as np

from one_hot_encode import encode, encode_vals


def load_data(path, seq_length):
    with open(path) as file:
        content = file.read().strip()
        key = sorted(list(set(content)))

        dataX = []
        dataY = []

        for i in range(0, len(content) - seq_length, 1):
            seq_in = content[i:i+seq_length]
            seq_out = content[i+seq_length]
            dataX.append(encode_vals(seq_in, key))
            dataY.append(encode(seq_out, key))

        X = np.reshape(dataX, (len(dataX), seq_length, len(key)))
        X = np.float_(X)
        Y = np.asarray(dataY)

        return (X, Y, key)
