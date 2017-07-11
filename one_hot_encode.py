import numpy as np

def encode(val, key):
    ret = np.repeat(0, len(key))
    try:
        ret[key.index(val)] = 1
    except ValueError:
        ret[1] = 1
    return ret

def encode_vals(vals, key):
    return [encode(val, key) for val in vals]

def decode(val, key):
    return key[val.argmax()]

def decode_vals(vals, key):
    return [decode(val, key) for val in vals]