import numpy as np
from konlpy.tag import Mecab

from seq2seq.models import Seq2Seq

from one_hot_encode import encode_vals, decode_vals
import train_data

hn = Mecab()

questions, answers, max_len, key = train_data.load_data('../traindata/data.txt')

input_length = max_len
input_dim = len(key)

output_length = max_len
output_dim = len(key)

hidden_dim = output_dim * 2

model = Seq2Seq(input_length=input_length,
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                output_length=output_length,
                output_dim=output_dim,
                depth=1)
model.load_weights('trained_weights')

while True:
    q = input('> ')
    q_tok = hn.morphs(q)
    q_tok += [';'] * (max_len - len(q_tok))
    result = model.predict(np.array([encode_vals(q_tok, key)[::-1]]))
    dec_result = decode_vals(result[0], key)
    print(' '.join(list(filter(lambda w: w != ';', dec_result))))
