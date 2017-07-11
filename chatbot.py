import numpy as np
from seq2seq.models import Seq2Seq

from one_hot_encode import encode_str, decode_str

max_len = 32
input_length = max_len
input_dim = 128

output_length = max_len
output_dim = 128

hidden_dim = 256

model = Seq2Seq(input_length=input_length,
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                output_length=output_length,
                output_dim=output_dim,
                depth=1)
model.load_weights('trained_weights')

while True:
    q = input('> ')
    result = model.predict(np.array([encode_str(q.ljust(max_len, ';'))[::-1]]))
    print(''.join(decode_str(result[0])))