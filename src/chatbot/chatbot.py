import sys
sys.path.insert(0, '../')

from keras import losses
from keras import optimizers

import numpy as np

from konlpy.tag import Mecab

from seq2seq.models import Seq2Seq

import chatbot_traindata
from one_hot_encode import encode_vals, decode_vals


class Chatbot:
    def __init__(self):
        self.hn = Mecab()

    def load_data(self, path):
        data = chatbot_traindata.load_data(path)
        self.questions, self.answers, self.max_len, self.key = data
        # ../traindata/data.txt

        self.input_length = self.max_len
        self.input_dim = len(self.key)

        self.output_length = self.max_len
        self.output_dim = len(self.key)

        self.hidden_dim = self.output_dim * 2

        self.x_data = self.questions
        self.y_data = self.answers

    def build_model(self):
        self.model = Seq2Seq(input_length=self.input_length,
                             input_dim=self.input_dim,
                             hidden_dim=self.hidden_dim,
                             output_length=self.output_length,
                             output_dim=self.output_dim,
                             depth=1)

    def compile_model(self):
        self.model.compile(loss=losses.mean_squared_error,
                           optimizer=optimizers.rmsprop(),
                           metrics=['accuracy'])

    def load_weights(self, path):
        self.model.load_weights(path)

    def train_model(self, batch_size, epochs, path):
        self.model.fit(np.array(self.x_data),
                       np.array(self.y_data),
                       batch_size=batch_size,
                       epochs=epochs)
        self.model.save_weights(path)
        # ../traindata/trained_weights

    def predict(self, q):
        tokens = self.hn.morphs(q)
        tokens += [';'] * (self.max_len - len(tokens))
        input_data = np.array([encode_vals(tokens, self.key)[::-1]])
        result = self.model.predict(input_data)
        dec_result = decode_vals(result[0], self.key)
        return (' '.join(list(filter(lambda w: w != ';', dec_result))))
