import sys
sys.path.insert(0, '../')

from keras import losses
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.layers import Activation

import tensorflow as tf

import numpy as np

import anywords_traindata
from one_hot_encode import encode, decode


class Anywords:
    def load_data(self, path, seq_length):
        self.seq_length = seq_length
        data = anywords_traindata.load_data(path, seq_length)
        self.x_data, self.y_data, self.key = data

    def build_model(self):
        self.model = Sequential()
        self.model.add(LSTM(256,
                            input_shape=(self.x_data.shape[1],
                                         self.x_data.shape[2]),
                            return_sequences=True))
        self.model.add(Dropout(0.2))
        self.model.add(LSTM(256))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(self.y_data.shape[1]))
        self.model.add(Activation('softmax'))

    def compile_model(self):
        self.model.compile(loss=losses.categorical_crossentropy,
                           optimizer=optimizers.rmsprop(),
                           metrics=['accuracy'])
        self.graph = tf.get_default_graph()

    def load_weights(self, path):
        self.model.load_weights(path)
        self.graph = tf.get_default_graph()

    def train_model(self, batch_size, epochs, path):

        filepath = "./traindata/checkpoints/weights-{epoch:02d}.hdf5"
        checkpoint = ModelCheckpoint(filepath,
                                     monitor='loss',
                                     verbose=1,
                                     save_best_only=True,
                                     mode='min')
        callbacks_list = [checkpoint]

        self.model.fit(self.x_data,
                       self.y_data,
                       batch_size=batch_size,
                       epochs=epochs,
                       callbacks=callbacks_list)
        self.model.save_weights(path)

    def predict(self, ch=''):
        start_idx = 0
        if ch == '':
            start_idx = np.random.randint(0, len(self.x_data)-1)
        else:
            encoded_ch = encode(ch, self.key)

            for (i, x) in enumerate(self.x_data):
                if (x[0] == encoded_ch).all():
                    start_idx = i
                    break

        pattern = self.x_data[start_idx]
        result = ""

        for i in range(500):
            x = np.reshape(pattern, (1, pattern.shape[0], pattern.shape[1]))
            x = np.float_(x)

            with self.graph.as_default():
                pred = self.model.predict(x, verbose=0)
                result += decode(pred, self.key)
                pattern = np.concatenate((pattern, pred))
                pattern = pattern[1:len(pattern)]

        return result



