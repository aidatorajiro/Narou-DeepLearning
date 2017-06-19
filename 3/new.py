from __future__ import print_function
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM
from keras.optimizers import RMSprop
from keras.utils.data_utils import get_file
import numpy as np
import random
import sys
import os
import pickle
import math

path = os.path.abspath(os.path.dirname(__file__)) + "/jyukugo.txt"
text = open(path).read().lower()
print('corpus length:', len(text))

chars = sorted(list(set(text)))
print('total chars:', len(chars))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

# cut the text in semi-redundant sequences of maxlen characters
max_seqs = 300000
maxlen = 40
step = 3
sentences = []
next_chars = []
for i in range(0, len(text) - maxlen, step):
    sentences.append(text[i: i + maxlen])
    next_chars.append(text[i + maxlen])
print('nb sequences(total):', len(sentences))


# build the model: a single LSTM
print('Build model...')
model = Sequential()
model.add(LSTM(128, input_shape=(maxlen, len(chars))))
model.add(Dense(len(chars)))
model.add(Activation('softmax'))

optimizer = RMSprop(lr=0.01)
model.compile(loss='categorical_crossentropy', optimizer=optimizer)


def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

# train the model, output generated text after each iteration
try:
    iteration = 0
    while True:
        for ft in range(0, math.ceil(len(sentences) / max_seqs)):
            print('Vectorization #' + str(ft) + '...')
            sentences_for_fit = sentences[ft*max_seqs:(ft+1)*max_seqs]
            X = np.zeros((len(sentences_for_fit), maxlen, len(chars)), dtype=np.bool)
            y = np.zeros((len(sentences_for_fit), len(chars)), dtype=np.bool)
            for i, sentence in enumerate(sentences_for_fit):
                for t, char in enumerate(sentence):
                    X[i, t, char_indices[char]] = 1
                y[i, char_indices[next_chars[i]]] = 1
            
            print()
            print('-' * 50)
            print('Iteration', iteration)
            model.fit(X, y, batch_size=128, nb_epoch=1)
                
            with open(os.path.abspath(os.path.dirname(__file__)) + '/model_' + str(iteration) + '_' + str(ft) + '.json', mode='w') as f:
                f.write(model.to_json())
            
            model.save_weights('param_' + str(iteration) + '_' + str(ft) + '.h5')
        
        iteration += 1

except KeyboardInterrupt:
    with open(os.path.abspath(os.path.dirname(__file__)) + '/model_int.json', mode='w') as f:
        f.write(model.to_json())
    model.save_weights('param_int')
    sys.exit()