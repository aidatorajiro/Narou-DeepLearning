from __future__ import print_function
from keras.models import Sequential, model_from_json
from keras.layers import Dense, Activation
from keras.layers import LSTM
from keras.optimizers import RMSprop
from keras.utils.data_utils import get_file
import numpy as np
import random
import sys
import os
import pickle

path = os.path.abspath(os.path.dirname(__file__)) + "/jyukugo.txt"
text = open(path).read().lower()
path_seed = os.path.abspath(os.path.dirname(__file__)) + "/jyukugo2.txt"
text_seed = open(path_seed).read().lower()
print('corpus length:', len(text))

chars = sorted(list(set(text)))
print('total chars:', len(chars))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

maxlen = 40

print('Load model...')
model = model_from_json(open(sys.argv[1]).read())

print('Load params...')
model.load_weights(sys.argv[2])

def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

print('Generating...')
start_index = random.randint(0, len(text_seed) - maxlen - 1)
for diversity in [0.2, 0.5, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4]:
    print()
    print('----- diversity:', diversity)

    generated = ''
    sentence = text_seed[start_index: start_index + maxlen]
    generated += sentence
    print('----- Generating with seed: "' + sentence + '"')
    sys.stdout.write(generated)

    for i in range(400):
        x = np.zeros((1, maxlen, len(chars)))
        for t, char in enumerate(sentence):
            x[0, t, char_indices[char]] = 1.
        
        preds = model.predict(x, verbose=0)[0]
        next_index = sample(preds, diversity)
        next_char = indices_char[next_index]

        generated += next_char
        sentence = sentence[1:] + next_char

        sys.stdout.write(next_char)
        sys.stdout.flush()
    print()