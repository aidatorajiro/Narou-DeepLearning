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
import re

path = os.path.abspath(os.path.dirname(__file__)) + "/jyukugo.txt"
text = open(path).read().lower()
print('corpus length:', len(text))

chars = sorted(list(set(text)))
print('total chars:', len(chars))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

# cut the text in semi-redundant sequences of maxlen characters
maxlen = 40
step = 3
sentences = []
next_chars = []
for i in range(0, len(text) - maxlen, step):
    sentences.append(text[i: i + maxlen])
    next_chars.append(text[i + maxlen])
print('nb sequences:', len(sentences))

print('Vectorization...')
X = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        X[i, t, char_indices[char]] = 1
    y[i, char_indices[next_chars[i]]] = 1


print('Load model...')
current_model_id = max(map(lambda x: int(x.group(1)), filter(lambda x: x != None, map(lambda x: re.match(r"model_(\d+)\.json", x), os.listdir(os.path.abspath(os.path.dirname(__file__)))))))
model = model_from_json(open("model_" + str(current_model_id) + ".json").read())

optimizer = RMSprop(lr=0.01)
model.compile(loss='categorical_crossentropy', optimizer=optimizer)


print('Load params...')
model.load_weights("param_" + str(current_model_id))


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
    iteration = current_model_id
    while True:
        iteration += 1
        
        print()
        print('-' * 50)
        print('Iteration', iteration)
        model.fit(X, y, batch_size=128, nb_epoch=1)
    
        start_index = random.randint(0, len(text) - maxlen - 1)
    
        for diversity in [0.2, 0.5, 1.0, 1.2]:
            print()
            print('----- diversity:', diversity)
    
            generated = ''
            sentence = text[start_index: start_index + maxlen]
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
        
        with open(os.path.abspath(os.path.dirname(__file__)) + '/model_' + str(iteration) + '.json', mode='w') as f:
            f.write(model.to_json())
            
        model.save_weights('param_' + str(iteration))
        
except KeyboardInterrupt:
    with open(os.path.abspath(os.path.dirname(__file__)) + '/model_int.json', mode='w') as f:
        f.write(model.to_json())
    model.save_weights('param_int')
    sys.exit()