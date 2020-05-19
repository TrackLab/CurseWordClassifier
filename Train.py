import string
import os
import numpy as np
import datetime
import pickle
from sklearn import utils
from tqdm import tqdm
from keras.models import Sequential, load_model
from keras.layers import Dense
from keras import optimizers
from matplotlib import pyplot
from TokenCreation import create_token_list

if not os.path.exists('networks/'):
    os.makedirs('networks/')
if not os.path.exists('images/'):
    os.makedirs('images/')

use_validation = True
plot_each_epoch = False
# More than 250 epochs seem to not improve anything anymore above 95% accuracy (Atleast with the current network setup)
epochs = 250
batch_size = 32

def convert_symbol_to_token(to_tokenize, token_list):
    return token_list[to_tokenize]

def convert_word_list_to_tokens(filename, token_list):
    tokenized_list = []
    with open(filename, 'r') as f:
        for line in tqdm(f, desc='Converting word list to tokens'):
            for word in line.split():
                tokenized_word = []
                for letter in word:
                    token = convert_symbol_to_token(letter, token_list)
                    tokenized_word.append(token)
            tokenized_list.append(tokenized_word)
    return tokenized_list

def zero_padding(Input):

    # Find longest word
    longest_length = 0
    for word in tqdm(Input, desc='Finding longest word'):
        if len(word) > longest_length:
            longest_length = len(word)
    print(f'Longest word has length of {longest_length}')

    # Zero Padding
    padded_words_input = []
    for word in tqdm(Input, desc=f'Zero Padding to length {longest_length}'):
        while len(word) != longest_length:
            word.append(0)
        padded_words_input.append(word)
    zero_padded_input = padded_words_input
    return zero_padded_input, longest_length

# Creating Token List if it does not exist
if os.path.exists('token_list.pickle'):
    token_list = pickle.load(open('token_list.pickle', 'rb'))
else:
    string_input =  string.printable + 'üöäÜÖÄ'
    token_list   =  create_token_list(string_input)

# Converting both word lists to tokens. This is done within 1 second so it will be repeated on every run
bad_words_input =  convert_word_list_to_tokens('dataset/bad_words.txt', token_list)
good_words_input = convert_word_list_to_tokens('dataset/good_words.txt', token_list)
print(f'{len(bad_words_input)} Bad Words | {len(good_words_input)} Good Words')

# Creating the labels for both word lists | 0 for bad | 1 for good
bad_words_label  = [0 for i in range(len(bad_words_input))]
good_words_label = [1 for i in range(len(good_words_input))]

# Putting seperated lists together
words_label = bad_words_label + good_words_label
words_input = bad_words_input + good_words_input

# Zero padding
words_input, longest_length = zero_padding(words_input)

# Shuffling the data
seed = 0
words_label = utils.shuffle(words_label, random_state=seed)
words_input = utils.shuffle(words_input, random_state=seed)

# Converting to Numpy Array
words_label = np.array(words_label)
words_input = np.array(words_input)

# Optimizer
adam = optimizers.Adam(learning_rate=0.001)

# Network
network = Sequential()
network.add(Dense(units=2048, activation='relu', input_shape=(longest_length,)))
network.add(Dense(units=1024, activation='relu'))
network.add(Dense(units=512, activation='relu'))
network.add(Dense(units=128, activation='relu'))
network.add(Dense(units=1, activation='sigmoid'))
network.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])
network.summary()

# If you want to train using validation data. See variable use_validation on top
if use_validation:
    history = network.fit(words_input, words_label, epochs=epochs, batch_size=batch_size, shuffle=False, validation_split=0.1)
else:
    history = network.fit(words_input, words_label, epochs=epochs, batch_size=batch_size, shuffle=False)

# If you want to save each epoch as single image for over time visualization
# Will be done after training | Can take a while!
# If plot_each_epoch is False, only one image will be saved to visualize the entire training process
if plot_each_epoch:
    for i in tqdm(range(len(history.history['accuracy']))):
        # plot loss during training
        pyplot.subplot(211)
        pyplot.title('Loss')
        pyplot.plot(history.history['loss'][0:i], label='Loss')
        pyplot.legend()
        # plot accuracy during training
        pyplot.subplot(212)
        pyplot.title('Accuracy')
        pyplot.plot(history.history['accuracy'][0:i], label='Accuracy')
        pyplot.legend()

        # To plot validation loss and accuracy
        if use_validation:
            # plot valid accuracy during training
            pyplot.subplot(212)
            pyplot.plot(history.history['val_accuracy'][0:i], label='Validation Accuracy')
            pyplot.legend()
            # plot valid loss during training
            pyplot.subplot(211)
            pyplot.plot(history.history['val_loss'][0:i], label='Validation Loss')
            pyplot.legend()
        pyplot.savefig(f'images/{i}.png', dpi=500)
        pyplot.clf()
else:
    # plot loss during training
    pyplot.subplot(211)
    pyplot.title('Loss')
    pyplot.plot(history.history['loss'], label='Loss')
    pyplot.legend()
    # plot accuracy during training
    pyplot.subplot(212)
    pyplot.title('Accuracy')
    pyplot.plot(history.history['accuracy'], label='Accuracy')
    pyplot.legend()

    # To plot validation loss and accuracy
    if use_validation:
        # plot valid accuracy during training
        pyplot.subplot(212)
        pyplot.plot(history.history['val_accuracy'], label='Validation Accuracy')
        pyplot.legend()
        # plot valid loss during training
        pyplot.subplot(211)
        pyplot.plot(history.history['val_loss'], label='Validation Loss')
        pyplot.legend()

    end_accuracy = history.history['accuracy'][-1]
    end_accuracy = round(round(end_accuracy, 2) * 100, 1)
    pyplot.savefig(f'validation_data-epochs-{epochs}-accuracy-{end_accuracy}%.png', dpi=500)
  
# After all is done, saving the network. Can then be used in classify_word.py
network.save(f'networks/{longest_length}-network-{end_accuracy}%.h5')

