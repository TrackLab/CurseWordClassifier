from keras.models import load_model
import string
import numpy as np
import os
import re
import pickle
from TokenCreation import create_token_list

def convert_word_to_token(to_tokenize, token_list):
    token = []
    for symbol in to_tokenize:
        token.append(token_list[symbol])
    return token

# Creating Token List if it does not exist
# This needs to be the same pickle file as the one created from the training script!
# If the token list changes after training, the network will have no clue on what number is what number!
if os.path.exists('token_list.pickle'):
    token_list = pickle.load(open('token_list.pickle', 'rb'))
else:
    string_input =  string.printable + 'üöäÜÖÄ'
    token_list   =  create_token_list(string_input)


def load_network():
    networks = []
    for f in os.listdir('networks/'):
        if f.endswith('.h5'):
            networks.append(f)
    print(f'Found {len(networks)} networks.')
    print('-'*15)
    for network, i in zip(networks, range(len(networks))):
        print(f'{i} : {network}')
    try:
        selection = int(input(f"Please select the one to be used with the assosiated number 0 - {len(networks)-1} "))
        if selection > 0 and selection <= len(networks):
            # Gets the integers in the filename to extract max word length
            nums = int(re.search(r'\d+', networks[selection]).group())
            return load_model('networks/' + f), nums
        else:
            print("The number you used is not valid!")
            load_network()
    except:
        print("Please input a number!")
        load_network()

def classify_word(model, maxlength):

    word = input(f'Please type in your word (not longer than {maxlength} characters): ')
    if len(word) > maxlength:
        print(f'Sorry, the word can`t be longer than {maxlength} characters')
        classify_word(model, maxlength)
    elif len(word) <= 0:
        print("Sorry, the word must be atleast 1 character")
        classify_word(model, maxlength)
    else:
        word = convert_word_to_token(word, token_list)
        # Padding the word to be lenght of maxlength
        for _ in word:
            while len(word) != maxlength:
                word.append(0)
        word = np.array([word])
        p = model.predict(word)[0][0]
        print('-'*15)
        if p > 0.5:
            print('Normal')
        if p < 0.5:
            print('Curse')
        print('Probability: ', round(p, 5))
        print('-'*15)

model, maxlength = load_network()
while True:
    classify_word(model, maxlength)
