from tqdm import tqdm
import pickle

'''
This function is used by the training script as well as the word classification script
The reason it is defined in a extra file is to allow people to change the algorithm
while keeping the token list creation technique the same for the network during and after training
Change it however you want, just keep it in this structure:

tokens = {letter1 : number1,
          letter2 : number2,
          letter3 : number3}

The network HAS to be retrained if the token list sorting and numbering is changed!
'''

def create_token_list(to_tokenize):

    '''
    Arguments:
    to_tokenize: A big string containing all characters to be tokenized.
                 No letters should repeat, except capital and lowercase.
    '''
    
    tokens = {}
    for token, symbol in zip(tqdm(range(len(to_tokenize)), desc='Creating token list'), to_tokenize):
        tokens.update({symbol : token})
    pickle.dump(tokens, open('token_list.pickle', 'wb'))
    return tokens