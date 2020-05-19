# Curse Word Classifier
# Made by https://github.com/TrackLab

This repository contains code to train a new network, use pre-existing ones and to visualize training.

The network trains with 2 text files.
One contains only good words, the other only contains bad words.
You can add new words to your liking.

So far, the dataset contains 2000 good and 2000 bad words. Giving you a perfect 50/50 split.
The script creates a token list for the network to use with the "TokenCreation.py" script.
If you wish to edit the token creation algorithm, you can do so in that file.
It also checks for the longest word in the entire dataset, and zero padds every other word to the same length.

The current dataset has a maximum length of 23 characters.
Should you add any words that are longer, the maximum length extends to that words lengths.

Classifying words will be possible up to the length that the longest word during training had. The "ClassifyWord.py" script informs you about that.

In the train script you can change several variables, to adjust the network or the way the training is visualized.
The repository comes with 1 pre-trained network that you can use.
