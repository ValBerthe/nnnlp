# Neural Network for Natural Language Processing: Emotion classification on Twitter
## A machine-learning approach to detect emotions out of tweets.

You can send an email at: valentin@ohtsuki.ics.keio.ac.jp for the different dictionaries, or the original tweet corpus if you want to build your own.

Python dependencies:

- numpy
- tensorflow

## Usage

You can find the pre-trained embedded vectors on the stanford.edu platform: https://nlp.stanford.edu/projects/glove/
It is recommended to download the four pre-trained vector set trained from Twitter (dimensions 25, 50, 100, 200).
All the tensorboards results are stored in the /runs dir.

`model.py` describes the convolutional neural network, and `classifier.py` is the glove implementation.

Flags:

- `-b`: embeds the training_data and the test_data. It saves it in the dir `/bin`, so you might want to create it first.
