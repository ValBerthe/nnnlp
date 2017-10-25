# Neural Network for Natural Language Processing: Emotion classification on Twitter
## A machine-learning approach to detect emotions out of tweets.

You can send an email at: valentin@ohtsuki.ics.keio.ac.jp for the different dictionaries, or the original tweet corpus if you want to build your own.

Python dependencies:

- gensim
- numpy
- sklearn
- tqdm

## Usage

As soon as you have the corpus of words, you can start training the dictionary.

Flags:

- -b: builds the dictionary. If you omit this flag, it will take the dictionary in the folder `dicts`, so choose the one that has the best results.
- -c: will build the linear regression model. You can build a new one when you build another dictionary. If you omit this flag, `classifier.sav` will be used as the LR model.
- -p: Allows you to test (classify) some sentences (tweets) by yourself.
