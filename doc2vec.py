"""Emotion recognition on Twitter datasets."""
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
from gensim import utils  # noqa
from gensim.models.doc2vec import TaggedDocument  # noqa
from gensim.models import Doc2Vec  # noqa

from tqdm import tqdm  # noqa

import numpy as np  # noqa
from sklearn.linear_model import LogisticRegression  # noqa
import random  # noqa
import sys  # noqa
import datetime  # noqa
import time  # noqa

start_time = time.clock()

emotions = ["ANGER", "HAPPINESS", "SADNESS", "NEUTRAL", "HATE", "FUN", "LOVE"]
EPOCHS = 20
VEC_DIMENSIONS = 400
WINDOW_SIZE = 25


def build_dictionary():
    """Build the dictionary."""
    formatted_tweets = []
    print("\nOpening dictionary...\n")
    tweets = open(
        "./Sentiment-Analysis-Dataset/formatted_corpus.txt",
        "r",
        encoding="utf-8"
    ).read().splitlines()
    for index, tweet in tqdm(enumerate(tweets)):
        formatted_tweets.append(
            TaggedDocument(utils.simple_preprocess(tweet), [index])
        )
    return formatted_tweets


def build_dataset():
    """Build the dataset."""
    formatted_tweets = []
    classes = []
    indexes = []
    print("\nOpening dataset...\n")
    tweets = open(
        "./Training_set.txt",
        "r",
        encoding="utf-8"
    ).read().splitlines()
    for index, tweet in tqdm(enumerate(tweets)):
        tweet_data = tweet.split(" ///")
        classes.append(emotions.index(tweet_data[0]))
        indexes.append(str(index))
        formatted_tweets.append(tweet_data[1])
    return list(formatted_tweets), len(tweets), classes, indexes


def build_model(training_data):
    """Train the model."""
    model = Doc2Vec(
        min_count=1,
        window=WINDOW_SIZE,
        size=VEC_DIMENSIONS,
        sample=1e-4,
        negative=5,
        workers=7)
    model.build_vocab(training_data)
    for epoch in tqdm(range(EPOCHS)):
        shuffled = list(training_data)
        random.shuffle(shuffled)
        model.train(shuffled, total_words=n_count, epochs=epoch)
    model.save('./dicts/dict_full_data_%se_%sw.d2v' % (EPOCHS, WINDOW_SIZE))
    print('Model saved in %ss' % (round(time.clock() - start_time, 2)))
    return model


training_data, n_count, classes, indexes = build_dataset()
if "-b" in sys.argv:
    dictionary = build_dictionary()
    '''
    open("formatted_tweets.txt", "w", encoding="utf-8").write(
        "%s" % dictionary
    )
    '''
    model = build_model(dictionary)
else:
    model = Doc2Vec.load("./dicts/dict_full_data_3e_5w.d2v")
print("Most similar to \"good\": %s" % (model.most_similar("pizza")))

train_arrays = np.zeros((n_count, VEC_DIMENSIONS))
train_labels = np.zeros(n_count)
test_arrays = np.zeros((n_count, VEC_DIMENSIONS))
test_labels = np.zeros(n_count)

for i in tqdm(range(n_count)):
    train_arrays[i] = model.infer_vector(training_data[i])
    train_labels[i] = classes[i]
    test_arrays[i] = model.infer_vector(training_data[i])
    test_labels[i] = classes[i]

classifier = LogisticRegression()
classifier.fit(train_arrays, train_labels)
accuracy = classifier.score(test_arrays, test_labels)
print("Accuracy: %s" % (accuracy))
with open("log.txt", "a") as log:
    log.write(
        "%s: {"
        "epochs: %s,"
        " window: %s,"
        " time: %s,"
        "accuracy: %s,"
        "dictionary_built: true"
        "}\n" % (
            datetime.datetime.now(),
            EPOCHS,
            WINDOW_SIZE,
            round(time.clock() - start_time, 2),
            accuracy
        )
    )

if "-p" in sys.argv:
    input_phrase = input("What\'s up today ?\n")
    inferred_vector = model.infer_vector(input_phrase.split())
    sims = model.docvecs.most_similar(
        [inferred_vector], topn=len(model.docvecs))
    print("Estimated emotion: " + emotions[
        classifier.predict([inferred_vector])[0].astype(int)])
else:
    # print("MOST SIMILAR WORDS:" + str(model.most_similar("pizza")))
    # print("Most similar vector: %s %s %s" % (
    #    sims[0], " : ", training_data[sims[0][0]]))
    print("Estimated emotion: %s" % (
        emotions[classifier.predict([train_arrays[1000]])[0].astype(int)]))
