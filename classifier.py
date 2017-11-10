"""Emotion recognition on Twitter datasets."""
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
from gensim import utils  # noqa
from gensim.models.doc2vec import TaggedDocument  # noqa
from gensim.models import Doc2Vec  # noqa

from tqdm import tqdm  # noqa

import numpy as np  # noqa
from sklearn.linear_model import LogisticRegression  # noqa
from sklearn import svm  # noqa
import random  # noqa
import sys  # noqa
import datetime  # noqa
import time  # noqa
import pickle # noqa
import re  # noqa


class classifier:
    """Classify a tweet in several given emotions."""

    def format_tweet(self, tweet):
        """Format the weet according to the dictionary's' rules."""
        def format_hashtag(hashtag):
            hashtag_body = hashtag.group(0)[1:]
            if hashtag_body.upper() == hashtag_body:
                result = "<HASHTAG> %s <ALLCAPS>" % (hashtag_body)
            else:
                result = " ".join(["<HASHTAG>"] + hashtag_body.split(
                    r"(?=[A-Z])")
                )
            return result

        def format_punct_repetition(mark):
            """Format punctuation rep. according to the dictionary's rules."""
            return "%s <REPEAT>" % mark.group(0)

        def format_elongated_words(part1, part2):
            """Format elongated words according to the dictionary's rules."""
            return "%s%s <ELONG>" % (part1.group(0), part2.group(0))

        def format_downcase(word):
            """Format an all capital word to a lowercase word."""
            return "%s" % word.group(0).lower()
        eyes = "[8:=;]"
        nose = "['`\-]?"
        tweet = re.sub(r"https?:\/\/\S+\b|www\.(\w+\.)+\S*", "<URL>", tweet)
        tweet = re.sub("/", " / ", tweet)
        tweet = re.sub(r"@\w+", "<USER>", tweet)
        tweet = re.sub(
            r"%s%s[)d]+|[)d]+%s%s" % (eyes, nose, nose, eyes), "<SMILE>", tweet
        )
        tweet = re.sub(r"%s%sp+" % (eyes, nose), "<LOLFACE>", tweet)
        tweet = re.sub(
            r"%s%s\(+|\)+%s%s" % (eyes, nose, nose, eyes), "<SADFACE>", tweet
        )
        tweet = re.sub(r"%s%s[\/|l*]" % (eyes, nose), "<NEUTRALFACE>", tweet)
        tweet = re.sub(r"<3", "<HEART>", tweet)
        tweet = re.sub(r"[-+]?[.\d]*[\d]+[:,.\d]*", "<NUMBER>", tweet)
        tweet = re.sub(r"\#\S+", format_hashtag, tweet)
        tweet = re.sub(r"([!?.]){2,}", format_punct_repetition, tweet)
        # re.sub(r"\b(\S*?)(.)\2{2,}\b", format_elongated_words, tweet)
        tweet = re.sub(r"([^a-z0-9()<>'`\-]){2,}", format_downcase, tweet)
        return tweet

    def build_dictionary(self):
        """Build the dictionary."""
        # start_time = time.clock()
        text_vectors = open(
            self.GLOVE_PATH, "r", encoding="utf-8"
        ).read().splitlines()
        words = [""] * len(text_vectors)
        embeddings = np.zeros((len(text_vectors), self.VEC_DIMENSIONS))
        regex1 = re.compile(r"^([^ ])+")
        regex2 = re.compile(r"(\-?[0-9]+(\.[0-9]+(e-[0-9]+)?)? ?)+")
        for index, line in tqdm(enumerate(text_vectors)):
            print(index)
            words[index] = regex1.search(line).group(0)
            embeddings[index] = regex2.search(line).group(0).split()
        return words, embeddings

    def build_dataset(self, data_file):
        """Build the dataset."""
        print("\nOpening dataset...\n")
        tweets = open(
            data_file,
            "r",
            encoding="utf-8"
        ).read().splitlines()
        embeddings = [""] * len(tweets)
        classes = [""] * len(tweets)
        for index, tweet in tqdm(enumerate(tweets)):
            tweet_data = tweet.split(" ///")
            class_vec = np.zeros(7)
            class_vec[self.emotions.index(tweet_data[0])] = 1
            classes[index] = class_vec
            embeddings[index] = self.lookup_tweet(
                self.format_tweet(tweet_data[1])
            )
        return embeddings, classes

    def lookup_tweet(self, tweet):
        """Lookup table for every words in the tweet."""
        tokenized_tweet = self.format_tweet(tweet)
        tokenized_words = tokenized_tweet.split()
        embedded_tweet = [[]] * len(tokenized_words)
        for i, word in enumerate(tokenized_words):
            if word in self.dict_words:
                embedded_tweet[i] = self.dict_embeddings[
                    self.dict_words.index(word)
                ]
            else:
                embedded_tweet[i] = self.dict_embeddings[
                    self.dict_words.index("<unknown>")
                ]
        return embedded_tweet

    def pad_dataset(self, dataset):
        """Add padding to the dataset so the sequence length is fixed."""
        padded_dataset = np.zeros((
            len(dataset),
            self.maxlen,
            self.VEC_DIMENSIONS
        ))
        for index, tweet in enumerate(dataset):
            padded_dataset[index][:len(tweet)] = np.array(tweet)
        return padded_dataset

    def __init__(self):
        """Execute primary functions and define constants."""
        # Constants accessible in "model.py"
        self.emotions = [
            "ANGER",
            "HAPPINESS",
            "SADNESS",
            "NEUTRAL",
            "HATE",
            "FUN",
            "LOVE"
        ]
        # Choose Embedding dimensions among the values : [25, 50, 100, 200]
        self.VEC_DIMENSIONS = 100
        self.DICT_PATH = "./Sentiment-Analysis-Dataset/formatted_corpus.txt"
        self.TRAINING_SET_PATH = "./Training_set.txt"
        self.TEST_SET_PATH = "./Test_set.txt"
        self.CURRENT_DICT_PATH = "./dicts/dict_full_data_40e_20w.d2v"
        self.WORD_TO_TEST = "good"
        self.CLASSIFIER_PATH = "classifier.sav"
        self.GLOVE_PATH = "glove.twitter.27B.%sd.txt" % self.VEC_DIMENSIONS
        # If "-b" is specified in the command line, then build the embedded
        # vectors from the two datasets and dump them. Else, load the
        # serialized objects.

        if "-b" in sys.argv:
            self.dict_words, self.dict_embeddings = self.build_dictionary()
            self.training_data, self.training_classes = self.build_dataset(
                self.TRAINING_SET_PATH
            )
            self.test_data, self.test_classes = self.build_dataset(
                self.TEST_SET_PATH
            )
            pickle.dump(
                self.dict_words,
                open("bin/dict_words%s.pkl" % self.VEC_DIMENSIONS, "wb")
            )
            pickle.dump(
                self.dict_embeddings,
                open("bin/dict_emb%s.pkl" % self.VEC_DIMENSIONS, "wb")
            )
            pickle.dump(
                self.training_data,
                open("bin/embedded_train%s.pkl" % self.VEC_DIMENSIONS, "wb")
            )
            pickle.dump(
                self.test_data,
                open("bin/embedded_test%s.pkl" % self.VEC_DIMENSIONS, "wb")
            )
            pickle.dump(
                self.training_classes,
                open("bin/classes_train%s.pkl" % self.VEC_DIMENSIONS, "wb")
            )
            pickle.dump(
                self.test_classes,
                open("bin/classes_test%s.pkl" % self.VEC_DIMENSIONS, "wb")
            )
        else:
            dict_words = open(
                "bin/dict_words%s.pkl" % self.VEC_DIMENSIONS, "rb"
            )
            dict_emb = open(
                "bin/dict_emb%s.pkl" % self.VEC_DIMENSIONS, "rb"
            )
            etrain = open(
                "bin/embedded_train%s.pkl" % self.VEC_DIMENSIONS, "rb"
            )
            etest = open(
                "bin/embedded_test%s.pkl" % self.VEC_DIMENSIONS, "rb"
            )
            ctrain = open(
                "bin/classes_train%s.pkl" % self.VEC_DIMENSIONS, "rb"
            )
            ctest = open(
                "bin/classes_test%s.pkl" % self.VEC_DIMENSIONS, "rb"
            )
            self.unpadded_train_data = pickle.load(etrain)
            self.unpadded_test_data = pickle.load(etest)
            self.maxlen = max(
                len(max(self.unpadded_train_data, key=len)),
                len(max(self.unpadded_test_data, key=len))
            )
            self.dict_words = pickle.load(dict_words)
            self.dict_embeddings = pickle.load(dict_emb)
            self.training_data = self.pad_dataset(
                self.unpadded_train_data
            )
            self.test_data = self.pad_dataset(
                self.unpadded_test_data
            )
            self.training_classes = pickle.load(ctrain)
            self.test_classes = pickle.load(ctest)
