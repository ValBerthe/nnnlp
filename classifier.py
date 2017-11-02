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

    start_time = time.clock()

    # CONSTANTS

    emotions = [
        "ANGER",
        "HAPPINESS",
        "SADNESS",
        "NEUTRAL",
        "HATE",
        "FUN",
        "LOVE"
    ]
    EPOCHS = 3
    VEC_DIMENSIONS = 25
    WINDOW_SIZE = 100
    DICT_PATH = "./Sentiment-Analysis-Dataset/formatted_corpus.txt"
    TRAINING_SET_PATH = "./Training_set.txt"
    CURRENT_DICT_PATH = "./dicts/dict_full_data_40e_20w.d2v"
    WORD_TO_TEST = "good"
    CLASSIFIER_PATH = "classifier.sav"
    GLOVE_PATH = "glove.twitter.27B.%sd.txt" % VEC_DIMENSIONS

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
        text_vectors = open(
            self.GLOVE_PATH, "r", encoding="utf-8"
        ).read().splitlines()
        words = [""] * len(text_vectors)
        embeddings = np.zeros((len(text_vectors), self.VEC_DIMENSIONS))
        regex1 = re.compile(r"^([^ ])+")
        regex2 = re.compile(r"(\-?[0-9]+(\.[0-9]+(e-[0-9]+)?)? ?)+")
        for index, line in tqdm(enumerate(text_vectors)):
            words[index] = regex1.search(line).group(0)
            embeddings[index] = regex2.search(line).group(0).split()
        return words, embeddings

    def build_dataset(self):
        """Build the dataset."""
        formatted_tweets = []
        classes = []
        indexes = []
        print("\nOpening dataset...\n")
        tweets = open(
            self.TRAINING_SET_PATH,
            "r",
            encoding="utf-8"
        ).read().splitlines()
        for index, tweet in tqdm(enumerate(tweets)):
            tweet_data = tweet.split(" ///")
            classes.append(self.emotions.index(tweet_data[0]))
            indexes.append(str(index))
            formatted_tweets.append(self.format_tweet(tweet_data[1]))
        return list(formatted_tweets), len(tweets), classes, indexes

    words, embeddings = build_dictionary()
    print("EMBEDDINGS: %s %s" % (embeddings[:5], words[:100]))
    training_data, n_count, classes, indexes = build_dataset()
    print("TRAININ_DATAG: %s" % (training_data[:20]))
