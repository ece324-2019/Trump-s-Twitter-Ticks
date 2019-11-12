import pandas as pd
import numpy as np
import preprocessor as p
import string
from gensim.models import KeyedVectors
import re
import torch

# Function to clean and process trump_tweets_json
def generateTweetTensor(tweets):
    tweet_text = tweets['text'].values

    # Load pre-trained model for words
    model = KeyedVectors.load_word2vec_format('glove.twitter.27B.100d.w2vformat.txt')

    # Process and clean tweets
    clean_tweets = tweet_text.copy()

    for t in range(0, len(clean_tweets)):

        clean_tweets[t] = clean_tweets[t].lower() # convert text to lower-case
        clean_tweets[t] = re.sub('((www\.[^\s]+)|(https?://[^\s]+))', '', clean_tweets[t]) # remove URLs
        clean_tweets[t] = re.sub('@[^\s]+', '', clean_tweets[t]) # remove usernames
        clean_tweets[t] = re.sub(r'#([^\s]+)', r'\1', clean_tweets[t]) # remove the # in #hashtag
        clean_tweets[t] = p.clean(clean_tweets[t])
        clean_tweets[t] = clean_tweets[t].encode('ascii', 'ignore')
        clean_tweets[t] = clean_tweets[t].decode('utf-8')
        tw = clean_tweets[t].split()
        table = str.maketrans('', '', string.punctuation)
        s = [w.translate(table) for w in tw]
        clean_tweets[t] = s

    lengths = []
    word_vector = []
    bad_words = []
    good_count = 0
    bad_count = 0
    max_tweet_len = 0
    for t in c:
        tweet_vector = []
        for w in t:
            try:
                r = model[w]
                tweet_vector += [r]
                good_count += 1
            except:
                bad_count += 1
                bad_words += [w]
        tweet_length = len(tweet_vector)
        lengths += [tweet_length]
        if tweet_length > max_tweet_len:
            max_tweet_len = tweet_length
        word_vector += [tweet_vector]
    padded_vector = []
    for t in range(0, len(word_vector)):
        padded_array = np.zeros((max_tweet_len, 100))
        for i in range(0, len(word_vector[t])):
            padded_array[i] += word_vector[t][i]

    padded_vector += [padded_array]
    padded_vector_export = np.array(padded_vector)
    tweet_tensor = torch.tensor(padded_vector_export)
    return tweet_tensor, lengths
