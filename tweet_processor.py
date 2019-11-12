import pandas as pd
import numpy as np
import preprocessor as p
import string
from gensim.models import KeyedVectors
import re

# Function to clean and process trump_tweets_json
def cleanTweet(tweets):
    clean_tweets = tweets.copy()
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


    return clean_tweets

# Load in tweets as JSON
tweets = pd.read_json(r'trump_tweets_json.json')
tweets = tweets[['created_at', 'text']]
tweet_text = tweets['text'].values

# Load pre-trained model for words
model = KeyedVectors.load_word2vec_format('glove.twitter.27B.100d.w2vformat.txt')

# Process and clean tweets
clean_tweets = cleanTweet(tweet_text)
tweet_vector =

word_vector = []
good_words = 0
bad_words = 0
for t in c:
    tweet_vector = []
    for w in t:
        try:
            r = model[w]
            tweet_vector += [r]
            good_words += 1
        except:
            bad_words += 1
    word_vector += [tweet_vector]   
