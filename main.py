# Imports
from tweet_processor import generateTweetTensor
import torch
from sklearn.model_selection import train_test_split


# Load tweets from json
tweets = pd.read_json(r'trump_tweets_json.json')
tweets = tweets[['created_at', 'text']]

# Add labels of Dow Jones
tweets_with_labels = genLabels(tweets)

# Split sets into train, test, and validation
rest_x, test_x, rest_y, test_y = train_test_split(tweets, tweets_with_labels['labels'], test_size=0.2, random_state=37)
train_x, validate_x, train_y, validate_y = train_test_split(rest_x, rest_y, test_size=0.2, random_state=37)

# Generate vector of tweets
train_tweet_vector = generateTweetTensor(train_x)
test_tweet_vector = generateTweetTensor(test_x)
validate_tweet_vector = generateTweetTensor(validate_x)

# Generate labels
