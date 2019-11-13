# Imports
from tweet_processor import generateTweetTensor
from preprocess import genLabels
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
import models
import pandas as pd
import numpy as np
from torchtext import data


# Load tweets from json
tweets = pd.read_json('trump_tweets_json.json')
tweets = tweets[['created_at', 'text']]

# Add labels of Dow Jones
print("Generating labels...")
tweets_with_labels = genLabels(tweets)
print("Labels generated.")

# Split sets into train, test, and validation
print("Splitting data...")
rest_x, test_x, rest_y, test_y = train_test_split(tweets, tweets_with_labels['labels'], test_size=0.2, random_state=37)
train_x, validate_x, train_y, validate_y = train_test_split(rest_x, rest_y, test_size=0.2, random_state=37)

print("Generating vectors...")
# Generate vector of tweets
print("Generating Train Vector...")
train_tweet_vector, train_lengths = generateTweetTensor(train_x)
train_y_tensor = torch.from_numpy(train_y.to_numpy())
train_dataset = torch.utils.data.TensorDataset(fields= [('text', train_tweet_vector), ('label', train_y_tensor)] )
train_iter = data.BucketIterator(train_dataset, batch_size=64,repeat=False, sort_key=lambda x: len(x.text))

print("Generating Test Vector...")
test_tweet_vector, test_lengths = generateTweetTensor(test_x)
test_y_tensor = torch.from_numpy(test_y.to_numpy())
test_dataset = torch.utils.data.TensorDataset( fields= [('text', test_tweet_vector), ('label', test_y_tensor)] )
test_iter = data.BucketIterator(test_tweet_vector, batch_size=64,repeat=False, sort_key=lambda x: len(x.text))

print("Generating Validate Vector...")
validate_tweet_vector, validate_lengths = generateTweetTensor(validate_x)
validate_y_tensor = torch.from_numpy(validate_y.to_numpy())
val_dataset = torch.utils.data.TensorDataset( fields= [('text', validate_tweet_vector), ('label', validate_y_tensor)] )
val_iter = data.BucketIterator(validate_tweet_vector, batch_size=64, repeat=False, sort_key=lambda x: len(x.text))

print("Generated buckets.")

# Create rnn
model = models.RNN(100, 100)
learning_rate = 0.01
num_epochs = 25

# Train models
loss_fnc = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
eps = []
training_loss = []
training_accs = []
validation_loss = []
validation_accs = []
print("Model Created.")

for e in range(num_epochs):
    eps += [e]
    print("Epoch: " + str(e))
    for j, batch in enumerate(train_iter):
        labels = batch.label
        inputs, lengths = batch.text
        optimizer.zero_grad()
        predicted = model(inputs, lengths)
        actual = labels.float()
        loss = loss_fnc(predicted, actual)
        loss.backward()
        optimizer.step()

    # Calculate Training Accuracy
    train_labels = []
    train_preds = []

    for k, t_batch in enumerate(train_iter):
        t_labels = t_batch.label
        t_inputs, v_lengths = t_batch.text
        train_labels += t_labels.tolist()
        t_predicted = model(t_inputs, v_lengths)
        train_preds += t_predicted.tolist()

    t_acc = calculateAcc(train_labels, train_preds)
    t_loss = loss_fnc(torch.FloatTensor(train_preds), torch.FloatTensor(train_labels))
    training_accs += [t_acc]
    training_loss += [t_loss.item()]
    print("Training Accuracy: " + str(t_acc))
    print("Training Loss: " + str(t_loss.item()))

    # Calculate Validation Accuracy
    valid_labels = []
    valid_preds = []
    for k, v_batch in enumerate(val_iter):
        v_labels = v_batch.label
        v_inputs, v_lengths = v_batch.text
        valid_labels += v_labels.tolist()
        v_predicted = model(v_inputs, v_lengths)
        valid_preds += v_predicted.tolist()

    v_acc = calculateAcc(valid_labels, valid_preds)
    v_loss = loss_fnc(torch.FloatTensor(valid_preds), torch.FloatTensor(valid_labels))
    validation_accs += [v_acc]
    validation_loss += [v_loss.item()]
    print("Validation Accuracy: " + str(v_acc))
    print("Validation Loss: " + str(v_loss.item()))

# Calculate Testing Accuracy
test_labels = []
test_preds = []
for k, t_batch in enumerate(test_iter):
    t_labels = t_batch.label
    t_inputs, t_lengths = t_batch.text
    test_labels += t_labels.tolist()
    t_predicted = model(t_inputs, t_lengths)
    test_preds += t_predicted.tolist()

test_loss = loss_fnc(torch.FloatTensor(test_preds), torch.FloatTensor(test_labels))
print("TESTING LOSS: " + str(test_loss.item()))
t_acc = calculateAcc(test_labels, test_preds)
print("TESTING ACCURACY: " + str(t_acc))

#   Display Accuracy vs. Epoch
fig, ax = plt.subplots()
ax.plot(eps, training_accs, label='Training Data')
ax.plot(eps, validation_accs, label='Validation Data')
ax.set(xlabel='Number of Epochs', ylabel='Accuracy', title='Accuracy vs. Epoch')
ax.set_ylim(0, 1)
ax.grid()
ax.legend()
plt.show()

#   Display Loss vs. Epoch
fig, ax = plt.subplots()
ax.plot(eps, training_loss, label='Training Data')
ax.plot(eps, validation_loss, label='Validation Data')
ax.set(xlabel='Number of Epochs', ylabel='Loss', title='Loss vs. Epoch')
ax.set_ylim(0, 1)
ax.grid()
ax.legend()
plt.show()
