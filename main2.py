# Imports
import pandas as pd
import numpy as np
import torch.nn.functional as F
from tweet_processor import generateTweetTensor
from preprocess import genLabels
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from gensim.models import KeyedVectors

from torch.utils.data import DataLoader


from sklearn.model_selection import train_test_split
import models

from torchtext import data

def calculateAcc(a, p):
    b = len(a)
    correct = 0
    for i in range(0, b):
        if a[i] == p[i]:
            correct += 1
    acc = correct / b
    return acc
#Load Pre-trained model for Words
print("Loading Pre-trained model of many tweets...")
glove = KeyedVectors.load_word2vec_format('glove.twitter.27B.100d.w2vformat.txt')
print("Model loaded.")

# Load tweets from json
print("Loading dataset...")
#tweets = pd.read_csv('overtrain.csv')

tweets = pd.read_json('trump_tweets_json.json')
tweets = tweets[['created_at', 'text']]
tweets_with_labels = genLabels(tweets)

# Split sets into train, test, and validation
print("Splitting data...")
rest_x, test_x, rest_y, test_y = train_test_split(tweets_with_labels, tweets_with_labels['onehot'], test_size=0.2, random_state=37)
train_x, validate_x, train_y, validate_y = train_test_split(rest_x, rest_y, test_size=0.2, random_state=37)

print("Generating vectors...")
# Generate vector of tweets
loss_fnc = nn.BCEWithLogitsLoss()

print("Generating Test Vector...")
test_tweet_vector, test_lengths, test_nulls = generateTweetTensor(glove, test_x)
test_y_tensor = torch.LongTensor([x for x in np.array(test_y)])
for n in range(0, len(test_nulls)):
    i  = test_nulls[n] - n
    test_y_tensor = torch.cat([test_y_tensor[0: i], test_y_tensor[i+1:]])

test_dataset = torch.utils.data.TensorDataset(test_tweet_vector, test_y_tensor, torch.tensor(test_lengths))
test_iter = DataLoader(test_dataset, batch_size=16, shuffle=True, num_workers=0)



print("Generating Validate Vector...")
validate_tweet_vector, validate_lengths, validate_nulls = generateTweetTensor(glove, validate_x)
validate_y_tensor = torch.LongTensor([x for x in np.array(validate_y)])
for n in range(0, len(validate_nulls)):
    i  = validate_nulls[n] - n
    validate_y_tensor = torch.cat([validate_y_tensor[0: i], validate_y_tensor[i+1:]])
val_dataset = torch.utils.data.TensorDataset(validate_tweet_vector, validate_y_tensor, torch.tensor(validate_lengths))
val_iter = DataLoader(val_dataset, batch_size=16, shuffle=True, num_workers=0)


print("Generating Train Vector...")
train_tweet_vector, train_lengths, train_nulls = generateTweetTensor(glove, train_x)
train_y_tensor = torch.LongTensor([x for x in np.array(train_y)])


for n in range(0, len(train_nulls)):
    i  = train_nulls[n] - n
    train_y_tensor = torch.cat([train_y_tensor[0: i], train_y_tensor[i+1:]])
train_dataset = torch.utils.data.TensorDataset(train_tweet_vector, train_y_tensor, torch.tensor(train_lengths) )
train_iter = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)


print("Generated buckets.")

model = models.RNNClassifier(100, 100, 3)

# Create rnn
learning_rate = 0.015
num_epochs = 300

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train models
loss_fnc = nn.CrossEntropyLoss()

eps = []
training_loss = []
training_accs = []
validation_loss = []
validation_accs = []
print("Model Created.")

for e in range(num_epochs):
    eps += [e]
    labels = []
    optimizer.zero_grad()
    print("Epoch: " + str(e))
    for j, batch in enumerate(train_iter):
        inputs_float = batch[0]

        actual = batch[1]
        lengths_float = batch[2]
        lengths = lengths_float.long()
        inputs = inputs_float.long()
        optimizer.zero_grad()
        predicted = model(inputs)
        labels = []
        for l in actual.tolist():
            labels += [l.index(max(l))]
        label_tensor = torch.Tensor(labels).long()
        loss = loss_fnc(predicted.float(), label_tensor)
        loss.backward()
        optimizer.step()

    print("Calculating Training Accuracy...")
    # Calculate Training Accuracy
    train_labels = []
    train_preds = []

    for k, t_batch in enumerate(train_iter):
        t_inputs = t_batch[0]
        t_actual = t_batch[1]
        t_lengths_floats = t_batch[2]
        t_lengths = t_lengths_floats.long()

        train_labels += t_actual.tolist()
        t_predicted = model(t_inputs)
        train_preds += t_predicted.tolist()

    t_l_best = []
    t_p_best = []
    for l in train_labels:
        t_l_best += [l.index(max(l))]
    for p in train_preds:
        t_p_best += [p.index(max(p))]
    print(len(train_preds))
    print(len(train_lengths))
    t_acc = calculateAcc(t_l_best, t_p_best)
    t_loss = loss_fnc(torch.FloatTensor(train_preds), torch.LongTensor(t_l_best))

    training_accs += [t_acc]
    training_loss += [t_loss.item()]
    print("Training Accuracy: " + str(t_acc))
    print("Training Loss: " + str(t_loss.item()))

    # Calculate Validation Accuracy
    valid_labels = []
    valid_preds = []
    for k, v_batch in enumerate(val_iter):
        v_inputs = v_batch[0]
        v_actual = v_batch[1]
        v_lengths_floats = v_batch[2]
        v_lengths = v_lengths_floats.long()
        valid_labels += v_actual.tolist()
        v_predicted = model(v_inputs)
        valid_preds += v_predicted.tolist()

    v_l_best = []
    v_p_best = []
    for v in valid_labels:
        v_l_best += [v.index(max(v))]
    for p in valid_preds:
        v_p_best += [p.index(max(p))]
    print("Valid Accuracy:")
    v_acc = calculateAcc(v_l_best, v_p_best)
    # convert ohe labels to normal
    labels = []
    for l in valid_labels:
        labels += [l.index(max(l))]
    v_label_tensor = torch.Tensor(labels).long()

    v_loss = loss_fnc(torch.FloatTensor(valid_preds), torch.LongTensor(v_label_tensor))
    validation_accs += [v_acc]
    validation_loss += [v_loss.item()]
    print("Validation Accuracy: " + str(v_acc))
    print("Validation Loss: " + str(v_loss.item()))

# Calculate Testing Accuracy
test_labels = []
test_preds = []
for k, t_batch in enumerate(test_iter):
    t_inputs = t_batch[0]
    t_actual = t_batch[1]
    t_lengths_floats = t_batch[2]
    t_lengths = t_lengths_floats.long()
    test_labels += t_actual.tolist()
    t_predicted = model(t_inputs)
    test_preds += t_predicted.tolist()

labels = []
for l in test_labels:
    labels += [l.index(max(l))]
test_label_tensor = torch.Tensor(labels).long()

test_loss = loss_fnc(torch.FloatTensor(test_preds), torch.LongTensor(test_label_tensor))
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
