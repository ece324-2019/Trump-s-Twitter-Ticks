# Imports
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
import pandas as pd
import numpy as np
from torchtext import data


def calculateAcc(a, p):

    b = len(a)
    correct = 0
    for i in range(0, b):
        if a[i] < 0.5  and p[i] < 0.5:
            correct += 1
        elif a[i] > 0.5  and p[i] > 0.5:
            correct += 1

    acc = correct / b
    return acc

#Load Pre-trained model for Words
print("Loading Pre-trained model of many tweets...")
glove = KeyedVectors.load_word2vec_format('glove.twitter.27B.100d.w2vformat.txt')
print("Model loaded.")

# Load tweets from json

model = models.RNN(100, 100)
print(model.parameters())
optimizer = torch.optim.Adam(model.parameters(), lr= 0.01)
optimizer.zero_grad()

print("Loading dataset...")
tweets_with_labels = pd.read_csv(r'labeledSNP.csv')
print("Loaded dataset.")


# Split sets into train, test, and validation
print("Splitting data...")
rest_x, test_x, rest_y, test_y = train_test_split(tweets_with_labels['onehot'], tweets_with_labels['onehot'], test_size=0.2, random_state=37)
train_x, validate_x, train_y, validate_y = train_test_split(rest_x, rest_y, test_size=0.2, random_state=37)

print("Generating vectors...")
# Generate vector of tweets
loss_fnc = nn.BCEWithLogitsLoss()

print("Generating Test Vector...")
test_tweet_vector, test_lengths, test_nulls = generateTweetTensor(glove, test_x)
test_y_tensor = torch.from_numpy(test_y.to_numpy())
for n in range(0, len(test_nulls)):
    i  = test_nulls[n] - n
    test_y_tensor = torch.cat([test_y_tensor[0: i], test_y_tensor[i+1:]])

test_dataset = torch.utils.data.TensorDataset(test_tweet_vector, test_y_tensor, torch.tensor(test_lengths))
test_iter = DataLoader(test_dataset, batch_size=64, shuffle=True, num_workers=0)


print("Generating Validate Vector...")
validate_tweet_vector, validate_lengths, validate_nulls = generateTweetTensor(glove, validate_x)
validate_y_tensor = torch.from_numpy(validate_y.to_numpy())
for n in range(0, len(validate_nulls)):
    i  = validate_nulls[n] - n
    validate_y_tensor = torch.cat([validate_y_tensor[0: i], validate_y_tensor[i+1:]])
val_dataset = torch.utils.data.TensorDataset(validate_tweet_vector, validate_y_tensor, torch.tensor(validate_lengths))
#val_iter = data.BucketIterator(val_dataset, batch_size=64, repeat=False, sort_key=lambda x: len(x.text))
val_iter = DataLoader(val_dataset, batch_size=64, shuffle=True, num_workers=0)


print("Generating Train Vector...")
train_tweet_vector, train_lengths, train_nulls = generateTweetTensor(glove, train_x)
train_y_tensor = torch.from_numpy(train_y.to_numpy())
for n in range(0, len(train_nulls)):
    i  = train_nulls[n] - n
    train_y_tensor = torch.cat([train_y_tensor[0: i], train_y_tensor[i+1:]])
train_dataset = torch.utils.data.TensorDataset(train_tweet_vector, train_y_tensor, torch.tensor(train_lengths) )
train_iter = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0)


print("Generated buckets.")

# Create rnn
#model = models.RNN(100, 100)
print(model.parameters())
learning_rate = 0.01
num_epochs = 25

# Train models
loss_fnc = nn.BCEWithLogitsLoss()
#optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
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
        inputs = batch[0]
        actual = batch[1]
        lengths_float = batch[2]
        lengths = lengths_float.long()
        print("success")
        optimizer.zero_grad()
        predicted = model(inputs,lengths)
        loss = loss_fnc(predicted, actual.float())
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
        t_predicted = model(t_inputs, t_lengths)
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
        v_inputs = v_batch[0]
        v_actual = v_batch[1]
        v_lengths_floats = v_batch[2]
        v_lengths = v_lengths_floats.long()
        valid_labels += v_actual.tolist()
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
    t_inputs = t_batch[0]
    t_actual = t_batch[1]
    t_lengths_floats = t_batch[2]
    t_lengths = t_lengths_floats.long()
    test_labels += t_actual.tolist()
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
