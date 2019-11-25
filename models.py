import torch
import torch.nn as nn
import torch.nn.functional as F

class RNN(nn.Module):
    def __init__(self, embedding_dim, hidden_dim):
        super(RNN, self).__init__()
        self.hidden_dim = hidden_dim
        self.gru = nn.GRU(embedding_dim, hidden_dim)
        self.hidden = torch.zeros(1, 1, hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, 3)


    def forward(self, data, lengths=None):
        print("IN MODEL: ")
        print(data.shape)
        length = lengths.cpu()
        data = torch.transpose(data, 0, 1)
        x = torch.nn.utils.rnn.pack_padded_sequence(data, lengths= length, enforce_sorted = False) # unpad
        x_float = x.float()
        a, x_next = self.gru(x_float)
        final = self.fc1(x_next)
        final = F.softmax(final.flatten(), dim = -1)
        return final

class RNN2(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN2, self).__init__()
        self.hidden_size = hidden_size
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

def forward(self, data, length, hidden):
        length = lengths.cpu()
        data = torch.transpose(data, 0, 1)
        x = torch.nn.utils.rnn.pack_padded_sequence(data, lengths= length, enforce_sorted = False) # unpad
        combined = torch.cat((embeds.view(1, -1), hidden), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden
