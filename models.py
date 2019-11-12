import torch
import torch.nn as nn
import torch.nn.functional as F

class RNN(nn.Module):
    def __init__(self, embedding_dim, hidden_dim):
        super(RNN, self).__init__()
        self.gru = nn.GRU(embedding_dim, hidden_dim)
        self.hidden = torch.zeros(1, 1, hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, 1)


    def forward(self, x, lengths=None):
        x = torch.nn.utils.rnn.pack_padded_sequence(x, lengths= lengths, enforce_sorted = False) # unpad
        a, x = self.gru(x)
        final = self.fc1(x)
        final = torch.sigmoid(final.flatten())
        return final
