import torch
import torch.nn as nn
import torch.nn.functional as F

class RNN(nn.Module):
    def __init__(self, embedding_dim, hidden_dim):
        super(RNN, self).__init__()
        #self.gru = nn.GRU(embedding_dim, hidden_dim)
        #self.hidden = torch.zeros(1, 1, hidden_dim)
        #self.fc1 = nn.Linear(hidden_dim, 3)


    def forward(self, data, lengths=None):
        print("IN MODEL: ")
        print(data.shape)
        length = lengths.cpu()
        #data = torch.transpose(data, 0, 1)
        #x = torch.nn.utils.rnn.pack_padded_sequence(data, lengths= length, enforce_sorted = False) # unpad
        #x_float = x.float()
        #a, x_next = self.gru(x_float)
        #final = self.fc1(x_next)
        #final = F.softmax(final.flatten(), dim = -1)
        return final
