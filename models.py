import torch
import torch.nn as nn
import torch.nn.functional as F

class RNNClassifier(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, num_labels):
        super(RNNClassifier, self).__init__()

        self.num_layers = 1
        self.gru = nn.GRU(input_size=embedding_dim, hidden_size=hidden_dim, num_layers=self.num_layers)
        self.linear = nn.Linear(hidden_dim, num_labels)

    def forward(self, x):
        x = torch.transpose(x, 0, 1)

        x = x.float()
        output, h_n = self.gru(x)
        x = h_n[0].float()
        x = self.linear(x)
        x = F.softmax(x)
        return x