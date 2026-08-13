import torch
from torch import nn


class RNN(nn.Module):
    def __init__(self, vocab_size, num_hidden = 256, num_layers =1):
        super().__init__()
        self.vocab_size = vocab_size
        self.num_hidden = num_hidden
        self.num_layers = num_layers

        self.embedding = nn.Embedding(vocab_size,num_hidden)

        self.rnn = nn.GRU(
            input_size=num_hidden,
            hidden_size=num_hidden,
            num_layers=num_layers
        )

        self.fc = nn.Linear(num_hidden, vocab_size)

    def forward(self, X, state):
        X = X.permute(1,0)
        X = self.embedding(X)
        output, state = self.rnn(X, state)
        output = output.reshape(-1, output.shape[-1]) # (steps*batch, num_hiddens)
        output = self.fc(output)
        return output, state

    def begin_state(self, batch_size, device):
        return torch.zeros(self.num_layers, batch_size, self.num_hidden,device=device)


