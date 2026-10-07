import torch
from torch import nn


class LSTM(nn.Module):
    def __init__(self,vocab_size ,num_hiddens = 256,num_layers =1):
        super().__init__()
        self.vocab_size = vocab_size
        self.num_hidden = num_hiddens
        self.num_layers = num_layers

        self.embedding = nn.Embedding(vocab_size, num_hiddens)

        self.lstm = nn.LSTM(
            input_size = num_hiddens,
            hidden_size = num_hiddens,
            num_layers = num_layers
        )

        self.fc = nn.Linear(num_hiddens, vocab_size)

    def forward(self,x,state):
        # state 现在是一个元组 (H, C)

        x=x.permute(1,0)
        x = self.embedding(x)

        # H: (num_layers, batch, num_hiddens)   ← 最后一个时间步的隐藏状态
        # C: (num_layers, batch, num_hiddens)   ← 最后一个时间步的记忆单元

        output, (H,C) = self.lstm(x, state)
        output = output.reshape(-1 , output.shape[-1])
        output = self.fc(output)

        return output , (H,C)

    def begin_state(self,batch_size,device):
        # LSTM 需要两个全零矩阵：一个给 H（隐藏状态），一个给 C（记忆单元）
        # 形状都是 (num_layers, batch_size, num_hiddens)
        H = torch.zeros((self.num_layers, batch_size, self.num_hidden),device=device)
        C = torch.zeros((self.num_layers, batch_size, self.num_hidden),device=device)
        return H,C




