import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import math


class MLP(nn.Module):
    def __init__(self, input_size, h1, h2, num_classes_1, num_classes_2, num_classes_3):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, h1)
        self.relu = nn.ReLU()
        self.fc2= nn.Linear(h1, h2)
        self.output1 = nn.Linear(h2, num_classes_1)
        self.output2 = nn.Linear(h2, num_classes_2)
        self.output3 = nn.Linear(h2, num_classes_3)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x1 = self.output1(x)
        x2 = self.output2(x)
        x3 = self.output3(x)
        return x1, x2, x3

class CNN(nn.Module):
    def __init__(self, input_channel, input_size, h1, h2, h3, num_classes_1, num_classes_2, num_classes_3, kernel_size):
        super(CNN, self).__init__()
        self.conv1d = nn.Conv1d(input_channel, h1, kernel_size=kernel_size)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(h1 * (input_size-kernel_size+1), h2)
        self.fc2 = nn.Linear(h2, h3)
        self.output1 = nn.Linear(h3, num_classes_1)
        self.output2 = nn.Linear(h3, num_classes_2)
        self.output3 = nn.Linear(h3, num_classes_3)

    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        elif x.dim() == 2:
            x = x.unsqueeze(1)

        x = self.conv1d(x)
        x = self.relu(x)
        if x.dim() == 3:
            x = x.view(x.size(0), -1)
        elif x.dim() == 2:
            x = x.reshape(1, -1)
            x = x.squeeze(0)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x1 = self.output1(x)
        x2 = self.output2(x)
        x3 = self.output3(x)
        return x1, x2, x3

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes_1, num_classes_2, num_classes_3):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.rnn = nn.RNN(input_size[1], hidden_size, num_layers, batch_first=True)
        self.fc1 = nn.Linear(hidden_size*input_size[0], num_classes_1)
        self.fc2 = nn.Linear(hidden_size*input_size[0], num_classes_2)
        self.fc3 = nn.Linear(hidden_size*input_size[0], num_classes_3)

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(0)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.rnn(x, h0)
        
        out = out.reshape((out.shape[0], out.shape[1] * out.shape[2]))
        if out.shape[0] == 1:
            out = out.squeeze(0)
        x1 = self.fc1(out)
        x2 = self.fc2(out)
        x3 = self.fc3(out)

        return x1, x2, x3

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes_1, num_classes_2, num_classes_3):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size[1], hidden_size, num_layers, batch_first=True)
        self.fc1 = nn.Linear(hidden_size * input_size[0], num_classes_1)
        self.fc2 = nn.Linear(hidden_size * input_size[0], num_classes_2)
        self.fc3 = nn.Linear(hidden_size * input_size[0], num_classes_3)

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(0)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        out, _ = self.lstm(x, (h0, c0))

        out = out.reshape((out.shape[0], out.shape[1] * out.shape[2]))
        if out.shape[0] == 1:
            out = out.squeeze(0)
        x1 = self.fc1(out)
        x2 = self.fc2(out)
        x3 = self.fc3(out)

        return x1, x2, x3
    

class Transformer(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes_1, num_classes_2, num_classes_3, max_length):
        super(Transformer, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embedding = nn.Linear(input_size[1], hidden_size)
        self.pos_encoder = PositionalEncoding(hidden_size, max_len=max_length)
        encoder_layers = TransformerEncoderLayer(hidden_size, nhead=8, batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers)
        self.dropout = nn.Dropout(p=0.2)
        self.fc1 = nn.Linear(hidden_size * input_size[0], num_classes_1)
        self.fc2 = nn.Linear(hidden_size * input_size[0], num_classes_2)
        self.fc3 = nn.Linear(hidden_size * input_size[0], num_classes_3)

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(0)

        x = self.embedding(x)
        x = self.pos_encoder(x)

        out = self.transformer_encoder(x)
        out = self.dropout(out)

        out = out.reshape(out.size(0), -1)
        x1 = self.fc1(out)
        x2 = self.fc2(out)
        x3 = self.fc3(out)

        x1 = x1.squeeze(0)
        x2 = x2.squeeze(0)
        x3 = x3.squeeze(0)
        return x1, x2, x3

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.2, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x[:self.pe.size(0), :] += self.pe[:x.size(0), :]
        return self.dropout(x)