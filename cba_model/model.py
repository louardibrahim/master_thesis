import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionWithContext(nn.Module):
    def __init__(self, feature_dim, bias=True):
        super(AttentionWithContext, self).__init__()
        self.feature_dim = feature_dim
        self.bias = bias

        self.W = nn.Parameter(torch.Tensor(feature_dim, feature_dim))
        if self.bias:
            self.b = nn.Parameter(torch.Tensor(feature_dim))
        self.u = nn.Parameter(torch.Tensor(feature_dim))

        self.init_parameters()

    def init_parameters(self):
        nn.init.xavier_uniform_(self.W, gain=nn.init.calculate_gain('tanh'))
        if self.bias:
            nn.init.zeros_(self.b)
        nn.init.uniform_(self.u, a=-0.1527, b=0.1527)

    def forward(self, x):
        uit = torch.matmul(x, self.W)
        if self.bias:
            uit += self.b
        uit = torch.tanh(uit)
        ait = torch.matmul(uit, self.u)
        a = torch.softmax(ait, dim=1).unsqueeze(-1)
        weighted_input = x * a
        return torch.sum(weighted_input, dim=1)


class CNN_BiLSTM_Attention(nn.Module):
    def __init__(self, vocab_size, embedding_dim, output_dim, dropout=0.2):
        super(CNN_BiLSTM_Attention, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.spatial_dropout = nn.Dropout2d(dropout)
        self.conv1 = nn.Conv1d(in_channels=embedding_dim, out_channels=128, kernel_size=5)
        self.maxpool1 = nn.MaxPool1d(kernel_size=3)
        self.conv2 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=5)
        self.maxpool2 = nn.MaxPool1d(kernel_size=3)
        self.conv3 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=5)
        self.maxpool3 = nn.MaxPool1d(kernel_size=3)
        self.lstm = nn.LSTM(128, 128, bidirectional=True, batch_first=True)
        self.attention = AttentionWithContext(256)
        self.fc1 = nn.Linear(256, 96)
        self.fc2 = nn.Linear(96, output_dim)

    def forward(self, x):
        embedded = self.embedding(x).permute(0, 2, 1)
        embedded = self.spatial_dropout(embedded)

        conv1_out = F.relu(self.conv1(embedded))
        conv1_out = self.maxpool1(conv1_out)

        conv2_out = F.relu(self.conv2(conv1_out))
        conv2_out = self.maxpool2(conv2_out)

        conv3_out = F.relu(self.conv3(conv2_out))
        conv3_out = self.maxpool3(conv3_out)
        lstm_out, _ = self.lstm(conv3_out.permute(0, 2, 1))

        attended_out = self.attention(lstm_out)
        fc1_out = F.relu(self.fc1(attended_out))
        return self.fc2(fc1_out)
