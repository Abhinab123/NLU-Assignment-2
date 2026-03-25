import torch
import torch.nn as nn
import torch.nn.functional as F

class VanillaRNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, layers=2, dropout=0.2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.RNN(embed_dim, hidden_dim, layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden=None):
        x = self.embedding(x)
        out, hidden = self.rnn(x, hidden)
        return self.fc(out), hidden

class BidirectionalLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, layers=2, dropout=0.2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, layers, batch_first=True, 
                            bidirectional=True, dropout=dropout)
        # Multiply hidden_dim by 2 because it is bidirectional
        self.fc = nn.Linear(hidden_dim * 2, vocab_size)

    def forward(self, x, hidden=None):
        x = self.embedding(x)
        out, hidden = self.lstm(x, hidden)
        return self.fc(out), hidden

class AttentionRNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, layers=2, dropout=0.2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.RNN(embed_dim, hidden_dim, layers, batch_first=True, dropout=dropout)
        
        # Simple Dot-Product Attention layers
        self.attn_weights = nn.Linear(hidden_dim, hidden_dim)
        self.v = nn.Parameter(torch.rand(hidden_dim))
        self.fc = nn.Linear(hidden_dim * 2, vocab_size)

    def forward(self, x, hidden=None):
        rnn_out, hidden = self.rnn(self.embedding(x), hidden)
        # Compute Attention
        energy = torch.tanh(self.attn_weights(rnn_out))
        weights = F.softmax(torch.matmul(energy, self.v), dim=1)
        context = torch.bmm(weights.unsqueeze(1), rnn_out)
        
        # Concatenate context with original RNN output
        combined = torch.cat((rnn_out, context.expand(-1, rnn_out.size(1), -1)), dim=2)
        return self.fc(combined), hidden