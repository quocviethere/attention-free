# models.py

import torch
import torch.nn as nn

# 1. RNN Model
class RNNModel(nn.Module):
    """Simple RNN model."""
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim, num_layers=1):
        super(RNNModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.rnn = nn.RNN(embed_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        embedded = self.embedding(x)
        _, hidden = self.rnn(embedded)
        out = self.fc(hidden[-1])
        return out

# 2. LSTM Model
class LSTMModel(nn.Module):
    """LSTM model."""
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim, num_layers=1):
        super(LSTMModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        embedded = self.embedding(x)
        _, (hidden, _) = self.lstm(embedded)
        out = self.fc(hidden[-1])
        return out

# 3. BiLSTM Model
class BiLSTMModel(nn.Module):
    """Bidirectional LSTM model."""
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim, num_layers=1):
        super(BiLSTMModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.bilstm = nn.LSTM(embed_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, x):
        embedded = self.embedding(x)
        _, (hidden, _) = self.bilstm(embedded)
        out = self.fc(torch.cat((hidden[-2], hidden[-1]), dim=1))
        return out

# 4. TextCNN Model
class TextCNN(nn.Module):
    """TextCNN model for text classification."""
    def __init__(self, vocab_size, embed_dim, num_classes, kernel_sizes=[3,4,5], num_channels=100, dropout=0.5):
        super(TextCNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.convs = nn.ModuleList([
            nn.Conv2d(1, num_channels, (k, embed_dim)) for k in kernel_sizes
        ])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(len(kernel_sizes) * num_channels, num_classes)

    def forward(self, x):
        embedded = self.embedding(x)  # [batch_size, seq_len, embed_dim]
        embedded = embedded.unsqueeze(1)  # [batch_size, 1, seq_len, embed_dim]
        conved = [torch.relu(conv(embedded)).squeeze(3) for conv in self.convs]
        pooled = [torch.max(c, dim=2)[0] for c in conved]
        cat = torch.cat(pooled, dim=1)
        dropped = self.dropout(cat)
        out = self.fc(dropped)
        return out

# 5. gMLP Model
class gMLPBlock(nn.Module):
    """Single block of gMLP."""
    def __init__(self, dim, seq_len):
        super(gMLPBlock, self).__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, dim * 4)
        self.gate = nn.Parameter(torch.randn(1, seq_len, dim * 4))
        self.act = nn.GELU()
        self.fc2 = nn.Linear(dim * 4, dim)

    def forward(self, x):
        y = self.norm(x)
        y = self.fc1(y)
        y = y * self.gate
        y = self.act(y)
        y = self.fc2(y)
        return x + y

class gMLPModel(nn.Module):
    """gMLP model for text classification."""
    def __init__(self, vocab_size, embed_dim, num_classes, seq_len, num_blocks=2, dropout=0.5):
        super(gMLPModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.Sequential(
            *[gMLPBlock(embed_dim, seq_len) for _ in range(num_blocks)]
        )
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        x = self.dropout(x)
        x = self.blocks(x)
        x = x.mean(dim=1)
        x = self.fc(x)
        return x

# 6. Mamba Model
try:
    from mamba_ssm import Mamba
except ImportError:
    Mamba = None
    # Optionally raise an error or provide instructions

class Mamba2ForSentimentAnalysis(nn.Module):
    """Mamba model for sentiment analysis."""
    def __init__(self, vocab_size, d_model=128, num_layers=2, num_classes=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.mamba = Mamba(d_model=d_model)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, input_ids):
        x = self.embedding(input_ids)
        x = self.mamba(x)
        x = x.mean(dim=1)
        return self.classifier(x)
