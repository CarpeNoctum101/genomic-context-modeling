import torch
import torch.nn as nn

class BiLSTMClassifier(nn.Module):
    def __init__(self, embed_dim=128, hidden_dim=128, proj_dim=128, num_layers=2, vocab_size=5):
        super(BiLSTMClassifier, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        self.lstm = nn.LSTM(
            input_size=embed_dim, 
            hidden_size=hidden_dim, 
            num_layers=num_layers,
            batch_first=True, 
            bidirectional=True
        )
        
        self.proj = nn.Linear(hidden_dim * 2, proj_dim)

        self.classifier = nn.Sequential(
            nn.Linear(proj_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        x = self.embedding(x)

        lstm_out, _ = self.lstm(x)

        x = lstm_out.mean(dim=1)

        x = self.proj(x)

        logits = self.classifier(x)

        return logits.squeeze()