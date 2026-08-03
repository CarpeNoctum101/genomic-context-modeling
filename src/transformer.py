import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=2000):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

class TransformerClassifier(nn.Module):
    def __init__(
            self,
            num_layers=2,
            vocab_size=5,
            d_model=128,
            proj_dim=128,
            nhead=8,
            dim_feedforward=256,
            dropout=0.1,
    ):
        super(TransformerClassifier, self).__init__()

        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        
        self.pos_encoder = PositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward, 
            dropout=dropout, 
            batch_first=True)
        
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        self.proj = nn.Linear(d_model, proj_dim)

        self.classifier = nn.Sequential(
            nn.Linear(proj_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
    
    def forward(self, x):
        mask = (x == 0)

        x = self.embedding(x)

        x = x * math.sqrt(x.size(-1))

        x = self.pos_encoder(x)

        B = x.size(0)

        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls,x], dim=1)

        cls_mask = torch.zeros(B, 1, device=x.device, dtype=torch.bool)
        mask = torch.cat([cls_mask, mask], dim=1)

        x = self.transformer(x, src_key_padding_mask=mask)

        x = x[:, 0]    

        x = self.proj(x)
        logits = self.classifier(x)
        return logits.squeeze(-1)
