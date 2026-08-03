import torch
from torch.utils.data import Dataset
from src.tokenizer import tokenize_sequence

class DNADataset(Dataset):
    def __init__(self, df, seq_type="peak_seq"):
        self.df = df.reset_index(drop=True)
        self.seq_type = seq_type
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        seq = row[self.seq_type]
        label = row["label"]

        tokenized_seq = tokenize_sequence(seq)

        return {
            "sequence": torch.tensor(tokenized_seq, dtype=torch.long),
            "label": torch.tensor(label, dtype=torch.float)
        }