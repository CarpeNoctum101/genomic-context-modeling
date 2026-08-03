import torch
from torch.nn.utils.rnn import pad_sequence

def collate_fn(batch):
    sequences = [item["sequence"] for item in batch]
    labels = torch.stack([item["label"] for item in batch])

    padded_sequences = pad_sequence(sequences, batch_first=True, padding_value=0)

    lengths = torch.tensor([len(seq) for seq in sequences], dtype=torch.long)

    return {
        "sequences": padded_sequences,
        "labels": labels,
        "lengths": lengths
    }