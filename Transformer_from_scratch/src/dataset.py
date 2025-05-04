import torch
from torch.utils.data import Dataset, DataLoader
import os
from collections import Counter

class ShakespeareDataset(Dataset):
    """
    Dataset for character-level Shakespeare text.
    
    Args:
        seq_len: Length of input sequences
        stride: Stride for sliding window
    """
    def __init__(self, seq_len=50, stride=1):
        self.seq_len = seq_len
        self.stride = stride
        
        # Load text from local file
        with open('data/shakespeare/data.txt', 'r') as f:
            self.text = f.read()
        
        # Create vocabulary
        chars = sorted(list(set(self.text)))
        self.vocab = {ch: i for i, ch in enumerate(chars)}
        self.vocab['<pad>'] = len(self.vocab)
        self.vocab['<unk>'] = len(self.vocab)
        self.vocab['<sos>'] = len(self.vocab)
        self.vocab['<eos>'] = len(self.vocab)
        
        # Create reverse vocabulary
        self.idx2char = {i: ch for ch, i in self.vocab.items()}
        
        # Create sequences
        self.sequences = []
        for i in range(0, len(self.text) - seq_len, stride):
            self.sequences.append(self.text[i:i + seq_len])
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        seq = self.sequences[idx]
        # Convert to indices
        indices = [self.vocab[ch] for ch in seq]
        # Input is sequence[:-1], target is sequence[1:]
        return torch.tensor(indices[:-1]), torch.tensor(indices[1:]) 