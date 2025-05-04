import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        # TODO: Initialize positional encoding
        pass

    def forward(self, x):
        # TODO: Implement positional encoding
        pass

class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        # TODO: Initialize token embedding
        pass

    def forward(self, x):
        # TODO: Implement token embedding
        pass 