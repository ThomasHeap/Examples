import torch
import torch.nn as nn

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        # TODO: Initialize encoder layer
        pass

    def forward(self, x, mask=None):
        # TODO: Implement encoder layer
        pass

class Encoder(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        # TODO: Initialize encoder stack
        pass

    def forward(self, x, mask=None):
        # TODO: Implement encoder stack
        pass 