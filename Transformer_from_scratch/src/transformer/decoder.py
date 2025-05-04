import torch
import torch.nn as nn

class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        # TODO: Initialize decoder layer
        pass

    def forward(self, x, enc_output, src_mask=None, tgt_mask=None):
        # TODO: Implement decoder layer
        pass

class Decoder(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        # TODO: Initialize decoder stack
        pass

    def forward(self, x, enc_output, src_mask=None, tgt_mask=None):
        # TODO: Implement decoder stack
        pass 