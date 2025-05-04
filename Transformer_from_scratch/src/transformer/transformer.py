import torch
import torch.nn as nn

class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, dropout=0.1):
        super().__init__()
        # TODO: Initialize transformer
        pass

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        # TODO: Implement transformer forward pass
        pass

    def encode(self, src, src_mask=None):
        # TODO: Implement encoding
        pass

    def decode(self, tgt, enc_output, src_mask=None, tgt_mask=None):
        # TODO: Implement decoding
        pass 