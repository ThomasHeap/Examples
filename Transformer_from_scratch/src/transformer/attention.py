import torch
import torch.nn as nn

from torch.

class ScaledDotProductAttention(nn.Module):
    def __init__(self, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        d_k = k.shape[1]
        attn_scores =  q @ k.transpose(1,2) / torch.sqrt(d_k)
        attn_weights = nn.functional.softmax(attn_scores)
        out = attn_weights @ v
        return out
        

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        # TODO: Initialize multi-head attention
        pass

    def forward(self, q, k, v, mask=None):
        # TODO: Implement multi-head attention
        pass 