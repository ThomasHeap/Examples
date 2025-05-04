import torch
import torch.nn as nn

from .attention import MultiHeadAttention
from .feed_forward import PositionwiseFeedForward

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        """
        Initialize a single encoder layer.
        
        Args:
            d_model: Dimension of the model
            num_heads: Number of attention heads
            d_ff: Dimension of the feed-forward network's hidden layer
            dropout: Dropout probability
        """
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        
    def forward(self, x, mask=None):
        """
        Process input through encoder layer.
        
        Args:
            x: Tensor of shape [batch_size, seq_len, d_model]
            mask: Optional mask tensor for attention
        Returns:
            Tensor of same shape after processing through encoder layer
        """
        # Self-attention block
        attn_output = self.self_attn(x, x, x, mask)
        x = self.layer_norm1(x + attn_output)
        
        # Feed-forward block
        ffn_output = self.ffn(x)
        x = self.layer_norm2(x + ffn_output)
        
        return x
        

class Encoder(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, d_ff, dropout=0.1):
        """
        Initialize encoder stack.
        
        Args:
            num_layers: Number of encoder layers
            d_model: Dimension of the model
            num_heads: Number of attention heads
            d_ff: Dimension of the feed-forward network's hidden layer
            dropout: Dropout probability
        """
        super().__init__()
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout) 
            for _ in range(num_layers)
        ])

    def forward(self, x, mask=None):
        """
        Process input through encoder stack.
        
        Args:
            x: Tensor of shape [batch_size, seq_len, d_model]
            mask: Optional mask tensor for attention
        Returns:
            Tensor of same shape after processing through encoder stack
        """
        for layer in self.layers:
            x = layer(x, mask)
        
        return x