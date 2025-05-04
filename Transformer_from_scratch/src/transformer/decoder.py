import torch
import torch.nn as nn

from attention import MultiHeadAttention
from feed_forward import PositionwiseFeedForward



class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        """
        Initialize a single decoder layer.
        
        Args:
            d_model: Dimension of the model
            num_heads: Number of attention heads
            d_ff: Dimension of the feed-forward network's hidden layer
            dropout: Dropout probability
        """
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.enc_dec_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.layer_norm3 = nn.LayerNorm(d_model)

    def forward(self, x, enc_output, src_mask=None, tgt_mask=None):
        """
        Process input through decoder layer.
        
        Args:
            x: Target tensor of shape [batch_size, seq_len, d_model]
            enc_output: Encoder output of shape [batch_size, seq_len, d_model]
            src_mask: Optional mask for encoder attention
            tgt_mask: Optional mask for decoder attention
        Returns:
            Tensor of same shape after processing through decoder layer
        """
        # Self-attention on target sequence (with causal mask)
        tgt_attn_output = self.self_attn(x, x, x, tgt_mask)
        x = self.layer_norm1(x + tgt_attn_output)
        
        # Encoder-decoder attention (target queries, encoder keys/values)
        enc_dec_attn_output = self.enc_dec_attn(x, enc_output, enc_output, src_mask)
        x = self.layer_norm2(x + enc_dec_attn_output)
        
        # Feed-forward
        ffn_output = self.ffn(x)
        x = self.layer_norm3(x + ffn_output)
        
        return x
        

class Decoder(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, d_ff, dropout=0.1):
        """
        Initialize decoder stack.
        
        Args:
            num_layers: Number of decoder layers
            d_model: Dimension of the model
            num_heads: Number of attention heads
            d_ff: Dimension of the feed-forward network's hidden layer
            dropout: Dropout probability
        """
        super().__init__()
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout) 
            for _ in range(num_layers)
        ])

    def forward(self, x, enc_output, src_mask=None, tgt_mask=None):
        """
        Process input through decoder stack.
        
        Args:
            x: Target tensor of shape [batch_size, seq_len, d_model]
            enc_output: Encoder output of shape [batch_size, seq_len, d_model]
            src_mask: Optional mask for encoder attention
            tgt_mask: Optional mask for decoder attention
        Returns:
            Tensor of same shape after processing through decoder stack
        """
        for layer in self.layers:
            x = layer(x, enc_output, src_mask, tgt_mask)
        
        return x