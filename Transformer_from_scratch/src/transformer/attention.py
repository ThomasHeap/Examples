import torch
import torch.nn as nn


class ScaledDotProductAttention(nn.Module):
    def __init__(self, dropout=0.1):
        """
        Initialize scaled dot-product attention mechanism.
        
        Args:
            dropout: Dropout probability for attention weights
        """
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        """
        Compute attention weights and apply them to values.
        
        Args:
            q: Query tensor of shape [batch_size, num_heads, seq_len, d_k]
            k: Key tensor of shape [batch_size, num_heads, seq_len, d_k]
            v: Value tensor of shape [batch_size, num_heads, seq_len, d_v]
            mask: Optional mask tensor to prevent attention to certain positions
        Returns:
            Tensor of shape [batch_size, num_heads, seq_len, d_v]
        """
        d_k = k.shape[-1]
        attn_scores = q @ k.transpose(-2, -1) / torch.sqrt(d_k)
        
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))
        
        attn_weights = nn.functional.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        out = attn_weights @ v
        return out
        

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        """
        Initialize multi-head attention mechanism.
        
        Args:
            d_model: Dimension of the model
            num_heads: Number of attention heads
            dropout: Dropout probability
        """
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.w_q = nn.Linear(d_model, d_model)  # Query projection
        self.w_k = nn.Linear(d_model, d_model)  # Key projection
        self.w_v = nn.Linear(d_model, d_model)  # Value projection
        self.w_o = nn.Linear(d_model, d_model)  # Output projection
        
        self.sdpa = ScaledDotProductAttention(dropout)
   

    def forward(self, q, k, v, mask=None):
        """
        Apply multi-head attention to input sequences.
        
        Args:
            q: Query tensor of shape [batch_size, seq_len, d_model]
            k: Key tensor of shape [batch_size, seq_len, d_model]
            v: Value tensor of shape [batch_size, seq_len, d_model]
            mask: Optional mask tensor
        Returns:
            Tensor of shape [batch_size, seq_len, d_model]
        """
        batch_size = q.shape[0]
        
        # Project and reshape for multi-head attention
        q = self.w_q(q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        k = self.w_k(k).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        v = self.w_v(v).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Apply scaled dot-product attention
        attn_output = self.sdpa(q, k, v, mask)
        
        # Reshape and project back to original dimension
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.w_o(attn_output)
        
        return output
        