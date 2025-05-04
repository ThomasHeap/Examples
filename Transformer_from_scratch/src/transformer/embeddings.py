import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        """
        Initialize positional encoding using sinusoidal functions.
        
        Args:
            d_model: Dimension of the model embeddings
            max_len: Maximum sequence length to support
        """
        super().__init__()
        
        # position encoding matrix
        pe = torch.zeros(max_len, d_model)
        
        # position indices [0, ..., max_len]
        position = torch.arange(0,max_len, dtype=torch.float).unsqueeze(1)
        
        # term that divides position in frequency calculation
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        # sin for even, cos for odd
        pe[:, ::2] = torch.sin(position * div_term)
        pe[:,1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Add positional encoding to input embeddings.
        
        Args:
            x: Tensor of shape [batch_size, seq_len, d_model]
        Returns:
            Tensor of same shape with positional encoding added
        """
        return x + self.pe[:, :x.size(1)]

class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model):
        """
        Initialize token embedding layer.
        
        Args:
            vocab_size: Number of unique tokens in vocabulary
            d_model: Dimension of embedding vectors
        """
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model

    def forward(self, x):
        """
        Convert token indices to embedding vectors.
        
        Args:
            x: Tensor of shape [batch_size, seq_len] containing token indices
        Returns:
            Tensor of shape [batch_size, seq_len, d_model] containing embedding vectors
        """
        # Multiply by sqrt(d_model) to scale the embeddings
        return self.embedding(x) * math.sqrt(self.d_model) 