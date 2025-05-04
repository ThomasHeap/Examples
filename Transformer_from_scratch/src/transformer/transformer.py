import torch
import torch.nn as nn

from embeddings import TokenEmbedding, PositionalEncoding
from encoder import Encoder
from decoder import Decoder


class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, dropout=0.1):
        """
        Initialize transformer model.
        
        Args:
            src_vocab_size: Size of source vocabulary
            tgt_vocab_size: Size of target vocabulary
            d_model: Dimension of the model
            num_heads: Number of attention heads
            num_layers: Number of encoder/decoder layers
            d_ff: Dimension of the feed-forward network's hidden layer
            dropout: Dropout probability
        """
        super().__init__()
        self.src_emb = TokenEmbedding(src_vocab_size, d_model)
        self.src_pe = PositionalEncoding(d_model)
        
        self.tgt_emb = TokenEmbedding(tgt_vocab_size, d_model)
        self.tgt_pe = PositionalEncoding(d_model)
        
        self.encoder = Encoder(num_layers, d_model, num_heads, d_ff, dropout)
        self.decoder = Decoder(num_layers, d_model, num_heads, d_ff, dropout)
        
        self.linear = nn.Linear(d_model, tgt_vocab_size)
        
        # Initialize parameters
        self._init_parameters()

    def _init_parameters(self):
        """
        Initialize parameters with Xavier uniform initialization.
        """
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        """
        Process source and target sequences through transformer.
        
        Args:
            src: Source tensor of shape [batch_size, src_seq_len]
            tgt: Target tensor of shape [batch_size, tgt_seq_len]
            src_mask: Optional mask for source sequence
            tgt_mask: Optional mask for target sequence
        Returns:
            Tensor of shape [batch_size, tgt_seq_len, tgt_vocab_size]
        """
        enc_output = self.encode(src, src_mask)
        dec_output = self.decode(tgt, enc_output, src_mask, tgt_mask)
        output = self.linear(dec_output)
        return output

    def encode(self, src, src_mask=None):
        """
        Encode source sequence.
        
        Args:
            src: Source tensor of shape [batch_size, src_seq_len]
            src_mask: Optional mask for source sequence
        Returns:
            Tensor of shape [batch_size, src_seq_len, d_model]
        """
        # Apply source embedding and positional encoding
        src = self.src_emb(src)
        src = self.src_pe(src)
        
        # Pass through encoder
        return self.encoder(src, src_mask)

    def decode(self, tgt, enc_output, src_mask=None, tgt_mask=None):
        """
        Decode target sequence using encoder output.
        
        Args:
            tgt: Target tensor of shape [batch_size, tgt_seq_len]
            enc_output: Encoder output of shape [batch_size, src_seq_len, d_model]
            src_mask: Optional mask for source sequence
            tgt_mask: Optional mask for target sequence
        Returns:
            Tensor of shape [batch_size, tgt_seq_len, d_model]
        """
        # Apply target embedding and positional encoding
        tgt = self.tgt_emb(tgt)
        tgt = self.tgt_pe(tgt)
        
        # Pass through decoder
        return self.decoder(tgt, enc_output, src_mask, tgt_mask)