import torch
import torch.nn as nn

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        """
        Initialize position-wise feed-forward network.
        
        Args:
            d_model: Dimension of the model
            d_ff: Dimension of the feed-forward network's hidden layer
            dropout: Dropout probability
        """
        super().__init__()
        self.fc_1 = nn.Linear(d_model, d_ff)
        self.fc_2 = nn.Linear(d_ff, d_model)
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Apply position-wise feed-forward network to input.
        
        Args:
            x: Tensor of shape [batch_size, seq_len, d_model]
        Returns:
            Tensor of same shape after applying feed-forward network
        """
        x = nn.functional.relu(self.fc_1(x))
        x = self.dropout(x)
        x =  self.fc_2(x)
        x = self.dropout(x)
        return x