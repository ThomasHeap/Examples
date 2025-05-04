import torch

def create_padding_mask(seq):
    """
    Create a mask for padding tokens.
    Args:
        seq: Tensor of shape [batch_size, seq_len]
    Returns:
        mask: Tensor of shape [batch_size, 1, 1, seq_len]
    """
    # Create a mask where padding tokens (0) are True
    mask = (seq == 0).unsqueeze(1).unsqueeze(2)
    return mask

def create_look_ahead_mask(size):
    """
    Create a look ahead mask
    Args:
        size: Int giving size of mask
    Returns
        mask: Tensor of shape [size,size]
    """
    return torch.triu(torch.ones(size,size) * float('-inf'), diagonal=1)

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")