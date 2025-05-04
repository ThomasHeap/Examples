import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from src.transformer import Transformer
from src.dataset import ShakespeareDataset
from src.utils.helpers import get_device, create_padding_mask, create_look_ahead_mask

def train_epoch(model, train_loader, optimizer, criterion, device, pad_idx):
    """
    Train for one epoch.
    """
    model.train()
    total_loss = 0
    
    pbar = tqdm(train_loader, desc="Training")
    for batch_idx, (src, tgt) in enumerate(pbar):
        src, tgt = src.to(device), tgt.to(device)
        
        # Debug input tensors
        if torch.isnan(src).any() or torch.isnan(tgt).any():
            print(f"NaN detected in input tensors at batch {batch_idx}")
            print(f"src has NaN: {torch.isnan(src).any()}")
            print(f"tgt has NaN: {torch.isnan(tgt).any()}")
            continue
        
        # Create masks
        src_mask = create_padding_mask(src, pad_idx)
        tgt_mask = create_padding_mask(tgt, pad_idx) & create_look_ahead_mask(tgt.shape[1])
        
        # Debug masks
        if torch.isnan(src_mask).any() or torch.isnan(tgt_mask).any():
            print(f"NaN detected in masks at batch {batch_idx}")
            continue
        
        # Forward pass
        output = model(src, tgt[:, :-1], src_mask, tgt_mask[:, :, :-1, :-1])
        
        # Debug model output
        if torch.isnan(output).any():
            print(f"NaN detected in model output at batch {batch_idx}")
            print(f"Output shape: {output.shape}")
            print(f"Output min: {output.min()}, max: {output.max()}")
            continue
        
        # Calculate loss
        loss = criterion(output.reshape(-1, output.shape[-1]), tgt[:, 1:].reshape(-1))
        
        # Debug loss
        if torch.isnan(loss):
            print(f"NaN detected in loss at batch {batch_idx}")
            print(f"Output values: {output.reshape(-1, output.shape[-1])[:5]}")
            print(f"Target values: {tgt[:, 1:].reshape(-1)[:5]}")
            continue
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Debug gradients
        for name, param in model.named_parameters():
            if param.grad is not None and torch.isnan(param.grad).any():
                print(f"NaN detected in gradients for {name} at batch {batch_idx}")
                continue
        
        optimizer.step()
        
        total_loss += loss.item()
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    return total_loss / len(train_loader)

def main():
    # Hyperparameters
    d_model = 256
    num_heads = 4
    num_layers = 3
    d_ff = 1024
    dropout = 0.1
    batch_size = 128
    num_epochs = 10
    learning_rate = 0.0001
    seq_len = 50
    stride = 1
    
    # Set device
    device = get_device()
    
    # Load dataset
    train_dataset = ShakespeareDataset(seq_len=seq_len, stride=stride)
    pad_idx = train_dataset.vocab['<pad>']
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )
    
    # Initialize model
    model = Transformer(
        src_vocab_size=len(train_dataset.vocab),
        tgt_vocab_size=len(train_dataset.vocab),
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        d_ff=d_ff,
        dropout=dropout
    ).to(device)
    
    # Initialize optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
    
    # Training loop
    for epoch in range(num_epochs):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, pad_idx)
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {train_loss:.4f}')
    
    # Save model
    torch.save(model.state_dict(), 'transformer_model.pth')

if __name__ == "__main__":
    main() 