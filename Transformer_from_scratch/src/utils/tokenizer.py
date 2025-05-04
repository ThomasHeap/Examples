import torch

class SimpleTokenizer:
    def __init__(self):
        """
        Initialize a simple tokenizer for transformer model.
        
        This class should handle:
        - Vocabulary creation from training data
        - Special tokens (e.g., [PAD], [UNK], [SOS], [EOS])
        - Token to index mapping
        - Index to token mapping
        """
        self.vocab = {}
        self.inv_vocab = {}
        self.pad_token = "[PAD]"
        self.unk_token = "[UNK]"
        self.bos_token = "[SOS]"
        self.eos_token = "[EOS]"
        
        # Add special tokens to vocabulary
        for token in [self.pad_token, self.unk_token, self.bos_token, self.eos_token]:
            self.vocab[token] = len(self.vocab)
            self.inv_vocab[len(self.vocab)-1] = token

    def encode(self, text):
        """
        Convert text to token indices.
        
        Args:
            text: List of strings or list of list of strings (batch of sequences)
            
        Returns:
            Tensor of shape [batch_size, seq_len] containing token indices
            - Should include special tokens (e.g., [SOS], [EOS])
            - Should handle unknown tokens
            - Should pad sequences to same length
        """
        # Ensure input is a list of sequences
        if isinstance(text, str):
            text = [text]
        if isinstance(text[0], str):
            text = [text]
            
        sequences = []
        for seq in text:
            tokens = [self.vocab[self.bos_token]]  # Start with [SOS]
            for word in seq:
                # Use [UNK] for unknown words
                token = self.vocab.get(word, self.vocab[self.unk_token])
                tokens.append(token)
            tokens.append(self.vocab[self.eos_token])  # End with [EOS]
            sequences.append(tokens)
        
        # Find max sequence length
        max_length = max(len(seq) for seq in sequences)
        
        # Pad sequences to max length
        padded_sequences = []
        for seq in sequences:
            # Pad with [PAD] token
            padded_seq = seq + [self.vocab[self.pad_token]] * (max_length - len(seq))
            padded_sequences.append(padded_seq)
        
        # Convert to tensor
        return torch.tensor(padded_sequences)

    def decode(self, tokens):
        """
        Convert token indices back to text.
        
        Args:
            tokens: Tensor of shape [batch_size, seq_len] containing token indices
            
        Returns:
            List of strings (one for each sequence in the batch)
            - Should remove special tokens
            - Should handle padding tokens
            - Should convert unknown tokens to a placeholder
        """
        # Convert tensor to list if needed
        if torch.is_tensor(tokens):
            tokens = tokens.tolist()
            
        decoded_sequences = []
        for seq in tokens:
            decoded_tokens = []
            for token_idx in seq:
                # Skip special tokens and padding
                if token_idx in [self.vocab[self.pad_token], 
                               self.vocab[self.bos_token], 
                               self.vocab[self.eos_token]]:
                    continue
                    
                # Convert to word or [UNK]
                word = self.inv_vocab.get(token_idx, self.unk_token)
                decoded_tokens.append(word)
                
            decoded_sequences.append(' '.join(decoded_tokens))
            
        return decoded_sequences 