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
        # TODO: Initialize tokenizer
        pass

    def encode(self, text):
        """
        Convert text to token indices.
        
        Args:
            text: String or list of strings to tokenize
            
        Returns:
            Tensor of shape [batch_size, seq_len] containing token indices
            - Should include special tokens (e.g., [SOS], [EOS])
            - Should handle unknown tokens
            - Should pad sequences to same length
        """
        # TODO: Implement text encoding
        pass

    def decode(self, tokens):
        """
        Convert token indices back to text.
        
        Args:
            tokens: Tensor of shape [batch_size, seq_len] containing token indices
            
        Returns:
            String or list of strings
            - Should remove special tokens
            - Should handle padding tokens
            - Should convert unknown tokens to a placeholder
        """
        # TODO: Implement token decoding
        pass 