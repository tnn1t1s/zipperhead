import torch
import torch.nn as nn
import math

class BaseTransformerModel(nn.Module):
    def __init__(self, max_int: int = 100, embed_dim: int = 64, device=None):
        """
        A minimal transformer for arithmetic sequence prediction.
        Uses single-head attention and minimal layers.
        
        Args:
            max_int: Maximum integer in sequences
            embed_dim: Embedding dimension (kept fixed for simplicity)
            device: Computation device
        """
        super().__init__()
        self.device = device if device is not None else torch.device('cpu')
        
        # Vocabulary: numbers up to 2*max_int + operators
        self.vocab_size = (2 * max_int) + 3  # +3 for 0 and two operators
        
        # Simple embedding of fixed size
        self.embedding = nn.Embedding(self.vocab_size, embed_dim)
        
        # Single attention head
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=1,
            batch_first=True
        )
        
        # Simple feedforward after attention
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.ReLU(),
            nn.Linear(embed_dim * 2, embed_dim)
        )
        
        # Output projection
        self.output = nn.Linear(embed_dim, self.vocab_size)
        
        # Move to device
        self.to(self.device)
        
    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Forward pass processes sequence:
        1. Embed tokens
        2. Apply self-attention
        3. Apply feedforward
        4. Project to vocabulary
        
        Args:
            tokens: [batch_size, seq_len] input tokens
        Returns:
            [batch_size, vocab_size] logits for next token prediction
        """
        # Move input to device
        tokens = tokens.to(self.device)
        
        # Embed the input tokens
        x = self.embedding(tokens)  # [batch_size, seq_len, embed_dim]
        
        # Self-attention
        # attention expects: [batch_size, seq_len, embed_dim]
        attn_out, _ = self.attention(x, x, x)
        
        # Feedforward on last token's representation
        ff_out = self.ff(attn_out[:, -1])  # Use last token
        
        # Project to vocabulary
        logits = self.output(ff_out)
        
        return logits
    
    def predict(self, tokens: torch.Tensor) -> torch.Tensor:
        """Generate predictions for input sequence"""
        with torch.no_grad():
            return self.forward(tokens).argmax(dim=-1).cpu()

    def save(self, path: str):
        torch.save(self.state_dict(), path)
    
    def load(self, path: str):
        self.load_state_dict(torch.load(path, map_location=self.device))
