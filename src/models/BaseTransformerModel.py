import torch
import torch.nn as nn
import math

class BaseTransformerModel(nn.Module):
    def __init__(self, max_int: int = 50, multiplier: int = 2):
        """Simple Transformer model for arithmetic sequence prediction
        Automatically sizes network based on information content
        
        Args:
            max_int: Maximum integer value in sequences
                    Vocab will be sized to handle sums up to 2*max_int
        """
        super().__init__()

        # multiply the size of network for scaling experiment.
        self.multiplier = multiplier
        
        # Vocab size needs to handle: numbers up to 2*max_int + operators
        self.vocab_size = (2 * max_int) + 3  # +3 for 0 and two operators
        
        # Calculate dimensions based on information content
        info_per_token = math.ceil(math.log2(self.vocab_size))
        # Power of 2, 4x info content
        self.d_model = self.multiplier ** math.ceil(math.log2(info_per_token * 4))  # Power of 2, 4x info content
        
        # Transformer components
        self.embedding = nn.Embedding(self.vocab_size, self.d_model)
        self.pos_encoding = PositionalEncoding(self.d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=4,  # Use 4 attention heads
            dim_feedforward=self.d_model * 2,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        self.output_layer = nn.Linear(self.d_model, self.vocab_size)
        
    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Forward pass
        
        Args:
            tokens: Input sequence tokens [batch_size, seq_len]
            
        Returns:
            Logits for next token prediction [batch_size, vocab_size]
        """
        
        # Embed and add positional encoding
        x = self.embedding(tokens) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        
        # Pass through transformer
        x = self.transformer(x)
        
        # Use last token's representation for prediction
        x = x[:, -1]  # [batch_size, d_model]
        
        # Project to vocabulary
        logits = self.output_layer(x)  # [batch_size, vocab_size]
        return logits
    
    def predict(self, tokens: torch.Tensor) -> torch.Tensor:
        """Generate predictions for input sequence
        
        Args:
            tokens: Input sequence tokens [batch_size, seq_len]
            
        Returns:
            Predicted next tokens [batch_size]
        """
        with torch.no_grad():
            return self.forward(tokens).argmax(dim=-1)

    def save(self, path: str):
        """Save model to path"""
        torch.save(self.state_dict(), path)
    
    def load(self, path: str):
        """Load model from path"""
        self.load_state_dict(torch.load(path))

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5):
        """Simple positional encoding using sin/cos functions
        
        Args:
            d_model: Embedding dimension
            max_len: Maximum sequence length
        """
        super().__init__()
        
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input"""
        return x + self.pe[:, :x.size(1)]
