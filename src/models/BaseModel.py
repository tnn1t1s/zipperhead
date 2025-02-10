import torch
import torch.nn as nn
import math

class BaseModel(nn.Module):
    def __init__(self, max_int: int = 100):
        """Simple Seq2Seq model for arithmetic sequence prediction
        Automatically sizes network based on information content
        
        Args:
            max_int: the maximum size of x and y values
        """
        super().__init__()
       
        vocab_size = (2 * max_int) + 3 # +3 for 0, + and = 
        # Calculate minimal dimensions based on information content
        info_per_token = 8 * math.ceil(math.log2(vocab_size))
        embed_dim = info_per_token + 1  # Slight overhead for embedding
        hidden_dim = info_per_token * 2  # Double the info for hidden layer
        
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.seq = nn.Sequential(
            nn.Linear(embed_dim * 4, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, vocab_size)
        )
        
    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(tokens)
        flattened = embedded.reshape(embedded.shape[0], -1)
        return self.seq(flattened)
    
    def predict(self, tokens: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.forward(tokens).argmax(dim=-1)

    def save(self, path: str):
        torch.save(self.state_dict(), path)
    
    def load(self, path: str):
        self.load_state_dict(torch.load(path))
