import math
import torch
from torch import nn

# TODO:
# POSTITIONAL EMBEDDING
#   - [ ] Understand abstractly on how RoPe works
#   - [X] Impletment a simple Absolute Encoding using SinSudal
#   - [X] Initialize all the maps.
#   - [X] Figure out how I want to structure the class, shall we keep as a simple func or make it into a layer

# - [ ] Not Tested :)

class CompleteEmbedding(nn.Module):
    """
    Given a sequence of tokens, returns a token + position embedded matrix
    """
    
    def __init__(self, embedding_dim, vocab_size, context_len):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.tok_embedding = nn.Embedding(embedding_dim=self.embedding_dim, num_embeddings=vocab_size) # Creates a token embedding of total_tokens X dim_size
        self.pos_embedding = SinCosPositionEmbedding(self.embedding_dim, context_len)
    
    def forward(self, X):
        """
        Applies Token Embedding and Positional Embedding to Tokens
        
        Keyword arguments:
        X -> (batch_size, total_tokens)
        
        Returns
        (batch_size, token_tokens, embedding_dim)
        """
        tok_emb = self.tok_embedding(X)
        
        # Apply scaling factor from the paper
        scaled_tok_emb = tok_emb * math.sqrt(self.embedding_dim)
        
        # Add positional encodings
        return self.pos_embedding(scaled_tok_emb)
        
class SinCosPositionEmbedding(nn.Module):
    def __init__(self, embedding_dim:int, max_num_tokens:int, n=1000):
        super().__init__()
        
        # Initialize empty tensors of type float - [len , dims]
        pos_embedding = torch.zeros(max_num_tokens, embedding_dim, dtype=torch.float)
        
        # Positions
        # Adding extra "col" so that no need to index during `Populate`
        k = torch.arange(0, max_num_tokens).unsqueeze(1)
        
        # Calculate the denominators
        # 2i    --> torch.arange(0, dims, 2)
        # - n/d --> - torch.log(1000)/dims
        denom = torch.exp(torch.arange(0, embedding_dim, 2) * (- torch.log(torch.tensor(1000))/embedding_dim))
        
        # Populate
        pos_embedding[:, ::2] = torch.sin(k * denom)
        pos_embedding[:, 1::2] = torch.cos(k * denom)
        
        # Move the pos_embedding to GPU
        # Also change the Postional Embedding shape 
        #   from (embedding_dim, max_num_tokens) -> (1, embedding_dim, max_num_tokens)
        #   Easier during the add, as batchs will be involved
        self.register_buffer('pos_embedding', pos_embedding.unsqueeze(0))
                
    def get_embedding_matrix(self) -> torch.Tensor:
        return self.pos_embedding
    
    def forward(self, X):
        """
        Adds Postional Information to Input
                
        Keyword arguments:
        Shape X: (batch_size, total_tokens, embedding_dim)
        """
        # X -> (batch_size, total_tokens, embedding_dim)
        
        # self.pos_embedding is [1, max_seq_len, embedding_dim]
        # We slice up to the sequence length of the input X (X.size(1))
        # The result is [1, seq_len, embedding_dim]
        return X + self.pos_embedding[:, :X.size(1), :]
        


if __name__ == "__main__":
    D_MODEL = 128  # Dimension of the model
    MAX_LEN = 100  # Max sequence length to visualize
    
    posEmb = SinCosPositionEmbedding(embedding_dim=D_MODEL, max_num_tokens=MAX_LEN)
    pe_matrix = posEmb.get_embedding_matrix()
    pe_numpy = pe_matrix.squeeze().numpy() # Convert to NumPy for Matplotlib

    # --- Plotting ---
    print(f"Plotting positional encoding for d_model={D_MODEL} and max_len={MAX_LEN}")

    import matplotlib.pyplot as plt
    plt.figure(figsize=(15, 6))
    pos_enc_plot = plt.imshow(
        pe_numpy,
        cmap='viridis',      # Use a perceptually uniform colormap
        aspect='auto',       # Adjust aspect ratio to fit the figure
        interpolation='none' # No interpolation, show raw pixels
    )
    plt.show()