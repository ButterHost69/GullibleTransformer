import torch
from torch import nn

# TODO:
# POSTITIONAL EMBEDDING
#   - [X] Impletment a simple Absolute Encoding using SinSudal
#   - [ ] Understand abstractly on how RoPe works
#   - [X] Initialize all the maps.
#   - [ ] Figure out how I want to structure the class, shall we keep as a simple func or make it into a layer




class SinCosPositionEmbedding(nn.Module):
    def __init__(self, dims:int, len_n:int, n=1000):
        super().__init__()
        
        # Initialize empty tensors of type float - [len , dims]
        self.pos_embedding = torch.zeros(len_n, dims, dtype=torch.float)
        
        # Positions
        # Adding extra "col" so that no need to index during `Populate`
        k = torch.arange(0, len_n).unsqueeze(1)
        
        # Calculate the denominators
        # 2i    --> torch.arange(0, dims, 2)
        # - n/d --> - torch.log(1000)/dims
        denom = torch.exp(torch.arange(0, dims, 2) * (- torch.log(torch.tensor(1000))/dims))
        
        # Populate
        self.pos_embedding[:, ::2] = torch.sin(k * denom)
        self.pos_embedding[:, 1::2] = torch.cos(k * denom)
        
        print("Embedding:\n", self.pos_embedding)
        print("Pos embedding:\n", self.pos_embedding)
                
    def get_embedding_matrix(self) -> torch.Tensor:
        return self.pos_embedding
    
    def get_embedding(self, pos):
        """
        Returns the Embedding Matrix for that specific index
        
        Keyword arguments:
        pos : k/position of token
        """
        
        return self.pos_embedding[pos,:]


if __name__ == "__main__":
    D_MODEL = 128  # Dimension of the model
    MAX_LEN = 100  # Max sequence length to visualize
    
    posEmb = SinCosPositionEmbedding(dims=D_MODEL, len_n=MAX_LEN)
    pe_matrix = posEmb.get_embedding_matrix()
    pe_numpy = pe_matrix.numpy() # Convert to NumPy for Matplotlib

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