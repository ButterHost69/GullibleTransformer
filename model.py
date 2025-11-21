# TODO:
#   - [X] Implement Self Attention
#   - [X] Implement Multihead Part
#   - [X] Implement the Complete Encoder

from torch import nn
import torch
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    """sumary_line
    
    Keyword arguments:
    Input -> Token Embedding
    """
    
    def __init__(self, embedding_dim, total_num_tokens, num_heads, batch_size, dropout=0.2):
        assert embedding_dim % num_heads == 0, "embedding_dim % num_heads has to be 0"
        
        self.self_attention_dim = embedding_dim // num_heads
        self.embedding_dim = embedding_dim
        self.total_num_tokens = total_num_tokens
        self.num_heads = num_heads
        self.batch_size = batch_size
        
        # X -> tokens_len X embedding_dim
        self.q_nn = nn.Linear(self.embedding_dim , self.embedding_dim)
        self.k_nn = nn.Linear(self.embedding_dim , self.embedding_dim)
        self.v_nn = nn.Linear(self.embedding_dim , self.embedding_dim)
        self.dropout = nn.Dropout(dropout)       
        self.res_dropout = nn.Dropout(dropout)       
    
    def forward(self, X):
        B, T, C = X.size()
        
        # Shape X: (batch_size, total_num_tokens, embedding_dim)
        # Q Shape: (total_num_tokens, embedding_dim)
        # K Shape: (total_num_tokens, embedding_dim)
        # V Shape: (total_num_tokens, embedding_dim)
        q_w = self.q_nn(X)
        k_w = self.k_nn(X)
        v_w = self.v_nn(X)
        
        # Dot Product
        Q = X @ q_w.T # (batch_size, total_tokens, embedding_dim) 
        K = X @ k_w.T # (batch_size, total_tokens, embedding_dim)
        V = X @ v_w.T # (batch_size, total_tokens, embedding_dim)
        
        # Reshape the Q, K, V Matrix to (batch_size, num_heads, total_num_tokens, self_embedding_dim)
        # Q, K, V: (batch_size, total_num_tokens, num_heads, self_embedding_dim)
        Q = Q.reshape(self.batch_size, self.total_num_tokens, self.num_heads, self.self_attention_dim)
        K = K.reshape(self.batch_size, self.total_num_tokens, self.num_heads, self.self_attention_dim)
        V = V.reshape(self.batch_size, self.total_num_tokens, self.num_heads, self.self_attention_dim)
        
        # Regroup Q, K, V: (batch_size, , num_heads, total_num_tokens, self_embedding_dim)
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)
        
        # Perform Self Attention
        attention = F.scaled_dot_product_attention(Q, K, V)
        attention = F.softmax(attention, dim=-1)
        attention = self.dropout(attention)
        
        y = attention @ V
        # Re-assemble the heads side by side and project back to residual stream
        y = y.transpose(1, 2).contiguous().view(B, T, -1)
        y = self.res_dropout(self.c_proj(y))
        return y
        
class MLP(nn.Module):
    
    def __init__(self, embedding_dim:int, bias:bool, dropout=0.2):
        super().__init__()
        self.c_fc    = nn.Linear(embedding_dim, 4 * embedding_dim, bias=bias)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * embedding_dim, embedding_dim, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class LayerNorm(nn.Module):
    """Normalizing Layer in transformer"""

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)
    
class Block(nn.Module):
    
    def __init__(
        self, embedding_dim:int, 
        bias:bool,
        total_num_tokens:int, 
        num_heads:int, 
        batch_size:int,
        dropout=0.2,
    ):
        super().__init__()
        self.ln_1 = LayerNorm(embedding_dim, bias=bias)
        self.attn = MultiHeadAttention(
            embedding_dim=embedding_dim, 
            batch_size=batch_size,
            num_heads=num_heads,
            dropout=dropout,
            total_num_tokens=total_num_tokens,
        )
        self.ln_2 = LayerNorm(embedding_dim, bias=bias)
        self.mlp = MLP()

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x
    
    
class Encoder(nn.Module):
    def __init__(
        self, 
        n_layers:int, 
        block_size:int, 
        vocab_size:int,
        embedding_dim:int,
        bias:bool,
        total_num_tokens:int, 
        num_heads:int, 
        batch_size:int,
        output_size:int,
        dropout=0.2,
    ):
        
        assert vocab_size is not 0 , "Vocab Size cannot be 0"
        assert block_size is not 0 , "Block Size cannot be 0"
        
        self.attention_layers = nn.ModuleList([
            Block(embedding_dim=embedding_dim, 
                bias=bias,
                total_num_tokens=total_num_tokens, 
                num_heads=num_heads, 
                batch_size=batch_size,
                dropout=dropout,
            ) for _ in range(n_layers)
        ])
        
        self.normalizer = LayerNorm(embedding_dim, bias=bias),
        self.final_linear = nn.Linear(embedding_dim, output_size, bias=False)
    
    def forward(self, X, targets):

        for block in self.transformer.h:
            X = block(X)
        X = self.transformer.ln_f(X)
        
        logits = self.lm_head(X)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        return logits, loss