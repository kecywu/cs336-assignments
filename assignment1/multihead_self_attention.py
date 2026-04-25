import torch 
import torch.nn as nn 
from cs336_basics.scaled_dot_product_attention import scaled_dot_product_attention
from cs336_basics.linear import Linear
from einops import rearrange

class MultiHeadSelfAttention(nn.Module):

    def __init__(self, d_model, num_heads, rope=None, device=None, dtype=None):

        super().__init__()

        self.d_model = d_model 
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        self.rope = rope 
        self.Q = Linear(d_model, d_model, device=device, dtype=dtype)
        self.K = Linear(d_model, d_model, device=device, dtype=dtype)
        self.V = Linear(d_model, d_model, device=device, dtype=dtype)
        self.O = Linear(d_model, d_model, device=device, dtype=dtype)

    
    def forward(self, x, token_positions=None):
        
        seq_len = x.shape[-2]

        q = self.Q(x)
        k = self.K(x)
        v = self.V(x)

        q = rearrange(q, "... seq (h d) -> ... h seq d", h=self.num_heads) # gives multiple attention patterns
        k = rearrange(k, "... seq (h d) -> ... h seq d", h=self.num_heads)
        v = rearrange(v, "... seq (h d) -> ... h seq d", h=self.num_heads)

        mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool))
        
        if self.rope is not None:
            q = self.rope(q, token_positions)
            k = self.rope(k, token_positions)
            out = scaled_dot_product_attention(q, k, v, mask)
        else:
            out = scaled_dot_product_attention(q, k, v, mask)
        
        out = rearrange(out, "... h seq d -> ... seq (h d)")

        return self.O(out)