import torch 
import torch.nn as nn
from cs336_basics.transformer_block import TransformerBlock
from cs336_basics.embedding import Embedding
from cs336_basics.linear import Linear
from cs336_basics.rmsnorm import RMSNorm

class Transformer(nn.Module):

    def __init__(self, d_model, num_heads, d_ff, vocab_size, context_length, num_layers, rope=None, device=None, dtype=None):

        super().__init__()

        self.vocab_size = vocab_size
        self.context_length = context_length
        self.num_layers = num_layers

        self.embedding = Embedding(vocab_size, d_model, device=device, dtype=dtype)
        self.rope = rope
        
        self.transformer_block = nn.ModuleList(
            [
                TransformerBlock(d_model, num_heads, d_ff, rope, device=device, dtype=dtype)
                for _ in range(num_layers)
            ]
        )

        self.rms = RMSNorm(d_model, device=device, dtype=dtype)
        self.linear = Linear(d_model, vocab_size, device=device, dtype=dtype)


    def forward(self, x):

        assert x.shape[-1] <= self.context_length
        
        token_positions = torch.arange(x.shape[-1], device=x.device)
        embedding = self.embedding(x)

        for i in range(self.num_layers):
            embedding = self.transformer_block[i](embedding, token_positions)
        
        norm = self.rms(embedding)
        output = self.linear(norm)

        return output
