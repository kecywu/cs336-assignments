import torch
import torch.nn as nn 

class Embedding(nn.Module):

    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()

        emb = torch.empty((num_embeddings, embedding_dim), device=device, dtype=dtype)
        nn.init.trunc_normal_(emb, 0.0, 1.0, -3.0, 3.0)
        self.embedding = nn.Parameter(emb)
    
    def forward(self, token_ids):

        return self.embedding[token_ids]