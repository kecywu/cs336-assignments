import torch
import torch.nn as nn 
import math
from einops import einsum

class Linear(nn.Module):

    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()

        weight = torch.empty((out_features, in_features), device=device, dtype=dtype)
        sigma = math.sqrt(2.0 / (in_features + out_features))
        nn.init.trunc_normal_(weight, 0.0, sigma, -3*sigma, 3*sigma)

        self.W = nn.Parameter(weight)

    
    def forward(self, x):
        y = einsum(self.W, x, "d_out d_in, ... d_in -> ... d_out")
        return y
