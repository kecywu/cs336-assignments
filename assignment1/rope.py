import torch
import torch.nn as nn 
from einops import einsum

class RotaryPositionalEmbedding(nn.Module):

    def __init__(self, theta, d_k, max_seq_len, device=None):
        super().__init__()

        frequencies = 1.0 / (theta ** (torch.arange(0, d_k, 2, device=device).float() / d_k))
        positions = torch.arange(max_seq_len, device=device).float()
        angle_grid = torch.outer(positions, frequencies)

        self.register_buffer("cos_cache", torch.cos(angle_grid), persistent=False)
        self.register_buffer("sin_cache", torch.sin(angle_grid), persistent=False)

    
    def forward(self, x, token_positions):

        cos = self.cos_cache[token_positions]
        sin = self.sin_cache[token_positions]

        x_reshaped = x.view(*x.shape[:-1], -1, 2) # reshape last dimension into pairs of 2
        x1 = x_reshaped[..., 0]
        x2 = x_reshaped[..., 1]

        x1_rotated = x1 * cos - x2 * sin
        x2_rotated = x2 * cos + x1 * sin
        x_rotated = torch.stack((x1_rotated, x2_rotated), dim=-1)

        return x_rotated.view_as(x)