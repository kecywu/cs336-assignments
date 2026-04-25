import torch
import torch.nn as nn 


class RMSNorm(nn.Module):

    def __init__(self, d_model, eps=1e-5, device=None, dtype=None):

        super().__init__()

        self.d_model = d_model
        self.eps = eps
        g = torch.ones(d_model, device=device, dtype=dtype) 
        self.g = nn.Parameter(g)
        # don't save device and dtype in initialization, might cause inconsistency

    def forward(self, x):

        in_dtype = x.dtype
        x = x.to(torch.float32)

        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        result = (x / rms) * self.g

        return result.to(in_dtype)