import torch
import torch.nn as nn 
from cs336_basics.linear import Linear

class FFN(nn.Module):

    def __init__(self, d_model, d_ff, device=None, dtype=None):
        super().__init__()

        self.d_model = d_model 
        self.d_ff = d_ff
        self.W1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.W3 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.W2 = Linear(d_ff, d_model, device=device, dtype=dtype)


    def forward(self, x):

        gate = self.W1(x)
        silu = gate * torch.sigmoid(gate)
        
        return self.W2(silu * self.W3(x))

