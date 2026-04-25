import torch 
import torch.nn as nn 
from cs336_basics import rmsnorm, multihead_self_attention, positionwise_feedforward

class TransformerBlock(nn.Module):

    def __init__(self, d_model, num_heads, d_ff, rope=None, device=None, dtype=None):

        super().__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff 

        self.rms1 = rmsnorm.RMSNorm(d_model, device=device, dtype=dtype)
        self.rope = rope 
        self.multiheadattention = multihead_self_attention.MultiHeadSelfAttention(d_model, num_heads, rope, device=device, dtype=dtype)
        self.rms2 = rmsnorm.RMSNorm(d_model, device=device, dtype=dtype)
        self.ffn = positionwise_feedforward.FFN(d_model, d_ff, device=device, dtype=dtype)


    def forward(self, x, token_positions=None):

        # default is the entire sequence length
        if self.rope is not None and token_positions is None:
            token_positions = torch.arange(x.shape[-2], device=x.device)

        pre_norm1 = self.rms1(x)
        attention = self.multiheadattention(pre_norm1, token_positions)
        add = attention + x 
        pre_norm2 = self.rms2(add)
        ffn = self.ffn(pre_norm2)

        return ffn + add
