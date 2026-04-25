import torch 
from cs336_basics.softmax import softmax


def scaled_dot_product_attention(Q, K, V, mask=None):

    d_k = Q.shape[-1]
    scores = Q @ K.transpose(-2, -1) / (d_k ** 0.5)
    
    if mask is not None:
        scores = scores.masked_fill(mask == False, float("-inf"))
    
    attn = softmax(scores, -1)

    return attn @ V
