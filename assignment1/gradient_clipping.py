import torch

def gradient_clipping(params, max_l2_norm, eps=1e-6):

    grads = [p.grad for p in list(params) if p.grad is not None]
    total_norm = torch.sqrt(sum(torch.sum(g ** 2) for g in grads))

    if total_norm > max_l2_norm:
        scale = max_l2_norm / (total_norm + eps)
        for g in grads:
            g *= scale