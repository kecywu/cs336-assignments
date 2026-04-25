import torch

def softmax(x, i):

    max_val, _ = torch.max(x, dim=i, keepdim=True)
    stable_x = x - max_val
    exp_x = torch.exp(stable_x)
    exp_x_sum = torch.sum(exp_x, dim=i, keepdim=True)

    return exp_x / exp_x_sum
