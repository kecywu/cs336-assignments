import torch 
from einops import rearrange

def cross_entropy(logits, targets):

    target_indices = rearrange(targets, "... -> ... 1")
    correct_logits = torch.gather(logits, dim=-1, index=target_indices)
    correct_logits = rearrange(correct_logits, "... 1 -> ...")

    max_val, _ = torch.max(logits, dim=-1, keepdim=True)
    stable_logits = logits - max_val
    log_sum_exp = max_val.squeeze(-1) + torch.log(torch.sum(torch.exp(stable_logits), dim=-1))

    loss = -correct_logits + log_sum_exp 

    return torch.mean(loss)

