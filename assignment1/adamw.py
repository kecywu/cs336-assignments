import torch 
from collections.abc import Callable, Iterable
from typing import Optional
import math


class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=0.001, weight_decay=0.01, betas=(0.9, 0.999), eps=1e-8):

        defaults = {"lr": lr, "betas": betas, "lambda": weight_decay, "eps": eps}
        super().__init__(params, defaults)
    
    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()

        for group in self.param_groups:
            lr, betas, weight_decay, eps = group["lr"], group["betas"], group["lambda"], group["eps"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p] 
                t = state.get("t", 1) 
                m = state.get("m", 0)
                v = state.get("v", 0)

                grad = p.grad.data 
                alpha_t = lr*math.sqrt(1-math.pow(betas[1], t)) / (1-math.pow(betas[0], t))
                p.data = p.data * (1 - lr*weight_decay) # note weight decay happens before moment update
                state["m"] = betas[0]*m + (1-betas[0])*grad
                state["v"] = betas[1]*v + (1-betas[1])*torch.pow(grad, 2)
                p.data -= alpha_t * state["m"] / (torch.sqrt(state["v"]) + eps)
                state["t"] = t + 1 

        return loss