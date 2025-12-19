import os
import math
from collections.abc import Callable, Iterable

import torch
import torch.nn as nn


################### Saving tools ##############################
def save_checkpoint(model: nn.Module,
                    optimizer: torch.optim.Optimizer,
                    iteration: int,
                    out: str | os.PathLike):
    state_dict = {}
    model_state_dict = model.state_dict()
    opt_state_dict = optimizer.state_dict()
    
    state_dict["model_state_dict"] = model_state_dict
    state_dict["opt_state_dict"] = opt_state_dict
    state_dict["iter"] = iteration
    torch.save(state_dict, out)

def load_checkpoint(src: str | os.PathLike,
                    model: nn.Module,
                    optimizer: torch.optim.Optimizer):
    state_dict = torch.load(src, weights_only=False)
    model_state_dict = state_dict["model_state_dict"]
    opt_state_dict = state_dict["opt_state_dict"]
    iter = state_dict["iter"]

    model.load_state_dict(model_state_dict)
    optimizer.load_state_dict(opt_state_dict)
    return iter



################### Optimizer tools ##############################
def learning_rate_schedule(it: int, 
                           max_lr: float, 
                           min_lr: float, 
                           warmup_iters: int, 
                           cosine_cycle_iters: int) -> float:
    '''
    It: starts from 1
    '''
    if it <= 0:
        raise ValueError(f"Wrong iteration of {it}.")
    if it < warmup_iters:
        lr = it / warmup_iters * max_lr
    elif it <= cosine_cycle_iters:
        lr = min_lr + (max_lr - min_lr) * 0.5 * (1 + math.cos(((it - warmup_iters) / (cosine_cycle_iters - warmup_iters)) * math.pi))
    else:
        lr = min_lr
    return lr

def gradient_clipping_(params: Iterable[nn.Parameter], max_l2_norm: float):
    '''
    Clip gradients of parameters in place.
    '''
    grads = [param.grad for param in params if param.grad is not None]
    total_squared_norm = torch.zeros((1,))
    for g in grads:
        total_squared_norm += torch.sum(torch.square(g))
    l2_norm = torch.sqrt(total_squared_norm)
    if l2_norm > max_l2_norm:
        scale_factor = max_l2_norm / (l2_norm + 1e-6)
        for param in params:
            if param.grad is not None:
                param.grad.mul_(scale_factor)

################### Logging tools ##############################
def logger(content):
    print(content)