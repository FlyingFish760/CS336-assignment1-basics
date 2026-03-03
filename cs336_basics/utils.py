import os
import math
import random
from collections.abc import Callable, Iterable

import torch
import torch.nn as nn
import torch.distributed as dist
import numpy as np


################### Saving tools ##############################
def save_checkpoint(model: nn.Module | nn.parallel.DistributedDataParallel,
                    optimizer: torch.optim.Optimizer,
                    iteration: int,
                    out: str | os.PathLike):
    state_dict = {}
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model_state_dict = model.module.state_dict()
    else:
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

################### Other tools ##############################
def setup_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

################### Distributed training tools ##############################
def is_main_process():
    return not dist.is_initialized() or dist.get_rank() == 0

################### Trainer tools ##############################
from .model import TransformerLM
from .optimizer import AdamW

def init_model_optimizer(model_opt_config: dict, device: str):
    model = TransformerLM(
        vocab_size=model_opt_config["vocab_size"],
        d_model=model_opt_config["d_model"],
        num_heads= model_opt_config["num_heads"],
        d_ff = model_opt_config["d_ff"],
        context_length=model_opt_config["context_length"],
        theta = model_opt_config["theta"],
        num_layers=model_opt_config["num_layers"]
    )

    optimizer = AdamW(
        model.parameters(),
        model_opt_config["max_lr"],
        betas=model_opt_config["betas"],
        weight_decay=model_opt_config["weight_decay"],
        eps=1e-8
    )

    model = model.to(device)
    model.compile(mode="reduce-overhead")
    return model, optimizer

if __name__ == "__main__":

    test_model_config = {
        "num_layers": 4,
        "num_heads": 16,
        "d_model": 512,
        "d_ff": 1344,   
        "vocab_size": 50257,
        "context_length": 256
    }
    # model = TransformerLM(
    #     vocab_size=50257,
    #     d_model=512,
    #     d_ff=1344,
    #     num_heads=16,
    #     num_layers=4,
    #     context_length=256
    # )