from collections.abc import Callable, Iterable
from typing import Optional
import torch
import torch.nn as nn
import math

class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]
                t = state.get("t", 0)
                grad = p.grad.data
                p.data -= lr / math.sqrt(t + 1) * grad
                state["t"] = t + 1

        return loss
    
class AdamW(torch.optim.Optimizer):
    def __init__(self, 
                 params: Iterable[torch.Tensor], 
                 lr: float, 
                 betas: list[float],
                 weight_decay: float, 
                 eps: float = 1e-8):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        # if not 0 < beta1 < 1:
        #     raise ValueError(f"Invalid beta1: {beta1}")
        # if not 0 < beta2 < 1:
        #     raise ValueError(f"Invalid beta2: {beta2}")
        if not 0 < weight_decay < 1:
            raise ValueError(f"Invalid weight decay rate: {weight_decay}")
        if eps < 0:
            raise ValueError(f"Invalid epsilon: {eps}")
        
        defaults = {
            "lr": lr,
            "betas": betas,
            "weight_decay": weight_decay,
            "eps": eps
        }
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            weight_decay = group["weight_decay"]
            eps = group["eps"]

            for p in group["params"]:
                if p.grad is None: 
                    continue

                # Accquire state variables
                state = self.state[p]
                t = state.get("t", 1)
                m = state.get("m", torch.zeros_like(p))
                v = state.get("v", torch.zeros_like(p))

                # Update parameters
                grad = p.grad.data
                m = beta1 * m + (1 - beta1) * grad
                v = beta2 * v + (1 - beta2) * grad ** 2
                grad_des_rate = lr * (math.sqrt(1 - beta2 ** t) / (1 - beta1 ** t))
                p.data -= grad_des_rate * (m / (torch.sqrt(v) + eps))
                p.data -= lr * weight_decay * p.data

                # Update states
                state["m"] = m
                state["v"] = v
                state["t"] = t + 1

        return loss

    

if __name__ == "__main__":
    torch.manual_seed(123)
    weights = torch.nn.Parameter(5 * torch.randn((10, 10)))
    # opt = SGD([weights], lr=1)

    # for t in range(100):
    #     opt.zero_grad() # Reset the gradients for all learnable parameters.
    #     loss = (weights**2).mean() # Compute a scalar loss value.
    #     print(loss.cpu().item())
    #     loss.backward() # Run backward pass, which computes gradients.
    #     opt.step() # Run optimizer step.

    # def train(weights, lr):
    #     opt = SGD([weights], lr)

    #     for t in range(10):
    #         opt.zero_grad() # Reset the gradients for all learnable parameters.
    #         loss = (weights**2).mean() # Compute a scalar loss value.
    #         print(loss.cpu().item())
    #         loss.backward() # Run backward pass, which computes gradients.
    #         opt.step() # Run optimizer step.

    # weights = torch.nn.Parameter(5 * torch.randn((10, 10)))
    # for lr in [1e1,1e2, 1e3]:
    #     print("-------------------------")
    #     train(weights, lr)

    lr, betas, weight_decay, eps = 1e-1, [0.9, 0.99], 1e-3, 1e-8
    opt = AdamW([weights],
                lr, 
                betas,
                weight_decay,
                eps)

    # for t in range(10):
    opt.zero_grad() # Reset the gradients for all learnable parameters.
    loss = (weights**2).mean() # Compute a scalar loss value.
    print(loss.cpu().item())
    loss.backward() # Run backward pass, which computes gradients.
    opt.step() # Run optimizer step.
    print(opt.state_dict())

