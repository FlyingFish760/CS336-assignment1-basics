import os
import math
import random
from collections.abc import Callable, Iterable
from typing import Dict

import torch
from torch import Tensor
import torch.nn as nn
import torch.distributed as dist
from jaxtyping import Float
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
from .model import TransformerLM, TransformerLM_fullAttnRes, TransformerLM_BlockAttnRes
from .optimizer import AdamW

def init_model_optimizer(model_opt_config: dict, device: str):
    model_arch = model_opt_config["model_architecture"]
    assert model_arch in ["std_Transformer", "full_attn_res", "block_attn_res"], "model_architecture must be 'std_Transformer'/ 'full_attn_res'/ 'block_attn_res'"

    if model_arch == "std_Transformer":
        print("----------- Model arch: Standard Tranformer -----------")
        model = TransformerLM(
            vocab_size=model_opt_config["vocab_size"],
            d_model=model_opt_config["d_model"],
            num_heads= model_opt_config["num_heads"],
            d_ff = model_opt_config["d_ff"],
            context_length=model_opt_config["context_length"],
            theta = model_opt_config["theta"],
            num_layers=model_opt_config["num_layers"],
            use_LN=model_opt_config["use_layer_norm"]
        )
    elif model_arch == "full_attn_res":
        print("----------- Model arch: Full Attention Residual -----------")
        model = TransformerLM_fullAttnRes(
            d_model=model_opt_config["d_model"],
            d_ff=model_opt_config["d_ff"],
            num_heads=model_opt_config["num_heads"],
            num_layers=model_opt_config["num_layers"],
            context_length=model_opt_config["context_length"],
            theta=model_opt_config["theta"],
            vocab_size=model_opt_config["vocab_size"],
        )
    elif model_arch == "block_attn_res":
        print("----------- Model arch: Block Attention Residual -----------")
        model = TransformerLM_BlockAttnRes(
            d_model=model_opt_config["d_model"],
            d_ff=model_opt_config["d_ff"],
            num_heads=model_opt_config["num_heads"],
            num_layers=model_opt_config["num_layers"],
            context_length=model_opt_config["context_length"],
            theta=model_opt_config["theta"],
            vocab_size=model_opt_config["vocab_size"],
            block_size=model_opt_config["block_size"],
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

# def get_embed_layer_norm(model: nn.Module):
#     for p 

# def get_trf_layers_norm():

# def get_out_layer_norm():

def get_l2_grad_norm(params: Iterable[nn.Parameter]) -> float:
    total_grad_norm = 0
    for p in params:
        if p.requires_grad:
            total_grad_norm += p.grad.detach().pow(2).sum()

    return total_grad_norm.sqrt().item()

def get_layer_grad_norms(model: TransformerLM) -> Dict[str, float]:
    layer_grad_norms = {}

    # Get the grad norm of token embedding layer 
    embed_params = model.token_embedding.parameters()
    layer_grad_norms["embedding"] = get_l2_grad_norm(embed_params)

    # Get the grad norm of transformer block layers
    for i, trf_block in enumerate(model.transformer_blocks):
        layer_grad_norms[f"transformer_{i+1}"] = get_l2_grad_norm(trf_block.parameters())

    # Get the grad norm of output linear layer 
    output_params = model.out_proj.parameters()
    layer_grad_norms["output_linear"] = get_l2_grad_norm(output_params)

    return layer_grad_norms

def get_global_grad_norm(model: TransformerLM) -> float:
    return get_l2_grad_norm(model.parameters())

def compute_llama3_FLOPs(**kwargs) -> int:
    seq_len = kwargs["context_length"]
    d_model = kwargs["d_model"]
    d_ff = kwargs["d_ff"]
    num_heads = kwargs["num_heads"]
    d_head = d_model // num_heads
    vocab_size = kwargs["vocab_size"]
    num_layers = kwargs["num_layers"]

    ######## MHA ########

    # Q/K/V projection
    flops_qkv_proj = 3 * 2 * seq_len * d_model * (num_heads * d_head)  # (H, B, D_h)
    # Q @ K -> logits
    flops_qk_logits = 2 * num_heads * seq_len * d_head * seq_len   # (H, B, B)
    # softmax(logits) -> weights
    flops_softmax_logits = 3 * num_heads * seq_len * seq_len   # (H, B, B)
    # weights * V 
    flops_query_reduction = 2 * num_heads * seq_len * seq_len * d_head   # (H, B, D_h)
    # final linear
    flops_attn_linear = 2 * seq_len * (num_heads * d_head) * d_model

    # total
    flops_attn = flops_qkv_proj + flops_qk_logits + flops_softmax_logits + flops_query_reduction + flops_attn_linear

    ######## Feed Forward (use SwiGLU) ########
    flops_ff = 3 * 2 * seq_len * d_model * d_ff

    ######## Final logits ########
    flops_final_logits = 2 * seq_len * d_model * vocab_size
    flops_final_softmax = 3 * seq_len * vocab_size


    forward_flops = num_layers * (flops_attn + flops_ff) + flops_final_logits + flops_final_softmax
    total_training_flops = forward_flops * 3
    return total_training_flops

def compute_llama3_train_batches(
        flops_budget,
        batch_size,
        **cfg
) -> int:
    flops_per_seq = compute_llama3_FLOPs(**cfg)
    flops_per_batch = batch_size * flops_per_seq
    num_batches = flops_budget // flops_per_batch

    return num_batches

def compute_perplexity(loss: Float[Tensor, ""]) -> Float[Tensor, ""]:
    '''
    perplexity = exp(1/m * (loss_1 + loss_2 + ... + loss_m)) for a sequnce tokens of length m

    Args:
        loss: Float[Tensor, ""]. Average loss over a batch of losses

    Returns:
        perplexity: Float[Tensor, ""]
    '''
    return torch.exp(loss)

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