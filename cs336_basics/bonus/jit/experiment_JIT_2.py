'''
Compile the model: 有提速结果->
模型越大：speed up越小
数据越大：speed up越大

'''


import argparse
import time
import os

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader
from jaxtyping import Int, Float
import wandb

from cs336_basics.model import TransformerLM
from cs336_basics.nn_utils import cross_entropy
from cs336_basics.optimizer import AdamW
from cs336_basics.data import get_batch, PretrainDataset
from cs336_basics.utils import learning_rate_schedule, save_checkpoint, load_checkpoint, logger

TOKENIZER_VOCAB_SIZE = 50257

test_model_config_1 = {
    "num_layers": 4,
    "num_heads": 16,
    "d_model": 512,
    "d_ff": 1344,   
    "vocab_size": TOKENIZER_VOCAB_SIZE,
    "context_length": 256
}

test_model_config = {
    "num_layers": 24,
    "num_heads": 25,
    "d_model": 1600,
    "d_ff": 6400,   
    "vocab_size": TOKENIZER_VOCAB_SIZE,
    "context_length": 256
}

max_lr = 1e-4
betas = [0.9, 0.99]
weight_decay = 1e-5

train_data_path = ""
batch_size = 0
device = ""

compile_mode = "reduce-overhead"

def timed(fn):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    result = fn()
    end.record()
    torch.cuda.synchronize()
    return result, start.elapsed_time(end) / 1000

def train_step(inputs: Int[Tensor, "b seq_len"],
                targets: Int[Tensor, "b seq_len"],
                step: int) -> Float[Tensor, ""]:
    '''
    One training epoch of the complete given data.

    inputs: Input token ids;
    targets: Target token ids;

    '''
    model.train()

    optimizer.zero_grad()
    # Set the optimizer learning rate
    lr = learning_rate_schedule(
        step + 1, 
        max_lr=max_lr,
        min_lr=max_lr * 0.1,
        warmup_iters=int(train_steps * 0.1),
        cosine_cycle_iters=int(train_steps * 0.9)
    )
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    # Forward pass
    logits = model(inputs)

    # Compute loss
    loss = cross_entropy(logits, targets)

    # Back proporgation (to get gradients)
    loss.backward()

    # Optimizer step
    optimizer.step()

    return loss

if __name__ == "__main__":
    train_ds = PretrainDataset(train_data_path,
                               context_length=test_model_config["context_length"])
    train_dataloader = DataLoader(train_ds, 
                                  batch_size=batch_size,
                                  shuffle=True,
                                  drop_last=True,
                                  num_workers=0)
    train_steps = len(train_dataloader)

    ################# Eager ###################

    #--------------Init model, optimizer---------------
    model = TransformerLM(
        vocab_size=test_model_config["vocab_size"],
        d_model=test_model_config["d_model"],
        num_heads= test_model_config["num_heads"],
        d_ff = test_model_config["d_ff"],
        context_length=test_model_config["context_length"],
        theta = 10000,
        num_layers=test_model_config["num_layers"]
    )

    start_lr = 0.1 * max_lr
    optimizer = AdamW(
        model.parameters(),
        start_lr,
        betas=betas,
        weight_decay=weight_decay,
        eps=1e-8
    )
    
    #--------------Training loop---------------
    model = model.to(device)

    eager_times = []
    start_step = 0
    for step, (inputs, targets) in enumerate(train_dataloader, 
                                            start=start_step):
        if step < 10:
            # Train step
            inputs = inputs.to(device)
            targets = targets.to(device)
            _, eager_time = timed(lambda: train_step(inputs, targets, step))
            eager_times.append(eager_time)
            print(f"eager train time {step}: {eager_time}")
        else: break
    print("~" * 10)

    del model
    del optimizer

    ################# Compile ###################

    #--------------Init model, optimizer---------------
    model = TransformerLM(
        vocab_size=test_model_config["vocab_size"],
        d_model=test_model_config["d_model"],
        num_heads= test_model_config["num_heads"],
        d_ff = test_model_config["d_ff"],
        context_length=test_model_config["context_length"],
        theta = 10000,
        num_layers=test_model_config["num_layers"]
    )

    start_lr = 0.1 * max_lr
    optimizer = AdamW(
        model.parameters(),
        start_lr,
        betas=betas,
        weight_decay=weight_decay,
        eps=1e-8
    )

    model = model.to(device)
    model.compile(mode="reduce-overhead")

    #--------------Training loop---------------

    compile_times = []
    start_step = 0
    for step, (inputs, targets) in enumerate(train_dataloader, 
                                            start=start_step):
        if step < 10:
            # Train step
            inputs = inputs.to(device)
            targets = targets.to(device)
            _, compile_time = timed(lambda: train_step(inputs, targets, step))
            compile_times.append(compile_time)
            print(f"compile train time {step}: {compile_time}")
        else: break
    print("~" * 10)


    eager_med = np.median(eager_times)
    compile_med = np.median(compile_times)
    speedup = eager_med / compile_med
    assert speedup > 1
    print(
        f"(train) eager median: {eager_med}, compile median: {compile_med}, speedup: {speedup}x"
    )
    print("~" * 10)