import argparse
import time
import os

import numpy as np
import torch
from torch import Tensor
from jaxtyping import Int, Float
import wandb

from cs336_basics.model import TransformerLM
from cs336_basics.nn_utils import cross_entropy
from cs336_basics.optimizer import AdamW
from cs336_basics.data import get_batch
from cs336_basics.utils import learning_rate_schedule, save_checkpoint, load_checkpoint, logger


def train_step(inputs: Int[Tensor, "b seq_len"],
                targets: Int[Tensor, "b seq_len"],
                step: int) -> Float[Tensor, ""]:
    '''
    One training epoch of the complete given data.

    inputs: Input token ids;
    targets: Target token ids;

    '''
    optimizer.zero_grad()
    # Set the optimizer learning rate
    lr = learning_rate_schedule(
        step, 
        max_lr=args.max_lr,
        min_lr=args.max_lr * 0.1,
        warmup_iters=10,
        cosine_cycle_iters=50
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
    parser = argparse.ArgumentParser(description="model pre-train")

    # Other configs
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="Device to train on")

    # Data config 
    parser.add_argument("--data_path", type=str, default="", help="")

    # Model config 
    parser.add_argument("--vocab_size", type=int, default=50257, help="Model vocabulary size")
    parser.add_argument("--d_model", type=int, default=1600, help="Model hidden dimension size")
    parser.add_argument("--num_heads", type=int, default=25, help="Number of attention heads")
    parser.add_argument("--d_ff", type=int, default=6400, help="Feed-foward network dimension size")
    parser.add_argument("--context_length", type=int, default=1024, help="Context length")
    parser.add_argument("--theta", type=float, default=10000.0, help="Base angle of RoPE")
    parser.add_argument("--num_layers", type=int, default=48, help="Number of transformer blocks")

    # Optimizier config
    parser.add_argument("--max_lr", type=float, default=1e-5, help="Maximum learning rate")
    parser.add_argument("--betas", type=list, default=[0.9, 0.99], help="Moment updating parameters of the AdamW optimizer")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="Weigth decay rate of the AdamW optimizer")

    # Trainer config
    parser.add_argument("--num_epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--num_steps", type=int, help="Number of training steps")
    parser.add_argument("--save_steps", type=int, help="Number of steps to save checkpoints")
    parser.add_argument("--save_dir", type=str, default="../out", help="Directory to save checkpoints")
    parser.add_argument("--load_path", type=str, default=None, help="Path to load checkpoints")
    parser.add_argument("--log_steps", type=int, default=100, help="Number of steps to log training/validation performance")

    args = parser.parse_args()

    #--------------Load data---------------
    # data = np.load(args.data_path, mmap_mode="r")
    data = np.random.randint(0, high=args.vocab_size, size=(512,))

    #--------------Init model and optimizer---------------
    model = TransformerLM(
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        num_heads= args.num_heads,
        d_ff = args.d_ff,
        context_length=args.context_length,
        theta = args.theta,
        num_layers=args.num_layers
    )

    start_lr = 0.1 * args.max_lr
    optimizer = AdamW(
        model.parameters(),
        start_lr,
        betas=args.betas,
        weight_decay=args.weight_decay,
        eps=1e-8
    )

    # Load checkpoints if needed
    if args.load_path is not None:
        start_step = load_checkpoint(args.load_path, model, optimizer)
    else:
        start_step = 0

    #--------------Init wandb---------------
    wandb_run = wandb.init(
        entity="cs336_assign1",
        project="cs336_assignment1",
        config={
            "max_learning_rate": args.max_lr,
            "steps": args.num_steps,
        }
    )
    
    #--------------Training loop---------------
    start_time = time.time()
    for step in range(start_step, args.num_steps):
        inputs, targets = get_batch(
            data,
            batch_size=4,
            context_length=args.context_length,
            device=args.device
        )
        print(inputs.dtype)
        train_loss = train_step(inputs, targets, step + 1)

        # Save checkpoints
        os.makedirs(args.save_dir, exist_ok=True)
        save_path = f"{args.save_dir}/{step}.pt"
        if (step + 1) % args.save_steps == 0:
            save_checkpoint(model, optimizer, step, save_path)

        # Log training/ validation performance
        if (step + 1) * args.log_steps == 0:
            lr = optimizer.param_groups[0]["lr"]
            cur_time = time.time()
            spent_time = (cur_time - start_time) // 60
            log_info = f"(Step: {step + 1}/{args.num_steps}), train_loss: {train_loss:.4f}, lr: {lr:.6f}, spent time: {spent_time}min"
            logger(log_info)
            wandb_log = {
                "train loss": train_loss,
                "lr": lr,
                "spent time (min)": spent_time
            }
            wandb_run.log(wandb_log)

    wandb_run.finish()