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
from cs336_basics.data import get_batch, CustomDataloader
from cs336_basics.utils import learning_rate_schedule, save_checkpoint, load_checkpoint, logger

TOKENIZER_VOCAB_SIZE = 50257


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
        max_lr=args.max_lr,
        min_lr=args.max_lr * 0.1,
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

def evaluate():
    model.eval()

    total_loss = 0
    with torch.no_grad():
        for step, (inputs, targets) in enumerate(val_dataloader.load_data()):
            logits = model(inputs)
            loss = cross_entropy(logits, targets)
            total_loss += loss
    return total_loss / step


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="model pre-train")

    # Model config 
    parser.add_argument("--vocab_size", type=int, default=TOKENIZER_VOCAB_SIZE, help="Model vocabulary size")
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
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="Device for training")
    parser.add_argument("--batch_size", type=int, help="Number of samples per batch")
    parser.add_argument("--train_data_path", type=str, default="../data/TinyStoriesV2-GPT4-train.npy", help="Path of the training dataset")
    parser.add_argument("--val_data_path", type=str, default="../data/TinyStoriesV2-GPT4-valid.npy", help="Path of the validation dataset")
    # parser.add_argument("--num_epochs", type=int, default=1, help="Number of training epochs")
    # parser.add_argument("--num_steps", type=int, help="Number of training steps")
    parser.add_argument("--save_steps", type=int, help="Number of steps to save checkpoints")
    parser.add_argument("--save_dir", type=str, default="../out", help="Directory to save checkpoints")
    parser.add_argument("--load_path", type=str, default=None, help="Path to load checkpoints")
    parser.add_argument("--log_steps", type=int, default=100, help="Number of steps to log training performance")
    parser.add_argument("--eval_steps", type=int, default=100, help="Number of steps to evaluate validation loss")
    parser.add_argument("--use_wandb", type=bool, default=False, help="Whether to use wandb to log")

    args = parser.parse_args()

    #--------------Init model, optimizer, dataloader---------------
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

    train_dataloader = CustomDataloader(args.train_data_path, 
                                        batch_size=args.batch_size,
                                        context_length=args.context_length,
                                        shuffle=True,
                                        device=args.device)
    val_dataloader = CustomDataloader(args.val_data_path, 
                                      batch_size=args.batch_size,
                                      context_length=args.context_length,
                                      shuffle=False,
                                      device=args.device)
    train_steps = len(train_dataloader)

    #--------------Init wandb---------------
    if args.use_wandb:
        wandb_run = wandb.init(
            entity="cs336_assign1",
            project="cs336_assignment1",
            config={
                "max_learning_rate": args.max_lr,
                "steps": train_steps,
            }
        )
    
    #--------------Training loop---------------
    start_time = time.time()
    num_epochs = 1
    for epoch in range(num_epochs):
        for step, (inputs, targets) in enumerate(train_dataloader.load_data(), 
                                                start=start_step):
            train_loss = train_step(inputs, targets, step)

            # Save checkpoints
            os.makedirs(args.save_dir, exist_ok=True)
            save_path = f"{args.save_dir}/{step + 1}.pt"
            if (step + 1) % args.save_steps == 0:
                save_checkpoint(model, optimizer, step + 1, save_path)

            # Log training performance
            if (step + 1) % args.log_steps == 0:
                lr = optimizer.param_groups[0]["lr"]
                cur_time = time.time()
                spent_time = (cur_time - start_time) // 60
                log_info = f"(Step: {step + 1}/{train_steps}), train_loss: {train_loss:.4f}, lr: {lr:.6f}, spent time: {spent_time}min"
                logger(log_info)
                if args.use_wandb:
                    wandb_log = {
                        "train loss": train_loss,
                        "lr": lr,
                        "spent time (min)": spent_time
                    }
                    wandb_run.log(wandb_log)

            # Evaluate validation loss
            if (step + 1) % args.eval_steps == 0:
                val_loss = evaluate()
                log_info = f"(Step: {step + 1}/{train_steps}), val_loss: {val_loss:.4f}"
                logger(log_info)
                if args.use_wandb:
                    wandb_log = {
                        "val loss": val_loss
                    }
                    wandb_run.log(wandb_log)

    if args.use_wandb:
        wandb_run.finish()