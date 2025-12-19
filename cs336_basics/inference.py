from typing import Optional

from jaxtyping import Float
import torch
import torch.nn as nn
from torch import Tensor
from transformers import PreTrainedTokenizer, AutoTokenizer
from einops import rearrange

from cs336_basics.nn_utils import softmax
from cs336_basics.model import TransformerLM

def softmax_with_temperature(logits: Float[Tensor, "... vocab_size"],
                             dim: int,
                             temperature: float) -> Float[Tensor, "... vocab_size"]:
    if temperature <= 0:
        raise ValueError(f"Invalid temperature value of {temperature}, should be > 0.")
    scaled_logits = logits / temperature
    return softmax(scaled_logits, dim=dim)

def apply_top_p(prob_dist: Float[Tensor, "... vocab_size"],
                thres_p: float) -> Float[Tensor, "... vocab_size"]:
    if thres_p <= 0 or thres_p > 1:
        raise ValueError(f"Invalid thres_p value of {thres_p}, should be between (0, 1].")

    original_shape = prob_dist.shape

    prob_dist = rearrange(prob_dist, "... vocab_size -> (...) vocab_size")
    for b in range(prob_dist.shape[0]):
        probs = prob_dist[b]

        # Get the indices of the top_p probabilities
        probs_sorted = probs.sort(descending=True)
        probs_sorted, probs_sorted_inds = probs_sorted.values, probs_sorted.indices
        sum_probs = 0
        for i in range(len(probs_sorted)):
            sum_probs += probs_sorted[i]
            if sum_probs >= thres_p:
                break
        top_p_inds = probs_sorted_inds[:i + 1].tolist()
        
        # Modify the probabiltiy distribution
        for i in range(len(probs)):
            if i in top_p_inds:
                probs[i] /= sum_probs
            else:
                probs[i] = 0

    prob_dist = prob_dist.view((*original_shape[:-1], -1))

    return prob_dist


def generate(prompt: str, 
             tokenizer: PreTrainedTokenizer,
             model: nn.Module,
             max_generate_tokens: int, 
             temperature: Optional[float],
             thres_p: Optional[float]):
    token_ids = tokenizer(prompt, return_tensors="pt").input_ids   # (b seq_len)
    for _ in range(max_generate_tokens):
        logits = model(token_ids)   # (b seq_len vocab_size)
        logits_last = logits[..., -1, :]   # (b vocab_size)

        # Apply softmax (with temperature)
        if temperature is not None:
            logits_last_norm = softmax_with_temperature(logits_last, dim=-1, temperature=temperature)   # (b vocab_size)
        else:
            logits_last_norm = softmax(logits_last, dim=-1)   # (b vocab_size)

        # Apply top-p
        if thres_p is not None:
            logits_last_norm = apply_top_p(logits_last_norm, thres_p)    # (b vocab_size)

        # Sample from the probability distribution
        next_token_ids = torch.multinomial(logits_last_norm, num_samples=1)   # (b, 1)

        # Append the new token to the inputs
        token_ids = torch.cat((token_ids, next_token_ids), dim=-1)

        print(token_ids)
        text_generated = tokenizer.batch_decode(token_ids)
        print(text_generated)
    


if __name__ == "__main__":
    b, seq_len, vocab_size = 4, 16, 50256
    logits = torch.randn((b, seq_len, vocab_size))
    # prob_dist = softmax_with_temperature(logits, 0.1)
    # prob_dist_p = apply_top_p(prob_dist, thres_p=0.5)
    # print(prob_dist_p.shape)

    model = TransformerLM(
        vocab_size,
        d_model=16,
        num_heads=2, 
        d_ff=64,
        context_length=64,
        theta=10000,
        num_layers=4
    )
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    generate(prompt="Hello world,",
             tokenizer = tokenizer,
             model=model,
             max_generate_tokens=16,
             temperature=0.5,
             thres_p=0.8
             )
    