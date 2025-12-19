import torch
from torch import Tensor

from einops import rearrange, reduce
from jaxtyping import Float, Int

def softmax(x: torch.tensor, dim: int):
    x_exp = torch.exp(x - torch.max(x, dim=dim, keepdim=True).values)  # substract max value of the dim dimension to ensure numeric stability
    x_exp_sum = torch.sum(x_exp, dim=dim, keepdim=True)
    return x_exp / x_exp_sum

def cross_entropy(predicted: Float[Tensor, "... seq_len vocab_size"], 
                  targets: Int[Tensor, "... seq_len"]) -> Float[Tensor, ""]:
    # pred_flattened = rearrange(predicted, "... seq_len vocab_size -> (... seq_len) vocab_size")   # (a, b)
    # targets_flattened = rearrange(targets, "... seq_len -> (... seq_len)")   # (a,)
    # pred_ind_target = pred_flattened[torch.arange(len(targets_flattened)), targets_flattened]

    # Use math: cancel log and exp whenever possible
    # Subtract the largest element for numerical stability
    predicted -= torch.max(predicted, dim=-1, keepdim=True).values
    pred_ind_target = torch.gather(input=predicted, 
                                   dim=-1,
                                   index=targets.unsqueeze(-1))   # (... seq_len 1)
    pred_ind_target = pred_ind_target.squeeze(-1)   # (... seq_len)
    sum_exp = reduce(torch.exp(predicted), "... seq_len vocab_size -> ... seq_len", "sum")
    neg_log_like = -(pred_ind_target - torch.log(sum_exp))   # (... seq_len)
    loss = torch.mean(neg_log_like.flatten())

    # # Use predefined softmax function
    # softmax_pred = softmax(predicted, dim=-1)   # (... seq_len vocab_size)
    # p_ind_target = torch.gather(
    #     input=softmax_pred, 
    #     dim=-1,
    #     index=targets.unsqueeze(-1)
    # )   # (... seq_len 1)
    # p_ind_target = p_ind_target.squeeze(-1)    # (... seq_len)
    # neg_log_like = -torch.log(p_ind_target)
    # loss = torch.mean(neg_log_like.flatten())

    return loss

if __name__ == "__main__":
    torch.manual_seed(123)
    b, seq_len, vocab_size = 4, 16, 512
    predicted = torch.randn((b, seq_len, vocab_size))
    targets = torch.randint(low=0, high=vocab_size, size=(b, seq_len))
    loss = cross_entropy(predicted, targets)
    print(loss)