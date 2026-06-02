def compute_llama3_FLOPs(**kwargs):
    seq_len = kwargs["seq_len"]
    d_model = kwargs["d_model"]
    d_ff = kwargs["d_ff"]
    num_heads = kwargs["num_heads"]
    d_head = kwargs["d_head"]
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
):
    flops_per_seq = compute_llama3_FLOPs(**cfg)
    flops_per_batch = batch_size * flops_per_seq
    num_batches = flops_budget / flops_per_batch

    return num_batches


if __name__ == "__main__":
    # seq_len      = 4096
    # d_model      = 4096
    # d_ff         = 11008
    # num_heads    = 32
    # d_head       = 128
    # num_layers   = 32
    # vocab_size   = 32000

    model_config = {
        "seq_len": 512,
        "d_model": 512,
        "d_ff": 1344,
        "num_heads": 16,
        "d_head": 32,
        "num_layers": 4,
        "vocab_size": 50257
    }

    # Test compute_llama3_FLOPs
    llama3_training_flops = compute_llama3_FLOPs(
        # seq_len=seq_len,
        # d_model=d_model,
        # d_ff=d_ff,
        # num_heads=num_heads,
        # d_head=d_head,
        # vocab_size=vocab_size,
        # num_layers=num_layers,
        **model_config
    )

    print(f"Estimated training FLOPs is {llama3_training_flops:.3e}")

    # Test compute_llama3_train_batch
    # flops_budget = 1.6 * 10 ** 18
    # batch_size = 64

    # num_batches = compute_llama3_train_batches(
    #     flops_budget,
    #     batch_size,
    #     **model_config
    # )
    # print(f"Estimated number of training batches is {num_batches}")

    # Estimate CS336 leaderboard compute budget referring to a student writeup
    # model_config_student = {
    #     "seq_len": 512,
    #     "d_model": 1280,
    #     "d_ff": 3456,
    #     "num_heads": 16,
    #     "d_head": 80,
    #     "num_layers": 12,
    #     "vocab_size": 50257
    # }
    # batch_size = 128
    # train_steps = 13000
    # flops_per_seq = compute_llama3_FLOPs(
    #     **model_config_student
    # )
    # total_flops = batch_size * train_steps * flops_per_seq
    # print(f"Estimated training FLOPs is {total_flops:.3e}")