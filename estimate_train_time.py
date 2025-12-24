from cs336_basics.assignment_utils import estimate_train_days
from cs336_basics.assignment_utils import calculate_max_batch

TOKENIZER_VOCAB_SIZE = 50257

test_model_config = {
    "num_layers": 4,
    "num_heads": 16,
    "d_model": 512,
    "d_ff": 1344,   
    "vocab_size": TOKENIZER_VOCAB_SIZE,
    "context_length": 256
}

GPT2_XL = {
    "num_layers": 48,
    "d_model": 1600,
    "num_heads": 25,
    "d_ff": 4 * 1600,  # 6400
    "vocab_size": TOKENIZER_VOCAB_SIZE,
    "context_length": 1024
}


def estimate_time(model_config: dict, total_tokens: int, gpu_memory_size: int, gpu_flops, mfu):
    max_batch_size = calculate_max_batch(
        model_config,
        gpu_memory_size
    )

    steps = total_tokens // model_config["context_length"] // max_batch_size

    days = estimate_train_days(
        model_config,
        hardware_flops_tera=gpu_flops,
        mfu=mfu,
        train_steps=steps,
        batch_size=max_batch_size
    )
    hours = days * 24
    minutes = hours * 60
    print(f"Estimated training time is {hours} hours.")
    print(f"Estimated training time is {minutes} mins.")

if __name__ == "__main__":
    estimate_time(test_model_config,
                  total_tokens=531635766,
                  gpu_memory_size=24,
                  gpu_flops=73.54,
                  mfu=0.5)
    
    # estimate_time(test_model_config,
    #               total_tokens=531635766,
    #               gpu_memory_size=80,
    #               gpu_flops=67,
    #               mfu=0.5)
    