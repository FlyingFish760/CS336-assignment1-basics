import torch
import torch.nn as nn

def get_model_parameter(model: nn.Module):
    return sum(param.numel() for param in model.parameters())

def get_model_size(model: nn.Module):
    # Get size of parameters()
    total_params = 0
    total_params_bytes = 0
    for param in model.parameters():
        n_elements = param.numel()
        element_bytes = param.element_size()
        total_params += n_elements
        total_params_bytes += n_elements * element_bytes
    total_params_gb = total_params_bytes / (2 ** 30)

    # Get size of buffers()
    total_buffers = 0
    total_buffers_bytes = 0
    for buff in model.buffers():
        n_elements = buff.numel()
        element_bytes = buff.element_size()
        total_buffers += n_elements
        total_buffers_bytes += n_elements * element_bytes
    total_buffers_gb = total_buffers_bytes / (2 ** 30)

    print(f"Number of parameters: {total_params:,}.")
    print(f"Size of parameters: {total_params_gb:.2f} GB.")
    print(f"Number of parameters in training (with gradients): {total_params*2:,}.")
    print(f"Size of parameters in training (with gradients): {total_params_gb*2:.2f} GB.")
    print(f"Number of buffers: {total_buffers:,}.")
    print(f"Size of buffers: {total_buffers_gb:.2f} GB.")

def calculate_model_peak_memory(model_config: dict, optimizer_cat: str, batch_size: int, dtype: torch.dtype, ):
    '''
    Decompose the question based on parameters, activations, gradients, optimizer state
    '''
    # Parse model hyperparameters
    vocab_size     = model_config["vocab_size"]
    context_length = model_config["context_length"]
    num_layers     = model_config["num_layers"]
    d_model        = model_config["d_model"]
    d_ff           = model_config["d_ff"]
    num_heads      = model_config["num_heads"]

    # Other sizes
    element_size = torch.tensor([0], dtype=dtype).element_size()
    if optimizer_cat == "adamw":
        opt_state_mult = 2

    ################################
    ######## Parameters ############
    ################################

    # Token embedding layer
    param_token_embed = vocab_size * d_model

    # Transformer blocks
    # RMSNorm
    param_rms_norm = d_model
    # QKV projections
    qkv = 3
    param_qkv_projections = qkv * d_model * d_model
    # output projection
    param_output_projection = d_model * d_model
    # FFN
    num_ffn_projs = 3
    param_ffn_projections = num_ffn_projs * d_model * d_ff
    num_trf_norms = 2
    param_trf_block = num_trf_norms * param_rms_norm + param_qkv_projections + param_output_projection + param_ffn_projections

    # Final RMSNorm
    param_final_norm = param_rms_norm

    # Final output embedding
    param_final_embed = d_model * vocab_size

    total_params = param_token_embed + num_layers * param_trf_block + param_final_norm + param_final_embed

    ################################
    ######## Activations ###########
    ################################

    # Token embedding layer
    act_token_embed = batch_size * context_length * d_model

    # Transformer blocks
    # RMSNorm
    act_rms_norm = batch_size * context_length * d_model
    # QKV projections
    qkv = 3
    act_qkv_projections = qkv * batch_size * context_length * d_model
    # QKt
    act_qtk = batch_size * num_heads * context_length * context_length
    # softmax
    act_softmax = batch_size * num_heads  * context_length * context_length
    # attn_score * values
    act_weighted_sum_values = batch_size * context_length * d_model
    # output projection
    act_output_projection = batch_size * context_length * d_model
    # FFN
    # W1 linear
    act_ffn_linear1 = batch_size * context_length * d_ff
    # SiLU
    act_ffn_silu = batch_size * context_length * d_ff
    # W2 linear
    act_ffn_linear2 = batch_size * context_length * d_model
    num_trf_norms = 2
    act_trf_block = num_trf_norms * act_rms_norm + \
        act_qkv_projections + \
        act_qtk + \
        act_softmax + \
        act_weighted_sum_values + \
        act_output_projection + \
        act_ffn_linear1 + \
        act_ffn_silu + \
        act_ffn_linear2

    # Final RMSNorm
    act_final_norm = act_rms_norm

    # Final output embedding
    act_final_embed = batch_size * context_length * vocab_size

    # Cross entropy (loss): only consider the activation of softmax
    act_cross_entropy = batch_size * context_length * vocab_size

    total_activations = act_token_embed + \
        num_layers * act_trf_block + \
        act_final_norm + \
        act_final_embed + \
        act_cross_entropy

    ################################
    ######## Gradients #############
    ################################

    total_grads = total_params

    ################################
    ######## Optimizer states ######
    ################################

    total_opt_states = total_params * opt_state_mult

    param_gb = total_params * element_size / (2.0 ** 30)
    act_gb = total_activations * element_size / (2.0 ** 30)
    grad_gb = total_grads * element_size / (2.0 ** 30)
    opt_state_gb = total_opt_states * element_size / (2.0 ** 30)
    peak_memory_gb = param_gb + act_gb + grad_gb + opt_state_gb
    print(f"Size of parameters is {param_gb:.2f} GB")
    print(f"Size of activations is {act_gb:.2f} GB")
    print(f"Size of grads is {grad_gb:.2f} GB")
    print(f"Size of optimizer state is {opt_state_gb:.2f} GB")
    print(f"Estimated peak memory of the model is {peak_memory_gb:.2f} GB")

    act_gb_per_batch = act_gb / batch_size
    others_gb = param_gb + grad_gb + opt_state_gb

    return act_gb_per_batch, others_gb

def calculate_max_batch(model_config: dict, max_memory: int, maximum_batch_size: int = 16):
    activation_size_per_batch, others_size = calculate_model_peak_memory(
        model_config,
        "adamw",
        batch_size=1,
        dtype=torch.float32
    )
    
    batch_size = maximum_batch_size
    while batch_size * activation_size_per_batch + others_size >= max_memory:
        batch_size -= 1

    if batch_size <= 0: print("Memory size too small!!!")
    return batch_size

def calculate_model_parameters(model_config: dict):
    # Parse model hyperparameters
    vocab_size     = model_config["vocab_size"]
    context_length = model_config["context_length"]
    num_layers     = model_config["num_layers"]
    d_model        = model_config["d_model"]
    d_ff           = model_config["d_ff"]
    num_heads      = model_config["num_heads"]

    # Token embedding layer
    param_token_embed = vocab_size * d_model

    # Transformer blocks
    # RMSNorm
    param_rms_norm = d_model
    # QKV projections
    qkv = 3
    param_qkv_projections = qkv * d_model * d_model
    # output projection
    param_output_projection = d_model * d_model
    # FFN
    num_ffn_projs = 3
    param_ffn_projections = num_ffn_projs * d_model * d_ff
    num_trf_norms = 2
    param_trf_block = num_trf_norms * param_rms_norm + param_qkv_projections + param_output_projection + param_ffn_projections

    # Final RMSNorm
    param_final_norm = param_rms_norm

    # Final output embedding
    param_final_embed = d_model * vocab_size

    total_params = param_token_embed + num_layers * param_trf_block + param_final_norm + param_final_embed

    return total_params

def calculate_optimizer_flops_step(model_config: dict):
    flops_per_param = 0
    # Update first moment
    flops_per_param += 3
    # Update second moment
    flops_per_param += 4
    # Update parameters (grad descend)
    flops_per_param += 5
    # weight decay
    flops_per_param += 2

    model_paramters = calculate_model_parameters(model_config)
    return flops_per_param * model_paramters

def calculate_model_foward_flops(model_config: dict, batch_size):
    # Parse model hyperparameters
    vocab_size     = model_config["vocab_size"]
    context_length = model_config["context_length"]
    num_layers     = model_config["num_layers"]
    d_model        = model_config["d_model"]
    d_ff           = model_config["d_ff"]
    num_heads      = model_config["num_heads"]

    # --- MHA ---
    flops_mha_linear = 2 * context_length * d_model * d_model
    flops_mha_attn_weight = 2 * context_length * context_length * d_model
    flops_mha_context_vector = 2 * context_length * context_length * d_model
    flops_mha_output_linear = 2 * context_length * d_model * d_model

    qkv = 3
    flops_mha = qkv * flops_mha_linear + flops_mha_attn_weight + flops_mha_context_vector + flops_mha_output_linear

    # --- FFN ---
    flops_ffn_linear1 = 2 * context_length * d_model * d_ff
    flops_ffn_linear2 = 2 * context_length * d_model * d_ff
    flops_ffn_linear3 = 2 * context_length * d_model * d_ff

    ffn_linears = 3
    flops_ffn = ffn_linears * flops_ffn_linear1

    # --- Transformer block(s) ---
    flops_transformer_block = flops_mha + flops_ffn
    flops_trf_blocks = num_layers * flops_transformer_block

    # --- Final linear projection ---
    flops_final_linear = 2 * context_length * d_model * vocab_size

    # --- Total FLOPs per batch ---
    total_flops_batch = flops_trf_blocks + flops_final_linear

    total_flops = total_flops_batch * batch_size
    return total_flops


def estimate_train_days(model_config: dict,
                        hardware_flops_tera: float,
                        mfu: float,
                        train_steps: int,
                        batch_size: int):
    foward_flops_step = calculate_model_foward_flops(model_config, batch_size)
    # assue backward flops is 2* forward flops
    backward_flops_step = 2 * foward_flops_step
    opt_flops_step = calculate_optimizer_flops_step(model_config)
    total_flops_step = foward_flops_step + backward_flops_step + opt_flops_step

    hardware_flops = hardware_flops_tera * 1e12 * mfu
    train_days = total_flops_step * train_steps / hardware_flops / (60 * 60 * 24)

    # print(f"Estimated training days is {train_days} days.")
    return train_days