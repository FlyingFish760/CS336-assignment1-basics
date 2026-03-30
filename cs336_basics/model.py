import math
from typing import Tuple, List

import torch
import torch.nn as nn
from torch import Tensor, LongTensor

from jaxtyping import Bool, Float, Int
from einops import rearrange, reduce, repeat, einsum

# Local imports
from cs336_basics.nn_utils import softmax

class Linear(nn.Module):
    def __init__(self, in_features:int, out_features:int, device:torch.device=None, dtype:torch.dtype=None):
        '''
        in_features: int final dimension of the input
        out_features:int final dimension of the output
        device: torch.device |None=None Device to store the parameters on
        dtype: torch.dtype| None=None Data type of the parameters
        '''
        super().__init__()

        # Init the linear projection weight
        w = torch.empty((out_features, in_features), device=device, dtype=dtype)
        delta = math.sqrt(2 / (in_features + out_features))
        nn.init.trunc_normal_(w, mean=0, std=delta, a=-3 * delta, b=3 * delta)

        self.W = nn.Parameter(data=w)
        
    def forward(self, x: Float[Tensor, "... in_features"]) -> torch.tensor:
        return x @ self.W.T

class Embedding(nn.Module):
    def __init__(self, 
                 num_embeddings: int, 
                 embedding_dim: int, 
                 device: torch.device = None, 
                 dtype: torch.dtype = None):
        '''
        num_embeddings: int Size of the vocabulary
        embedding_dim: int Dimension of the embedding vectors, i.e., d_model
        device: torch.device | None = None Device to store the parameters on
        dtype: torch.dtype | None = None Data type of the parameters
        '''
        super().__init__()

        # Init an embedding matrix
        embed_mat = torch.empty((num_embeddings, embedding_dim), device=device, dtype=dtype)
        torch.nn.init.trunc_normal_(embed_mat, mean=0, std=1)
        
        self.embed_mat = nn.Parameter(embed_mat)   

    def forward(self, x: Int[LongTensor, "... seq_len"]) -> Float[Tensor, "... seq_len embedding_dim"]:
        # b = x.shape[0]
        # x_flatten = rearrange(x, "... seq_len -> (... seq_len)")
        # x_embedded = rearrange(torch.index_select(self.embed_mat, dim=0, index=x_flatten),
        #                        "(b seq_len) embedding_dim -> b seq_len embedding_dim", b=b)

        x_embedded = self.embed_mat[x]   # (num_embeddings embedding_dim) -> (... seq_len embedding_dim)
        return x_embedded   

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float=1e-5, device: torch.device=None, dtype: torch.dtype=None):
        '''
        d_model: int Hidden dimension of the model
        eps: float = 1e-5 Epsilon value for numerical stability
        device: torch.device | None = None Device to store the parameters on
        dtype: torch.dtype | None = None Data type of the parameter
        '''
        super().__init__()
        self.eps = eps
        self.gain = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def forward(self, x: Float[Tensor, "b seq_len d_model"]) -> Float[Tensor, "b seq_len d_model"]:
        in_dtype = x.dtype
        x = x.to(torch.float32)

        x_rms = torch.sqrt(torch.mean(torch.square(x), dim=-1, keepdim=True) + self.eps) 
        x_norm = torch.div(x, x_rms) * self.gain

        x_norm = x_norm.to(in_dtype)
        return x_norm

class FFN(nn.Module):
    def __init__(self, d_model: int, d_hidden: int, device: torch.device=None, dtype: torch.dtype=None):
        super().__init__()
        self.W1 = Linear(d_model, d_hidden, device, dtype)
        self.W3 = Linear(d_model, d_hidden, device, dtype)
        self.W2 = Linear(d_hidden, d_model, device, dtype)

    def silu(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)

    def forward(self, x: Float[Tensor, "... d_model"]) -> Float[Tensor, "... d_model"]:
        x_swiglu = self.silu(self.W1(x)) * self.W3(x)
        x_out = self.W2(x_swiglu)
        return x_out
    
class RoPE(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device: torch.device=None):
        '''
        theta:float base Θ value for the RoPE
        d_k:int   dimension of query and key vectors
        max_seq_len: int    Maximum sequence length that will be inputted
        device: torch.device |None=None Device to store the buffer on
        '''
        super().__init__()

        # Pre-compute buffers for sin and cos values
        positions = torch.arange(max_seq_len, device=device)   # (max_seq_len, 1)
        freqs = 1 / theta ** (torch.arange(0, d_k, 2, device=device)[: d_k // 2].float() / d_k)   # (d_k/2, 1)
        angles = torch.outer(positions, freqs)   # (max_seq_len, d_k/2)
        sin_angles = repeat(torch.sin(angles), "max_seq_len d -> max_seq_len (d repeat)", repeat = 2)   # the last dimension: [sinΘ1, sinΘ1, sinΘ2, sinΘ2, ...]
        cos_angles = repeat(torch.cos(angles), "max_seq_len d -> max_seq_len (d repeat)", repeat = 2)   # the last dimension: [cosΘ1, cosΘ1, cosΘ2, cosΘ2, ...]
        self.register_buffer("sin", sin_angles)
        self.register_buffer("cos", cos_angles)

    def forward(self, x: Float[Tensor, "... seq_len d_k"], token_positions: Int[Tensor, "... seq_len"]) -> Float[Tensor, "... seq_len d_k"]:
        '''
        Process an input tensor of shape (..., seq_len, d_k) and return a tensor of the same shape. 
        The function should tolerate x with an arbitrary number of batch dimensions, and you may assume
        that the token positions are given as a tensor of shape (..., seq_len) specifying the positions 
        of x along the sequence dimension.
        '''
        sin = self.sin[token_positions]   # Automatically broadcast (max_seq_len, d_k) -> (... seq_len d_k)
        cos = self.cos[token_positions]   # Automatically broadcast (max_seq_len, d_k) -> (... seq_len d_k)

        # x1 = x[..., 0::2]    # (... seq_len d_k/2)
        # x2 = x[..., 1::2]    # (... seq_len d_k/2)
        # x_rotated = cos * torch.concat((x1, x2), dim=-1) + sin * torch.concat((-x2, x1), dim=-1)

        # To construct x_transformed as [-x2, x1, -x4, x3, ...]
        x_transformed = torch.zeros_like(x)
        x_transformed[..., 0::2] = -x[..., 1::2]
        x_transformed[..., 1::2] = x[..., 0::2]

        x_rotated = cos * x + sin * x_transformed
        return x_rotated

def attention(Q: Float[Tensor, "b ... q_len d_k"], 
              K: Float[Tensor, "b ... k/v_len d_k"],
              V: Float[Tensor, "b ... k/v_len d_v"],
              mask: Bool[Tensor, "q_len k/v_len"] | None = None) -> Float[Tensor, "b ... q_len d_v"]:
    # Compute scaled attention scores
    attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / Q.shape[-1] ** 0.5   # (b ... q_len k/v_len)

    # Apply mask
    if mask is not None:
        attn_scores.masked_fill_(~mask, -torch.inf)

    # Apply softmax to get attention weights
    attn_weights = softmax(attn_scores, dim=-1)   # # (b ... q_len k/v_len)

    # Compute the context vector
    output = torch.matmul(attn_weights, V)   # (b ... q_len d_v)

    return output

class MultiheadAttention(nn.Module):
    def __init__(self, 
                 d_model: int, 
                 num_heads: int,
                 max_seq_len: int,
                 theta: float = 10000.0):
        '''
        d_model: int Dimensionality of the Transformer block inputs.
        num_heads: int Number of heads to use in multi-head self-attention.
        '''
        super().__init__()

        self.rope = RoPE(
            theta=theta,
            d_k=d_model // num_heads,
            max_seq_len=max_seq_len
        )

        self.num_heads = num_heads
        # self.W_qkv = Linear(d_model, d_model*3)
        self.W_q = Linear(d_model, d_model)
        self.W_k = Linear(d_model, d_model)
        self.W_v = Linear(d_model, d_model)
        self.W_o = Linear(d_model, d_model)
        
    def forward(self, 
                x: Float[Tensor, "... seq_len d_model"], 
                token_positions: Int[Tensor, " ... sequence_length"] | None = None) -> Float[Tensor, "... seq_len d_model"]:
        # Construct casual mask
        seq_len = x.shape[-2]
        mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1)==0

        # Compute attention
        # (use W_qkv)
        # qkv = self.W_qkv(x)   # (... seq_len, d_model*3)
        # Q, K, V = rearrange(qkv, "... seq_len (qkv d_model) -> qkv ... seq_len d_model", qkv=3)   # (... seq_len d_model)
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)
        # Split Q, K, V into heads
        Q = rearrange(Q, "... seq_len (h d_k) -> ... h seq_len d_k", h=self.num_heads)
        K = rearrange(K, "... seq_len (h d_k) -> ... h seq_len d_k", h=self.num_heads)
        V = rearrange(V, "... seq_len (h d_v) -> ... h seq_len d_v", h=self.num_heads)

        # # Only apply RoPE to Q/K
        # if token_positions is None: 
        #     seq_len = x.shape[-2]
        #     token_positions = torch.arange(seq_len)
        #     token_positions = token_positions.expand(*x.shape[:-2], seq_len)
        # Q = self.rope(Q, token_positions)   
        # K = self.rope(K, token_positions)

        # Only apply RoPE to Q/K
        if token_positions is not None: 
            token_positions = repeat(token_positions, "... seq_len -> ... h seq_len", h=self.num_heads)
            Q = self.rope(Q, token_positions)   
            K = self.rope(K, token_positions)
            
        # Compute Multihead-attention
        heads = attention(Q, K, V, mask)   # (... h seq_len d_k)
        multi_heads = rearrange(heads, "... h seq_len d_k -> ... seq_len (h d_k)")
        output = self.W_o(multi_heads)    # ( ... seq_len d_model)

        return output
    
class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, max_seq_len, theta, use_LN=True):
        '''
        d_model: int Dimensionality of the Transformer block inputs.
        num_heads: int Number of heads to use in multi-head self-attention.
        d_ff: int Dimensionality of the position-wise feed-forward inner layer.
        '''
        super().__init__()
        if use_LN:
            self.attn_norm = RMSNorm(d_model)
            self.ff_norm = RMSNorm(d_model)
        self.attn = MultiheadAttention(
            d_model, 
            num_heads,
            max_seq_len,
            theta
        )
        self.ff = FFN(
            d_model,
            d_ff
        )

        self.use_LN = use_LN

    def forward(self, 
                x: Float[Tensor, "... seq_len d_model"]) -> Float[Tensor, "... seq_len d_model"]:
        # First sublayer
        seq_len = x.shape[-2]
        token_positions = torch.arange(seq_len)
        token_positions = token_positions.expand(*x.shape[:-2], seq_len)

        x_res = x
        x_attn = self.attn(self.attn_norm(x), token_positions) if self.use_LN else self.attn(x, token_positions)
        x = x_res + x_attn

        # Second sublayer
        x_res = x
        x_ff = self.ff(self.ff_norm(x)) if self.use_LN else self.ff(x)
        x = x_res + x_ff

        return x

class TransformerLM(nn.Module):
    def __init__(self, 
                 vocab_size: int,
                 d_model: int,
                 num_heads: int, 
                 d_ff: int, 
                 context_length: int, 
                 theta: float,
                 num_layers: int,
                 use_LN: bool = True):
        super().__init__()
        '''
        vocab_size: int — The size of the vocabulary, necessary for determining the dimensionality of
          the token embedding matrix.
        context_length: int — The maximum context length, necessary for determining the dimensionality
          of the position embedding matrix.
        num_layers: int — The number of Transformer blocks to use.
        '''
        # Experiment config indication
        if use_LN:
            print("-----------The model is initialized with LayerNorm-----------")
        else:
            print("-----------The model is NOT initialized with LayerNorm!!!-----------")

        # Token embeddding
        self.token_embedding = Embedding(vocab_size, d_model)

        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([TransformerBlock(
            d_model,
            num_heads,
            d_ff,
            context_length,
            theta,
            use_LN = use_LN
        ) for _ in range(num_layers)])

        # Output layer
        self.out_norm = RMSNorm(d_model)
        self.out_proj = Linear(d_model, vocab_size)

    def forward(self, x: Float[Tensor, "... seq_len"]) -> Float[Tensor, "... seq_len vocab_size"]:
        h = self.token_embedding(x)   # (... seq_len d_model)
        for layer in self.transformer_blocks:
            h = layer(h)
        logits = self.out_proj(self.out_norm(h))
        return logits
    


###################### FullAttnRes ######################
def attn_res(query: nn.Parameter, keys: list[Tensor], norm: RMSNorm) -> Tensor:
    '''
    Params:
        query: layer-specific learnable vector. Tensor of size [D]
        keys: list of outputs of previous layers. N tensors of size [B, T, D]

    Returns:
        hidden_state: Tensor of size [B, T, D]
    '''

    keys = torch.stack(keys, dim=0)    # [N, B, T, D]
    values = keys   # [N, B, T, D]
    
    keys_normed = norm(keys)
    attn_scores = einsum(query, keys_normed, "d, n b t d -> n b t")
    attn_weights = torch.softmax(attn_scores, dim=0)
    hidden_state = einsum(attn_weights, values, "n b t, n b t d -> b t d")
    return hidden_state

class TrfBlock_FullAttnRes(TransformerBlock):
    def __init__(self, 
                 d_model, 
                 num_heads, 
                 d_ff, 
                 max_seq_len, 
                 theta, 
                 use_LN=True) -> None:
        super().__init__(d_model, num_heads, d_ff, max_seq_len, theta, use_LN)

        self.query_attn = nn.Parameter(torch.zeros((d_model)))
        self.attn_res_norm1 = RMSNorm(d_model)

        self.query_mlp = nn.Parameter(torch.zeros((d_model)))
        self.attn_res_norm2 = RMSNorm(d_model)

    def forward(self, layer_outputs: list[Tensor]) -> list[Tensor]:
        '''
        Params:
            layer_outputs: outputs of previous layers

        Returns:
            layer_outputs: layer outputs with addition to those of this transformer block
        '''
        # First layer (Attention)
        h1 = attn_res(
            query=self.query_attn,
            keys=layer_outputs,
            norm=self.attn_res_norm1
        )   # (B T D)

        seq_len = h1.shape[-2]
        token_positions = torch.arange(seq_len)
        token_positions = token_positions.expand(*h1.shape[:-2], seq_len)

        output1 = self.attn(self.attn_norm(h1), token_positions)
        layer_outputs.append(output1)

        # Second layer (MLP)
        h2 = attn_res(
            query=self.query_mlp,
            keys=layer_outputs,
            norm=self.attn_res_norm2
        )
        output2 = self.ff(self.ff_norm(h2))
        layer_outputs.append(output2)

        return layer_outputs

class TransformerLM_fullAttnRes(nn.Module):
    def __init__(self, 
                 d_model: int,
                 d_ff: int, 
                 num_heads: int, 
                 num_layers: int,
                 context_length: int, 
                 theta: float,
                 vocab_size: int):
        super().__init__()
        '''
        vocab_size: int — The size of the vocabulary, necessary for determining the dimensionality of
          the token embedding matrix.
        context_length: int — The maximum context length, necessary for determining the dimensionality
          of the position embedding matrix.
        num_layers: int — The number of Transformer blocks to use.
        '''
        # Token embeddding
        self.token_embedding = Embedding(vocab_size, d_model)

        # Transformer blocks using full attention residual
        self.transformer_blocks = nn.ModuleList([TrfBlock_FullAttnRes(
            d_model,
            num_heads,
            d_ff, 
            context_length,
            theta
        ) for _ in range(num_layers)])

        # Output layer
        self.out_q = nn.Parameter(torch.zeros(d_model))
        self.out_attn_res_norm = RMSNorm(d_model)
        self.out_norm = RMSNorm(d_model)
        self.out_proj = Linear(d_model, vocab_size)

    def forward(self, token_ids: Int[Tensor, "... seq_len"]) -> Float[Tensor, "... seq_len vocab_size"]:
        # Token embedding layer
        h1 = self.token_embedding(token_ids)   # (... seq_len d_model)

        # Transformer layers with full attn res 
        layer_outputs = [h1]
        for layer in self.transformer_blocks:
            layer_outputs = layer(layer_outputs)
        
        # Output layer
        h_final = attn_res(self.out_q, layer_outputs, self.out_attn_res_norm)
        logits = self.out_proj(self.out_norm(h_final))
        return logits

###################### BlockAttnRes ######################
class TrfBlock_BlockAttnRes(nn.Module):
    def __init__(self, 
                 d_model: int, 
                 d_ff: int, 
                 num_heads: int, 
                 max_seq_len: int, 
                 theta: float, 
                 layer_num: int) -> None:
        super().__init__()
        self.layer_num = layer_num

        self.query_attn = nn.Parameter(torch.zeros(d_model))
        self.attn_res_norm1 = RMSNorm(d_model)
        self.attn_norm = RMSNorm(d_model)
        self.attn = MultiheadAttention(
            d_model,
            num_heads,
            max_seq_len,
            theta
        )
        
        self.query_mlp = nn.Parameter(torch.zeros(d_model))
        self.attn_res_norm2 = RMSNorm(d_model)
        self.mlp_norm = RMSNorm(d_model)
        self.mlp = FFN(
            d_model,
            d_ff
        )
    
    def forward(self, 
                block_keys: List[Tensor],
                partial_sum: Tensor | None,
                block_size: int) -> Tuple[List[Tensor], Tensor]:
        # First sub-layer
        # Compute hidden state
        if partial_sum is not None:
            keys = block_keys + [partial_sum]
        else:
            keys = block_keys
        h1 = attn_res(self.query_attn, keys, self.attn_res_norm1)

        # Derive token_positions tensor (for masking)
        seq_len = h1.shape[-2]
        token_positions = torch.arange(seq_len)
        token_positions = token_positions.expand(*h1.shape[:-2], seq_len)
        
        # Attention layer
        output1 = self.attn(self.attn_norm(h1), token_positions)
        partial_sum = partial_sum + output1 if partial_sum is not None else output1

        # Second sub-layer
        keys = block_keys + [partial_sum]
        h2 = attn_res(self.query_mlp, keys, self.attn_res_norm2)

        output2 = self.mlp(self.mlp_norm(h2))
        partial_sum = partial_sum + output2

        # If reaches block boundary, start new block
        # Add the partial_sum to block_keys; Reset partial sum to None
        if self.layer_num % (block_size // 2) == 0:
            block_keys.append(partial_sum)
            partial_sum = None

        return block_keys, partial_sum
    
class TransformerLM_BlockAttnRes(nn.Module):
    def __init__(self, 
                 d_model: int,
                 d_ff: int, 
                 num_heads: int, 
                 num_layers: int,
                 context_length: int, 
                 theta: float,
                 vocab_size: int,
                 block_size: int) -> None:
        '''
        num_layers: Number of TRANSFORMER layers
        block_size: block_size counts ATTN +MLP; each transformer layer has 2. MAKE SURE: block_size
                    is even. Better to make num_layers % (block_size // 2) == 0
        '''
        super().__init__()
        assert block_size > 0, "block_size must be greater than zero"
        assert block_size % 2 == 0, "block_size count ATTN +MLP layers and therefore must be even"

        # Token embeddding
        self.token_embedding = Embedding(vocab_size, d_model)

        # Transformer blocks using block attention residual
        self.transformer_blocks = nn.ModuleList([TrfBlock_BlockAttnRes(
            d_model,
            d_ff,
            num_heads,
            context_length,
            theta,
            layer_num
        ) for layer_num in range(1, num_layers + 1)])

        # Output layer
        self.out_query = nn.Parameter(torch.zeros(d_model))
        self.out_attn_res_norm = RMSNorm(d_model)
        self.out_norm = RMSNorm(d_model)
        self.out_proj = Linear(d_model, vocab_size)

        self.block_size = block_size

    def forward(self, token_ids: Float[Tensor, "... seq_len"]) -> Float[Tensor, "... seq_len vocab_size"]:
        # Token embedding layer
        h1 = self.token_embedding(token_ids)   # (... seq_len d_model)

        # Transformer layers with block attn res 
        block_outputs, partial_sum = [h1], None
        for layer in self.transformer_blocks:
            block_outputs, partial_sum = layer(block_outputs, partial_sum, self.block_size)
        
        # Output layer
        keys_final = block_outputs if partial_sum is None else block_outputs + [partial_sum]
        h_final = attn_res(self.out_query, keys_final, self.out_attn_res_norm)
        logits = self.out_proj(self.out_norm(h_final))
        return logits


if __name__ == "__main__":
    b, seq_len, d_model, vocab_size, num_heads, d_ff, num_layers = 4, 16, 512, 5120, 4, 1344, 4

    # token_ids = torch.randint(0, vocab_size, (b, seq_len))
    # embed = Embedding(vocab_size, d_model)
    # token_embedding = embed(token_ids)
    # print(token_embedding.shape)

    # x = torch.randn((b, seq_len, d_model))

    # ffn = FFN(512, 64)
    # x_ffn = ffn(x)
    # print("x_ffn.shape", x_ffn.shape)

    # rope = RoPE(theta=10000, d_k=d_model, max_seq_len=512)
    # x_rotated = rope(x, torch.arange(128, 144))
    # print(x_rotated.shape)
    # x_sm = softmax(x, dim=-1)
    # print(x_sm.shape)

    # Q, K, V = torch.randn((b, seq_len, d_model)), torch.randn((b, seq_len*2, d_model)), torch.randn((b, seq_len*2, d_model//2))
    # mask = torch.randint(low=-10, high=10, size=(seq_len, seq_len*2)) > 0
    # attn = attention(Q, K, V, mask)
    # print(attn.shape)

    # mha = MultiheadAttention(
    #     d_model,
    #     num_heads=4, 
    #     max_seq_len=128,
    #     theta=10000,
    # )
    # print(mha.parameters())
    # res = mha(x)
    # print(res.shape)

    # trans = Transformer(
    #     d_model,
    #     num_heads=4,
    #     d_ff= 4* d_model,
    #     max_seq_len=128
    # )
    # res = trans(x)
    # print(res.shape)

    # vocab_size = 512
    # token_ids = torch.randint(low=0, high=vocab_size, size=(seq_len,))
    # lm = TransformerLM(
    #     vocab_size,
    #     d_model,
    #     num_heads=4,
    #     d_ff= 4*d_model,
    #     context_length=128,
    #     theta=10000,
    #     num_layers=2
    # )


    # output = lm(token_ids)
    # print(output.shape)
    # print(output)
    # print(lm.state_dict)


    #----------------- Test FullAttnRes

    # Test attn_res
    # query = torch.randn((d_model))
    # keys = [torch.randn((b, seq_len, d_model)) for _ in range(num_layers)]
    # norm = RMSNorm(d_model)
    # res = attn_res(query, keys, norm)
    # print(res.shape)

    # Test TrfBlock_FullAttnRes
    # trf_block_attn_res = TrfBlock_FullAttnRes(
    #     d_model, 
    #     num_heads,
    #     d_ff,
    #     seq_len,
    #     theta=10000
    # )

    # layer_outputs = [torch.randn((b, seq_len, d_model)) for _ in range(num_layers)]
    # new_layer_outputs = trf_block_attn_res(
    #     layer_outputs
    # )
    # assert len(new_layer_outputs) == num_layers + 2
    # assert new_layer_outputs[0].shape == torch.Size([b, seq_len, d_model])

    # Test TransformerLM_fullAttnRes
    # lm = TransformerLM_fullAttnRes(
    #     d_model,
    #     d_ff,
    #     num_heads,
    #     num_layers,
    #     seq_len,
    #     theta=10000,
    #     vocab_size=vocab_size
    # )

    # token_ids = torch.randint(0, vocab_size, (b, seq_len))
    # logits = lm(token_ids)
    # assert logits.shape == torch.Size([b, seq_len, vocab_size])
    # print(logits)

    #----------------- Test BlockAttnRes
    # block_size = 4

    # Test TrfBlock_BlockAttnRes
    
    # # Reach block bound
    # trf_block_block_attn_res = TrfBlock_BlockAttnRes(
    #     d_model,
    #     d_ff,
    #     num_heads,
    #     seq_len,
    #     theta=10000,
    #     layer_num=4
    # )

    # block_keys = [torch.randn((b, seq_len, d_model)) for _ in range(3)]
    # partial_sum = torch.randn((b, seq_len, d_model))

    # new_block_keys, new_partial_sum = trf_block_block_attn_res(
    #     block_keys, partial_sum, block_size
    # )
    # print(len(new_block_keys))
    # print(new_block_keys[0].shape)
    # print(new_partial_sum)

    # # Not reach block bound
    # trf_block_block_attn_res = TrfBlock_BlockAttnRes(
    #     d_model,
    #     d_ff,
    #     num_heads,
    #     seq_len,
    #     theta=10000,
    #     layer_num=5
    # )

    # block_keys = [torch.randn((b, seq_len, d_model)) for _ in range(3)]
    # partial_sum = None

    # new_block_keys, new_partial_sum = trf_block_block_attn_res(
    #     block_keys, partial_sum, block_size
    # )
    # print(len(new_block_keys))
    # print(new_block_keys[0].shape)
    # print(new_partial_sum.shape)

    # Case: num_layers % (block_size // 2) != 0
    # For the attention res before the output layer, use the blocks of keys and the partial sum

    # Test TransformerLM_BlockAttnRes

    # Case1: num_layers % (block_size //2 ) == 0
    # num_layers = 4

    # lm = TransformerLM_BlockAttnRes(
    #     d_model,
    #     d_ff,
    #     num_heads,
    #     num_layers,
    #     seq_len,
    #     10000,
    #     vocab_size,
    #     block_size
    # )
    # token_ids = torch.randint(0, vocab_size, (b, seq_len))
    # logits = lm(token_ids)
    # print(logits.shape)
    
    # Case2: num_layers % (block_size //2 ) != 0
    # num_layers = 5

    # lm = TransformerLM_BlockAttnRes(
    #     d_model,
    #     d_ff,
    #     num_heads,
    #     num_layers,
    #     seq_len,
    #     10000,
    #     vocab_size,
    #     block_size
    # )
    # token_ids = torch.randint(0, vocab_size, (b, seq_len))
    # logits = lm(token_ids)
    # print(logits.shape)

