import torch

from einops import einsum, rearrange
from .linear import Linear

def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    '''
    '... seq_len d_model -> ... seq_len d_model'
    '''
    x_max = x.amax(dim=dim, keepdim=True)
    x_exp = torch.exp(x - x_max)
    x_softmax = x_exp / x_exp.sum(dim=dim, keepdim=True)
    return x_softmax

def scaled_dot_product_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.BoolTensor | None = None) -> torch.Tensor:
    '''
    'b ... i d_k, b ... j d_k, b ... j d_v, i j -> b ... i d_v'
    '''
    scale = q.shape[-1] ** -0.5
    qk = einsum(q, k, 'b ... i d_k, b ... j d_k -> b ... i j') * scale
    if mask is not None:
        qk = qk.masked_fill(~mask, float('-inf'))
    weights = softmax(qk, -1)
    out = einsum(weights, v, 'b ... i j, b ... j d_v -> b ... i d_v')
    return out

class MultiHeadSelfAttention(torch.nn.Module):

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        position_encoding: torch.nn.Module | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None
    ):
        super().__init__()
        self.num_heads = num_heads
        # only single Q/K/V is needed since n [d_model, d_k] is the same as one [d_model d_model]
        self.w_q = Linear(d_model, d_model, device=device, dtype=dtype)  # [(num_heads d_k) d_model]
        self.w_k = Linear(d_model, d_model, device=device, dtype=dtype)
        self.w_v = Linear(d_model, d_model, device=device, dtype=dtype)
        self.w_o = Linear(d_model, d_model, device=device, dtype=dtype)
        self.position_encoding = position_encoding

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        '''
        'b s d -> b s d'
        '''
        seq_len = x.shape[-2]
        q = rearrange(self.w_q(x), '... s (h d_k) -> ... h s d_k', h=self.num_heads)
        k = rearrange(self.w_k(x), '... s (h d_k) -> ... h s d_k', h=self.num_heads)
        if self.position_encoding:
            if token_positions is None:
                token_positions = torch.arange(seq_len, device=x.device)
            q = self.position_encoding(q, token_positions)       
            k = self.position_encoding(k, token_positions)
        v = rearrange(self.w_v(x), '... s (h d_v) -> ... h s d_v', h=self.num_heads)

        mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device))
        attn_out = scaled_dot_product_attention(q, k, v, mask)  # [... h s d_v]
        return self.w_o(rearrange(attn_out, '... h s d_v -> ... s (h d_v)'))
