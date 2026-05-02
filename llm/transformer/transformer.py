import torch

from .attention import MultiHeadSelfAttention
from .ffn import SwiGLU
from .norm import RMSNorm

class Transformer(torch.nn.Module):

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        position_encoding: torch.nn.Module | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None
    ):
        super().__init__()
        self.mha = MultiHeadSelfAttention(d_model, num_heads, position_encoding, device, dtype)
        self.ffn = SwiGLU(d_model, d_ff, device, dtype)
        self.layer_norm_1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.layer_norm_2 = RMSNorm(d_model, device=device, dtype=dtype)
        self.position_encoding = position_encoding


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x + self.mha(self.layer_norm_1(x))
        return y + self.ffn(self.layer_norm_2(y))
