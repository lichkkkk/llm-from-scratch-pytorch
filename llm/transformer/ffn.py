import torch

from .linear import Linear
from .activation import silu

class SwiGLU(torch.nn.Module):

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None
    ):
        super().__init__()
        self.gate = Linear(in_features=d_model, out_features=d_ff, device=device, dtype=dtype)
        self.linear_1 = Linear(in_features=d_model, out_features=d_ff, device=device, dtype=dtype)
        self.linear_2 = Linear(in_features=d_ff, out_features=d_model, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        '..., d_model -> ..., d_model'
        '''
        return self.linear_2(silu(self.linear_1(x)) * self.gate(x))
