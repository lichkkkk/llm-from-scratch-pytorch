from math import sqrt
import torch

from einops import einsum


class Linear(torch.nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.W = torch.nn.Parameter(
            torch.empty(out_features, in_features, device=device, dtype=dtype)
        )
        torch.nn.init.trunc_normal_(
            self.W,
            mean=0,
            std=sqrt(2 / (in_features + out_features)),
            a=-3,
            b=3,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # return x @ self.W.T
        return einsum(self.W, x, "d_out d_in, ... d_in -> ... d_out")
