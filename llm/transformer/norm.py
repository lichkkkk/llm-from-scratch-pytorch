import torch

class RMSNorm(torch.nn.Module):

    def __init__(self,
        d_model: int,
        eps: float = 1e-5,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None
    ):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.W = torch.nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        '..., d_model -> ..., d_model'
        '''
        in_dtype = x.dtype
        x_fp32 = x.to(torch.float32)

        rms = torch.sqrt(x_fp32.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        y = x_fp32 * self.W / rms

        return y.to(in_dtype)
