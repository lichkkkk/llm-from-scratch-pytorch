import torch

class RoPE(torch.nn.Module):

    def __init__(
        self,
        theta: float,
        d_k: int,
        max_seq_len: int,
        device: torch.device | None = None
    ):
        super().__init__()
        # create sin_cached and cos_cached with shape [seq_len, d_k // 2]
        inv_freq_vec = 1.0 / torch.pow(theta, torch.arange(0, d_k // 2, device=device) * 2 / d_k)  # d_k // 2
        pos_vec = torch.arange(0, max_seq_len, device=device)   # max_seq_len
        inv_freq_by_pos_mat = torch.outer(pos_vec, inv_freq_vec)
        sin_cached = torch.sin(inv_freq_by_pos_mat)
        cos_cached = torch.cos(inv_freq_by_pos_mat)
        self.register_buffer('sin_cached', sin_cached, persistent=False)
        self.register_buffer('cos_cached', cos_cached, persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        '''
        '... seq_len d_model, ... seq_len -> ... seq_len d_model'
        '''
        # look up cached sin and cos, each pos get d_k // 2 unique sin/cos values
        sin_val = self.sin_cached[token_positions]
        cos_val = self.cos_cached[token_positions]

        # notes: it would be more efficient if we split it by 1st_half and 2nd_half
        # split x by odd and even
        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]

        # mul sin/cos and then interleave
        x_even_rotated = x_even * cos_val - x_odd * sin_val
        x_odd_rotated = x_odd * cos_val + x_even * sin_val
        x_out = torch.empty_like(x)
        x_out[..., 0::2] = x_even_rotated
        x_out[..., 1::2] = x_odd_rotated

        return x_out
