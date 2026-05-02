import torch


def silu(x: torch.Tensor) -> torch.Tensor:
    '''
    '... d_model -> ... d_model'
    '''
    return x * torch.sigmoid(x)
