import math
import torch
import numpy as np

from collections.abc import Iterable

def get_lr_cosine_schedule(
    t: int,
    lr_max: float,
    lr_min: float,
    t_warmup: int,
    t_cosine_cycle: int,
) -> float:
    assert 0 <= t and 0 < t_warmup <= t_cosine_cycle
    if t <= t_warmup:
        return (t / t_warmup) * lr_max
    if t >= t_cosine_cycle:
        return lr_min
    rad = (t - t_warmup) / (t_cosine_cycle - t_warmup) * math.pi
    lr = lr_min + 0.5 * (1 + math.cos(rad)) * (lr_max - lr_min)
    return lr

def gradient_clipping(
    params: Iterable[torch.nn.Parameter],
    max_l2_norm: float,
    eps: float = 1e-6
) -> None:
    grads = [p.grad for p in params if p.grad is not None]
    if not grads:
        return

    l2_norm = torch.stack([(g ** 2).sum() for g in grads]).sum().sqrt()
    if l2_norm > max_l2_norm:
        scale = max_l2_norm / (l2_norm + eps)
        for grad in grads:
            grad.mul_(scale)

def get_batch(dataset: np.array, batch_size: int, context_length: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    # generate random start, use index to get slice from data aray
    # pay attention to dimensions
    max_start = len(dataset) - context_length
    assert max_start >= 0
    start_i = np.random.randint(low=0, high=max_start, size=batch_size)
    offsets = np.arange(start=0, stop=context_length)
    x_i = start_i.reshape(-1, 1) + offsets.reshape(1, -1)  # broadcast
    y_i = x_i + 1
    x = torch.as_tensor(data=dataset[x_i], dtype=torch.int64, device=device)
    y = torch.as_tensor(data=dataset[y_i], dtype=torch.int64, device=device)
    return x, y

