from collections.abc import Callable
import torch
import math

class SGD(torch.optim.Optimizer):

    def __init__(self, params, lr=1e-3):
        if lr < 0:
            raise ValueError(f'negative lr: {lr}')
        defaults = {'lr': lr}
        super().__init__(params, defaults)

    def step(self, closure: Callable | None = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            # hyper param is shared at group level, get it here
            lr = group['lr']
            for p in group['params']:
                # get grad
                if p.grad is None:
                    continue
                # calculate decay
                state = self.state[p]
                t = state.get('t', 0)
                decay = 1. / math.sqrt(t + 1)
                # update param
                grad = p.grad.data
                p.data -= lr * decay * grad
                # update state
                state['t'] = t + 1
        return loss


class AdamW(torch.optim.Optimizer):

    def __init__(
        self,
        params,
        lr: float = 1e-3,  # learning rate
        betas: tuple[float, float] = (0.9, 0.999),  # from Adam
        eps: float = 1e-8,  # from Adam/RMSProp
        weight_decay: float = 1e-2  # from AdamW
    ):
        defaults = {
            'lr': lr,
            'betas': betas,
            'eps': eps,
            'weight_decay': weight_decay
        }
        super().__init__(params, defaults)

    def step(self, closure: Callable | None = None):
        loss = None if not closure else closure()
        for group in self.param_groups:
            lr = group['lr']
            beta_1, beta_2 = group['betas']
            eps = group['eps']
            weight_decay = group['weight_decay']
            for p in group['params']:
                if p.grad is None:
                    continue
                # apply weight decay
                p.data -= lr * weight_decay * p.data
                # apply Adam
                grad = p.grad.data
                state = self.state[p]
                # calculate m
                m_prev = state.get('m', 0)
                m = beta_1 * m_prev + (1 - beta_1) * grad
                state['m'] = m
                # calculate v
                v_prev = state.get('v', 0)
                v = beta_2 * v_prev + (1 - beta_2) * grad ** 2
                state['v'] = v
                # adjust learning rate based on iteration t
                # (b/c at the beginning the accumulation of m/v is limited)
                t = state.get('t', 1)
                lr = lr / (1 - beta_1 ** t)
                lr = lr * math.sqrt(1 - beta_2 ** t)
                state['t'] = t + 1
                # update param
                p.data -= lr * m / (v.sqrt() + eps)
        return loss


# test

def test(opt_cls=SGD, lr=1, epoch=100):
    '''
    for this specific toy example, set lr = 50 will let the loss go to
    0 directly. for lr > 100, the initial gradient's abs value > 1 so
    it does not converge until the decay is big enough. For a too large
    lr like 200, the loss will go to inf before it converges.
    '''
    w = torch.nn.Parameter(5 * torch.randn((10, 10)))
    optimizer = opt_cls([w], lr)

    init_loss = (w ** 2).mean().cpu().item()
    for t in range(epoch):
        optimizer.zero_grad()
        loss = (w ** 2).mean()  # MSE to 0
        # print(loss.cpu().item())
        loss.backward()
        optimizer.step()

    final_loss = (w ** 2).mean().cpu().item()
    return init_loss, final_loss

# for lr in range(-3, 4):
#     print(lr)
#     print(test(lr=math.pow(10, lr), epoch=100))

# print(test(opt_cls=AdamW, lr=1, epoch=100))