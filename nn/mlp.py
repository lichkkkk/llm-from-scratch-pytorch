'''
TODO: list different ways to implement those fundamental components
TODO: use this simple code to show training dynamics like optimizer, activation, batch size, norm, etc.
'''


import torch


# def relu(x: torch.Tensor) -> torch.Tensor:
#     return x.maximum(torch.zeros_like(x))


# def softmax(x: torch.Tensor) -> torch.Tensor:
#     '''
#     softmax_i = exp(xi) / sum(exp(xi))
#     '''
#     x = x - x.max(dim=-1, keepdim=True).values
#     exp = x.exp()
#     exp_sum = exp.sum(dim=-1, keepdim=True)
#     return exp / exp_sum


# def cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
#     '''
#     CE = - log prob (x)[target]

#     logits: (b, c)
#     targets: (b)
#     loss: (,)
#     '''
#     probs = softmax(logits)  # (b, c)
#     targets_probs = probs[torch.arange(logits.shape[0]), targets]  # (b,)
#     return -1 * torch.log(targets_probs + 1e-8).mean()


class MLPTorch(torch.nn.Module):

    def __init__(self, d_in: int, d_hidden: int, d_out: int):
        super().__init__()
        self.w1 = torch.nn.Parameter(torch.empty((d_in, d_hidden), dtype=torch.float))
        self.b1 = torch.nn.Parameter(torch.zeros(size=(d_hidden,), dtype=torch.float))  # should remove since we have BN
        self.bn_gamma = torch.nn.Parameter(torch.ones(size=(d_hidden,), dtype=torch.float))
        self.bn_beta = torch.nn.Parameter(torch.zeros(size=(d_hidden,), dtype=torch.float))
        self.w2 = torch.nn.Parameter(torch.empty((d_hidden, d_out), dtype=torch.float))
        self.b2 = torch.nn.Parameter(torch.zeros(size=(d_out,), dtype=torch.float))

        self.bn_running_momentum = 0.1
        self.register_buffer('bn_running_mean', torch.zeros(size=(1, d_hidden), dtype=torch.float))
        self.register_buffer('bn_running_var', torch.ones(size=(1, d_hidden), dtype=torch.float))

        torch.nn.init.normal_(self.w1, mean=0, std=0.02)
        torch.nn.init.normal_(self.w2, mean=0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2:
            raise ValueError('invalid tensor shape')

        # linear
        # b1 can be removed as we have batch norm mean
        x = x @ self.w1 + self.b1  # (b, d_hidden)

        # batch norm (training and eval are different)
        batch_size = x.shape[0]
        if self.training:
            if batch_size < 2:
                raise ValueError('batch norm requires batch size >= 2')
            x_mean = x.mean(dim=0, keepdim=True)  # (1, d)
            x_var = x.var(dim=0, unbiased=False, keepdim=True)  # (1, d)
            with torch.no_grad():
                x_unbiased_var = x_var * batch_size / (batch_size - 1)
                self.bn_running_mean.lerp_(x_mean, self.bn_running_momentum)
                self.bn_running_var.lerp_(x_unbiased_var, self.bn_running_momentum)
        else:
            x_mean = self.bn_running_mean # (1, d)
            x_var = self.bn_running_var # (1, d)
        # In float16, 1e-8 rounds to zero, so a zero-variance feature can produce NaNs.
        x = (x - x_mean) / torch.sqrt(x_var + 1e-8)  # (b, d)
        x = x * self.bn_gamma + self.bn_beta

        # activation
        x = x.maximum(torch.zeros_like(x))

        # linear
        logits = x @ self.w2 + self.b2

        return logits


    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            raise ValueError('needs to switch to eval mode for batch norm to work properly')
        logits = self(x)
        y = torch.argmax(logits, dim=-1)
        return y


def main():
    model = MLPTorch(d_in=2, d_hidden=8, d_out=2)

    ## training task -> XOR
    data_size = 1000
    train_ratio = 0.8
    train_split_pos = int(data_size * train_ratio)

    X = torch.randint(0, 2, size=(data_size, 2))
    y = X[:, 0] ^ X[:, 1]
    X = X.float()

    X_train, y_train = X[:train_split_pos, :], y[:train_split_pos]
    X_eval, y_eval = X[train_split_pos:, :], y[train_split_pos:]

    ## baseline
    model.eval()
    y_pred = model.predict(X_eval)
    print('baseline acc=', (y_pred == y_eval).float().mean().item())

    ## train
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    batch_size = 64
    steps = 100
    for i in range(steps):
        optimizer.zero_grad()
        # random sample train batch
        idx = torch.randint(0, X_train.shape[0], (batch_size,))
        X_batch = X_train[idx, :] # (b, d)
        y_batch = y_train[idx] # (b)

        # forward
        logits = model(X_batch) # (b, d)

        # loss
        log_probs = torch.log_softmax(logits, dim=-1) # (b, d)
        target_log_probs = torch.gather(log_probs, dim=-1, index=y_batch.unsqueeze(dim=-1)) # (b, 1)
        loss = -1 * target_log_probs.mean()

        # backward
        loss.backward()
        optimizer.step()

        # loss monitoring
        if not i % 10:
            print(f'iter: {i} loss: {loss.item()}')

    ## after training
    model.eval()
    y_pred = model.predict(X_eval)
    print('final acc=', (y_pred == y_eval).float().mean().item())

    # # Sample Output
    # baseline acc= 0.5099999904632568
    # iter: 0 loss: 0.6932724714279175
    # iter: 10 loss: 0.4926047623157501
    # iter: 20 loss: 0.3205885589122772
    # iter: 30 loss: 0.18299424648284912
    # iter: 40 loss: 0.1145416870713234
    # iter: 50 loss: 0.11375779658555984
    # iter: 60 loss: 0.05539592355489731
    # iter: 70 loss: 0.034360457211732864
    # iter: 80 loss: 0.09337535500526428
    # iter: 90 loss: 0.021514974534511566
    # final acc= 1.0


if __name__ == "__main__":
    main()
