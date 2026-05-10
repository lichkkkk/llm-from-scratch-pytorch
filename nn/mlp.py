import torch

def relu(x: torch.Tensor) -> torch.Tensor:
    return x.maximum(torch.zeros_like(x))

def softmax(x: torch.Tensor) -> torch.Tensor:
    '''
    softmax_i = exp(xi) / sum(exp(xi))
    '''
    x = x - x.max(dim=-1, keepdim=True).values
    exp = x.exp()
    exp_sum = exp.sum(dim=-1, keepdim=True)
    return exp / exp_sum


def cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    '''
    CE = - log prob (x)[target]

    logits: (b, c)
    targets: (b)
    loss: (,)
    '''
    probs = softmax(logits)  # (b, c)
    targets_probs = probs[torch.arange(logits.shape[0]), targets]  # (b,)
    return -1 * torch.log(targets_probs + 1e-8).mean()


def batch_norm(x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
    '''
    x.shape = b, d
    gamma.shape = d,
    beta.shape = d,
    '''
    x_mean = x.mean(dim=0, keepdim=True)  # (1, d)
    x_var = x.var(dim=0, unbiased=False, keepdim=True)  # (1, d)
    x_hat = (x - x_mean) / torch.sqrt(x_var + 1e-8)  # (b, d)
    return x_hat * gamma + beta


class MLPTorch(torch.nn.Module):

    def __init__(self, d_in: int, d_hidden: int, d_out: int):
        super().__init__()
        self.w1 = torch.nn.Parameter(torch.empty((d_in, d_hidden), dtype=torch.float))
        self.b1 = torch.nn.Parameter(torch.zeros(size=(d_hidden,), dtype=torch.float))  # should remove since we have BN
        self.bn_gamma = torch.nn.Parameter(torch.ones(size=(d_hidden,), dtype=torch.float))
        self.bn_beta = torch.nn.Parameter(torch.zeros(size=(d_hidden,), dtype=torch.float))
        self.w2 = torch.nn.Parameter(torch.empty((d_hidden, d_out), dtype=torch.float))
        self.b2 = torch.nn.Parameter(torch.zeros(size=(d_out,), dtype=torch.float))

        torch.nn.init.normal_(self.w1, mean=0, std=0.02)
        torch.nn.init.normal_(self.w2, mean=0, std=0.02)

    def forward(self, x: torch.Tensor, targets: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = x @ self.w1 + self.b1  # (b, d_hidden)
        x = batch_norm(x, self.bn_gamma, self.bn_beta)
        x = relu(x)
        logits = x @ self.w2 + self.b2

        loss = None
        if targets is not None:
            loss = cross_entropy(logits, targets)

        return logits, loss

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        logits, _ = self.forward(x, None)
        y = torch.argmax(logits, dim=-1)
        return y


mlp_torch = MLPTorch(d_in=2, d_hidden=8, d_out=2)

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
y_pred = mlp_torch.predict(X_eval)
print('baseline acc=', (y_pred == y_eval).float().mean().item())

## train
optimizer = torch.optim.Adam(mlp_torch.parameters(), lr=0.01)

batch_size = 64
steps = 100
for i in range(steps):
    optimizer.zero_grad()
    # random sample train batch
    idx = torch.randint(0, X_train.shape[0], (batch_size,))
    X_batch = X_train[idx, :]
    y_batch = y_train[idx]
    _, loss = mlp_torch(X_batch, y_batch)
    loss.backward()
    optimizer.step()
    if not i % 10:
        print(f'iter: {i} loss: {loss.item()}')

## after training
y_pred = mlp_torch.predict(X_eval)
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
