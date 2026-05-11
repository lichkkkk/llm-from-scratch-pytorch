import numpy as np

class MLP:

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int
    ):
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim

        self.W1 = np.random.normal(loc=0, scale=np.sqrt(2.0 / in_dim), size=(in_dim, hidden_dim))
        self.b1 = np.zeros(shape=(hidden_dim,))
        self.W2 = np.random.normal(loc=0, scale=np.sqrt(2.0 / hidden_dim), size=(hidden_dim, out_dim))
        self.b2 = np.zeros(shape=(out_dim,))

    @staticmethod
    def _relu(x: np.ndarray) -> np.ndarray:
        return np.maximum(x, 0)

    def _forward(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        '''
        x: (B, D)
        '''
        z1 = x @ self.W1 + self.b1  # (B, H)
        h = self._relu(z1)  # (B, H)
        logits = h @ self.W2 + self.b2  # (B, D_out)

        return z1, h, logits

    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        logits = logits - np.max(logits, axis=1, keepdims=True)
        exp_logits = np.exp(logits)
        return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

    @staticmethod
    def _cross_entropy(logits: np.ndarray, y: np.ndarray) -> float:
        logits_shifted = logits - np.max(logits, axis=1, keepdims=True)
        logsumexp = np.log(
            np.sum(np.exp(logits_shifted), axis=1)
        )
        correct_logits = logits_shifted[np.arange(len(y)), y]
        loss = np.mean(logsumexp - correct_logits)
        return loss

    def train_cross_entropy(self, x: np.ndarray, targets: np.ndarray, lr: float) -> float:
        '''
        x: (B, D), targets: (B)
        '''
        B, D = x.shape

        z1, h, logits = self._forward(x)
        loss = self._cross_entropy(logits, targets)

        # d_L / d_logits 
        probs = self._softmax(logits)
        dlogits = probs
        dlogits[np.arange(B), targets] -= 1.0
        dlogits /= B  # (B, D_out)

        # d_L/d_W2 = d_L/d_logits * d_logits/d_W2
        # d_logits/d_W2 = h  (B, D_hidden)
        dW2 = h.T @ dlogits

        # d_L/d_b2 = d_L/d_logits * d_logits/d_b2
        # d_logits/d_b2 = 1
        db2 = dlogits.sum(axis=0)

        dh = dlogits @ self.W2.T
        dz1 = dh * (z1 > 0).astype(z1.dtype)  # (B, D_hidden)

        # d_L/d_W1 = d_L/d_logts * d_logits/d_h * d_h / d_z1  * d_z1 / d_W1
        dW1 = x.T @ dz1
        db1 = dz1.sum(axis=0)

        self.W1 -= lr * dW1
        self.b1 -= lr * db1
        self.W2 -= lr * dW2
        self.b2 -= lr * db2

        return loss

    def predict(self, x: np.ndarray) -> np.ndarray:
        _, _, y = self._forward(x)
        return np.argmax(y, axis=-1)



X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1],
], dtype=np.float64)

y = np.array([0, 1, 1, 0], dtype=np.int64)

mlp = MLP(in_dim=2, hidden_dim=8, out_dim=2)

for step in range(100):
    loss = mlp.train_cross_entropy(X, y, lr=0.1)

    if step % 20 == 0:
        pred = mlp.predict(X)
        acc = np.mean(pred == y)
        print(step, loss, acc)

# # Example output
# 0 0.9169417569663788 0.25
# 20 0.6091688219505791 1.0
# 40 0.4933069332499199 1.0
# 60 0.4046725284800917 1.0
# 80 0.3240121734592549 1.0