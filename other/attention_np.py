import numpy as np

class MultiHeadSelfAttention:

    def __init__(self, d_model: int, num_head: int):
        self.num_head = num_head
        self.d_head = d_model // num_head
        self.Wk = np.random.normal(0, 0.02, (d_model, d_model))  # (D, D)
        self.Wq = np.random.normal(0, 0.02, (d_model, d_model))
        self.Wv = np.random.normal(0, 0.02, (d_model, d_model))
        self.Wo = np.random.normal(0, 0.02, (d_model, d_model))

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        x = x - np.max(x, axis=-1, keepdims=True)
        exp_logits = np.exp(x)
        return exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

    def forward(self, x: np.ndarray) -> np.ndarray:
        '''
        x: (B, S, D)
        '''
        B, S, D = x.shape
        H = self.num_head
        D_H = self.d_head

        k = x @ self.Wk  # (B, S, D)
        q = x @ self.Wq
        v = x @ self.Wv

        k = k.reshape(B, S, H, D_H).transpose(0, 2, 1, 3)  # (B, H, S, D_H)
        q = q.reshape(B, S, H, D_H).transpose(0, 2, 1, 3)
        v = v.reshape(B, S, H, D_H).transpose(0, 2, 1, 3)

        scores = q @ k.transpose(0, 1, 3, 2)  # (B, H, S, S)
        scores /= np.sqrt(D_H)

        causal_mask = np.tril(np.ones((S, S), dtype=bool))
        att_weights = np.where(causal_mask[None, None, :, :], scores, -np.inf)

        att_weights = self._softmax(att_weights)

        att_out = att_weights @ v  # (B, H, S, D_H)
        att_out = att_out.transpose(0, 2, 1, 3).reshape(B, S, D)

        return att_out @ self.Wo, att_weights

# Test
B, S, D = 1, 2, 4

x = np.random.randn(B, S, D)
attn = MultiHeadSelfAttention(d_model=D, num_head=2)

out, weights = attn.forward(x)

print(out.shape)      # (1, 2, 4)
print(weights.shape)  # (1, 2, 2, 2)