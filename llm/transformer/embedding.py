from math import sqrt
import torch
# import torch.nn.functional as F

class Embedding(torch.nn.Module):

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None
    ) -> None:
        super().__init__()
        self.vocab_size = num_embeddings
        self.d_model = embedding_dim
        self.W = torch.nn.Parameter(
            torch.empty(
                size=(self.vocab_size, self.d_model),
                device=device,
                dtype=dtype)
        )
        # Xavier initialization
        torch.nn.init.trunc_normal_(
            tensor=self.W,
            mean=0,
            std=sqrt(2 / (self.vocab_size + self.d_model)),
            a = -3,
            b = 3)

    def forward(self, token_ids: torch.LongTensor) -> torch.Tensor:
        '''
        'batch_size, seq_len' -> 'batch_size, seq_len, d_model'
        '''
        # return F.one_hot(token_ids, num_classes=self.vocab_size).float() @ self.W
        # the lookup approach is more efficient since no need to materialize the onehot
        return self.W[token_ids]
