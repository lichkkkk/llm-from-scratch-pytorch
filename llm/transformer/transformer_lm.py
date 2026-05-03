import torch

from .embedding import Embedding
from .position import RoPE
from .transformer import Transformer
from .norm import RMSNorm
from .linear import Linear

class TransformerLM(torch.nn.Module):

    def __init__(
        self,
        # basic param
        d_model: int,
        num_layers: int,
        # embedding param
        vocab_size: int,
        # position encoding param
        rope_theta: float,
        context_length: int,
        # attention param
        num_heads: int,
        # ffn param
        d_ff: int,
        # other
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,

    ):
        super().__init__()
        self.embd = Embedding(num_embeddings=vocab_size, embedding_dim=d_model, device=device, dtype=dtype)
        self.rope = RoPE(theta=rope_theta, d_k=d_model//num_heads, max_seq_len=context_length, device=device)
        self.layers = torch.nn.ModuleList([
            Transformer(d_model=d_model, num_heads=num_heads, d_ff=d_ff, position_encoding=self.rope, device=device, dtype=dtype)
            for _ in range(num_layers)
        ])
        self.ln = RMSNorm(d_model=d_model, device=device, dtype=dtype)
        self.lm_head = Linear(in_features=d_model, out_features=vocab_size, device=device, dtype=dtype)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        (batch, seq) -> (batch, seq, vocab)
        '''
        x = self.embd(x)
        for layer in self.layers:
            x = layer(x)
        x = self.ln(x)
        x = self.lm_head(x)
        return x
