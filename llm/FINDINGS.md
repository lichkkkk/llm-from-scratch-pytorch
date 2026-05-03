# TransformerLM Training Findings

## Model

- 28M parameter decoder-only transformer
- 4 layers, d_model=256, 4 heads, d_ff=512
- RoPE positional encoding, RMSNorm, SwiGLU FFN
- GPT-2 tokenizer (tiktoken, vocab_size=50257)
- Context length: 256

## Hardware

- Apple M4, 16GB unified memory
- MPS backend (Metal 3, 8 GPU cores)
- Max batch size: 32 (at context_length=256, ~1.8GB GPU memory)
- Batch size 64 causes OOM
- Throughput: ~2.7s/step (32 batch × 256 seq = 8192 tokens/step)

## Training Run (1h, 1300 steps)

- Dataset: TinyStories (474M train tokens, 4.8M val tokens)
- Optimizer: AdamW (lr_max=1e-3, lr_min=1e-5, weight_decay=0.01)
- LR schedule: cosine with 100-step warmup
- Gradient clipping: max L2 norm = 1.0
- Tokens seen: ~10.6M (2.2% of training data)

| Metric | Start | End |
|---|---|---|
| Train loss | 10.72 | 10.25 |
| Val loss | 10.61 | 10.26 |

- No overfitting (train/val curves track closely)
- Loss still decreasing at end of run
- Generated text is gibberish at this stage

## Estimate to Reach Loss < 4

- Chinchilla-optimal: ~560M tokens for 28M params (~68K steps)
- Estimated steps needed: 30K-60K
- Estimated wall time: 22-45 hours on this hardware
- Key bottleneck: small batch size due to 16GB memory limit

## Potential Speedups

- Gradient accumulation (effective batch 128+ without more memory)
- Mixed precision (float16)
- torch.compile (if MPS supports it)
- Training on a CUDA GPU
