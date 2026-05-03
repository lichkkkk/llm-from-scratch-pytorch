# Training Notes & Discussion

## Metrics

### Loss vs Perplexity
- Training logs report **cross-entropy loss** (nats), not perplexity
- Perplexity = `exp(loss)`
- Our best: loss 1.91 → perplexity 6.7 (model "chooses" between ~7 equally likely tokens)

### How loss is computed
- **Train loss**: single batch at the logging step — noisy, explains jaggedness in curves
- **Val loss**: average of 20 random batches from val set (~164K tokens, 3.4% of val data)
- Neither is a full epoch-level metric

## Checkpoint Anatomy

A saved `.pt` file contains:

| Component | Size | Notes |
|---|---|---|
| Model weights | 108 MB | 28.4M params × 4 bytes (float32) |
| Optimizer state | 216 MB | AdamW stores `m` (momentum) and `v` (variance) per param — 2× model |
| Metadata | ~0.1 MB | step, val_loss, training args |
| **Total** | **325 MB** | |

### Where the parameters live (our model)

| Category | % params |
|---|---|
| Embedding | 45.4% |
| LM head | 45.4% |
| FFN (× 4 layers) | 5.5% |
| Attention (× 4 layers) | 3.7% |
| Norms | <0.1% |

90% of params are in embedding + LM head because vocab (50257) >> d_model (256). Weight tying would halve model size.

### Distribution shifts at scale

| Component | Our 28M | LLaMA 2 7B | LLaMA 2 70B |
|---|---|---|---|
| Embedding + LM head | 90.8% | 4.8% | 1.2% |
| Attention | 3.7% | 33.2% | 24.5% |
| FFN | 5.5% | 62.0% | 74.3% |

As d_model and num_layers grow, attention (4d² per layer) and FFN (3·d·d_ff per layer) dominate. GQA further reduces attention params — LLaMA 3 70B is ~80% FFN. This is why MoE targets FFN (e.g. Mixtral activates 2 of 8 FFN experts per token).

## Training Stability

### Why our training was stable
- Small model (28M) — forgiving to hyperparams
- fp32 — no precision issues
- Gradient clipping (max_norm=1.0)
- RMSNorm (prevents embedding norm blowup)
- Conservative batch size

### Common causes of instability

1. **LR too high** — larger models need lower LR (~1/√model_size)
2. **fp16 without loss scaling** — gradients underflow/overflow
3. **Gradient spikes** — rare extreme batches; PaLM skipped ~100 bad batches during training
4. **Attention entropy collapse** — softmax concentrates on few positions; QK norm helps (Gemma, DeepSeek)
5. **Insufficient warmup** — Adam state unreliable early on
6. **Bad init at depth** — residual variance grows with layers; GPT-2 scales by 1/√(n_layers)
7. **Small batch size** — high gradient variance, interacts with LR

### Techniques used at scale

- **bf16 over fp16** — larger dynamic range, less overflow
- **QK normalization** — stabilizes attention scores
- **μP** — hyperparams transfer across model scales
- **Z-loss** — penalizes large logits, prevents softmax saturation
- **Spike detection + rollback** — checkpoint frequently, replay from last good state

## Warmup

Warmup exists for two reasons:

**1. Optimizer cold start (primary)**
- Adam's `m` and `v` are initialized at 0
- Bias correction at step 1 amplifies noise by 10× (with β1=0.9)
- Need steps to build reliable running averages before full LR

**2. Loss landscape geometry (secondary)**
- Early in training, weights are random, gradients point in unstable directions
- Model hasn't found a good basin yet — large steps overshoot
- This is why warmup helps even SGD (no momentum state)
