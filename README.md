# LLM from Scratch with PyTorch

A from-scratch implementation of a decoder-only Transformer in PyTorch, with scripts to tokenize TinyStories, train the model, and generate text. The repository also includes small neural network examples in PyTorch and NumPy.

## Repository layout

- [`llm/`](llm/) — Transformer with RoPE, RMSNorm, and SwiGLU, plus data preparation, training, and text generation.
- [`nn/`](nn/) — XOR classifiers in PyTorch and NumPy. The NumPy version implements backpropagation by hand.
- [`other/attention_np.py`](other/attention_np.py) — Standalone multi-head self-attention in NumPy.
- [`rl/GSM8K_GRPO.ipynb`](rl/GSM8K_GRPO.ipynb) — Colab notebook for training with Group Relative Policy Optimization (GRPO) on GSM8K.

## Quick start

Run these commands from the repository root in a Python environment of your choice.

```sh
python -m pip install torch numpy einops tiktoken datasets matplotlib pytest
python llm/data.py
python llm/train.py
python llm/generate.py --checkpoint checkpoints/final.pt --prompt "Once upon a time"
```

The first data run downloads TinyStories. Training saves checkpoints and a loss curve under `checkpoints/`.

Commands for the XOR demos and Transformer tests

```sh
python nn/mlp.py
python nn/mlp_np.py
python -m pytest llm/transformer
```

See the [training notes](llm/NOTES.md) and [experiment results](llm/FINDINGS.md) for more details.
