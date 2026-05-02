"""Generate text from a trained TransformerLM checkpoint."""

import argparse
import torch
import tiktoken

from transformer import TransformerLM


@torch.no_grad()
def generate(model, prompt_tokens: list[int], max_new_tokens: int, temperature: float = 1.0, top_k: int = 50) -> list[int]:
    tokens = list(prompt_tokens)
    context_length = model.rope.cos_cached.shape[0]
    for _ in range(max_new_tokens):
        # Crop to context length
        x = torch.tensor([tokens[-context_length:]], dtype=torch.long, device=next(model.parameters()).device)
        logits = model(x)[:, -1, :]  # last position
        logits = logits / temperature
        # Top-k filtering
        if top_k > 0:
            topk_vals, _ = torch.topk(logits, top_k)
            logits[logits < topk_vals[:, -1:]] = float("-inf")
        probs = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1).item()
        tokens.append(next_token)
    return tokens


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--prompt", type=str, default="Once upon a time")
    p.add_argument("--max_tokens", type=int, default=200)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top_k", type=int, default=50)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    model_args = ckpt["args"]
    model = TransformerLM(
        d_model=model_args["d_model"],
        num_layers=model_args["num_layers"],
        vocab_size=model_args["vocab_size"],
        rope_theta=model_args["rope_theta"],
        context_length=model_args["context_length"],
        num_heads=model_args["num_heads"],
        d_ff=model_args["d_ff"],
        device=args.device,
    )
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"Loaded checkpoint from step {ckpt['step']}")

    # Encode prompt
    enc = tiktoken.get_encoding("gpt2")
    prompt_tokens = enc.encode(args.prompt)

    # Generate
    output_tokens = generate(model, prompt_tokens, args.max_tokens, args.temperature, args.top_k)
    print(enc.decode(output_tokens))


if __name__ == "__main__":
    main()
