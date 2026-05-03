"""Training loop for TransformerLM on TinyStories."""

import os
import csv
import time
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt

from transformer import TransformerLM, cross_entropy, AdamW
from transformer.training_utils import get_lr_cosine_schedule, gradient_clipping, get_batch


def train(args):
    device = args.device
    os.makedirs(args.ckpt_dir, exist_ok=True)

    # Load data
    print(f"Loading data from {args.data_dir}...")
    train_data = np.load(os.path.join(args.data_dir, "train.npy"))
    val_data = np.load(os.path.join(args.data_dir, "val.npy"))
    print(f"  train: {len(train_data):,} tokens, val: {len(val_data):,} tokens")

    # Model
    model = TransformerLM(
        d_model=args.d_model,
        num_layers=args.num_layers,
        vocab_size=args.vocab_size,
        rope_theta=args.rope_theta,
        context_length=args.context_length,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        device=device,
    )
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model: {param_count:,} parameters")

    optimizer = AdamW(model.parameters(), lr=args.lr_max, weight_decay=args.weight_decay)

    best_val_loss = float("inf")
    t0 = time.time()
    log_file = os.path.join(args.ckpt_dir, "log.csv")
    with open(log_file, "w", newline="") as f:
        csv.writer(f).writerow(["step", "train_loss", "val_loss"])
    train_log = []  # (step, loss) for every log_interval
    val_log = []    # (step, loss) for every eval_interval

    for step in range(1, args.max_steps + 1):
        model.train()
        # LR schedule
        lr = get_lr_cosine_schedule(step, args.lr_max, args.lr_min, args.warmup_steps, args.max_steps)
        for group in optimizer.param_groups:
            group["lr"] = lr

        # Forward / backward
        x, y = get_batch(train_data, args.batch_size, args.context_length, device)
        logits = model(x)
        loss = cross_entropy(logits, y)
        optimizer.zero_grad()
        loss.backward()
        gradient_clipping(model.parameters(), args.max_grad_norm)
        optimizer.step()

        # Logging
        if step % args.log_interval == 0:
            elapsed = time.time() - t0
            train_loss = loss.item()
            print(f"step {step:>6d} | loss {train_loss:.4f} | lr {lr:.2e} | {elapsed:.1f}s")
            train_log.append((step, train_loss))
            with open(log_file, "a", newline="") as f:
                csv.writer(f).writerow([step, f"{train_loss:.6f}", ""])

        # Validation
        if step % args.eval_interval == 0:
            model.eval()
            val_losses = []
            with torch.no_grad():
                for _ in range(args.eval_steps):
                    xv, yv = get_batch(val_data, args.batch_size, args.context_length, device)
                    val_loss = cross_entropy(model(xv), yv)
                    val_losses.append(val_loss.item())
            avg_val_loss = sum(val_losses) / len(val_losses)
            print(f"  --> val loss {avg_val_loss:.4f}")
            val_log.append((step, avg_val_loss))
            with open(log_file, "a", newline="") as f:
                csv.writer(f).writerow([step, "", f"{avg_val_loss:.6f}"])
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                ckpt_path = os.path.join(args.ckpt_dir, "best.pt")
                torch.save({
                    "step": step,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "val_loss": avg_val_loss,
                    "args": vars(args),
                }, ckpt_path)
                print(f"  --> saved checkpoint to {ckpt_path}")

    # Save final checkpoint
    ckpt_path = os.path.join(args.ckpt_dir, "final.pt")
    torch.save({
        "step": step,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "args": vars(args),
    }, ckpt_path)
    print(f"Training done. Final checkpoint: {ckpt_path}")

    # Plot loss curve
    fig, ax = plt.subplots()
    if train_log:
        steps, losses = zip(*train_log)
        ax.plot(steps, losses, label="train", alpha=0.7)
    if val_log:
        steps, losses = zip(*val_log)
        ax.plot(steps, losses, label="val", marker="o", markersize=4)
    ax.set_xlabel("step")
    ax.set_ylabel("loss")
    ax.legend()
    ax.set_title("Training Loss")
    plot_path = os.path.join(args.ckpt_dir, "loss_curve.png")
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Loss curve saved to {plot_path}")
    print(f"Training log saved to {log_file}")


def main():
    p = argparse.ArgumentParser()
    # Data
    p.add_argument("--data_dir", type=str, default="data/tinystories")
    p.add_argument("--ckpt_dir", type=str, default="checkpoints")
    # Model
    p.add_argument("--d_model", type=int, default=256)
    p.add_argument("--num_layers", type=int, default=4)
    p.add_argument("--num_heads", type=int, default=4)
    p.add_argument("--d_ff", type=int, default=512)
    p.add_argument("--vocab_size", type=int, default=50257)  # GPT-2 vocab
    p.add_argument("--context_length", type=int, default=256)
    p.add_argument("--rope_theta", type=float, default=10000.0)
    # Training
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--max_steps", type=int, default=5000)
    p.add_argument("--lr_max", type=float, default=1e-3)
    p.add_argument("--lr_min", type=float, default=1e-5)
    p.add_argument("--warmup_steps", type=int, default=200)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    # Eval / logging
    p.add_argument("--log_interval", type=int, default=50)
    p.add_argument("--eval_interval", type=int, default=500)
    p.add_argument("--eval_steps", type=int, default=20)
    # Device
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    train(p.parse_args())


if __name__ == "__main__":
    main()
