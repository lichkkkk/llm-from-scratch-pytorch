"""Download TinyStories and tokenize to numpy memmap files."""

import os
import numpy as np
import tiktoken
from datasets import load_dataset


def prepare_tinystories(out_dir: str = "data/tinystories"):
    os.makedirs(out_dir, exist_ok=True)
    train_path = os.path.join(out_dir, "train.npy")
    val_path = os.path.join(out_dir, "val.npy")
    if os.path.exists(train_path) and os.path.exists(val_path):
        print(f"Data already exists at {out_dir}, skipping.")
        return train_path, val_path

    print("Loading TinyStories from HuggingFace...")
    ds = load_dataset("roneneldan/TinyStories")
    enc = tiktoken.get_encoding("gpt2")

    for split, path in [("train", train_path), ("validation", val_path)]:
        print(f"Tokenizing {split}...")
        tokens = []
        for example in ds[split]:
            tokens.extend(enc.encode_ordinary(example["text"]))
            tokens.append(enc.eot_token)
        arr = np.array(tokens, dtype=np.uint16)
        np.save(path, arr)
        print(f"  {split}: {len(arr):,} tokens -> {path}")

    return train_path, val_path


if __name__ == "__main__":
    prepare_tinystories()
