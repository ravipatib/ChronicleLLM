"""
prepare.py — Tokenise cleaned texts into binary files for nanoGPT training.

Reads all .txt files from the cleaned/ directory (sibling of this script),
encodes them with the GPT-2 BPE tokeniser (tiktoken), separates books with
the end-of-text token, and writes a 90/10 train/val split as train.bin and
val.bin in the same directory.

Usage (from anywhere in the repo):
    python data/australia_1850_1950/prepare.py
"""

import os
import numpy as np
import tiktoken
from tqdm import tqdm

# Paths are relative to this script so it can be run from any working directory
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
CLEANED_DIR = os.path.join(SCRIPT_DIR, "cleaned")
OUT_DIR     = SCRIPT_DIR


def main():
    enc = tiktoken.get_encoding("gpt2")
    EOT = enc.eot_token  # token 50256 — used as a book separator

    files = sorted([f for f in os.listdir(CLEANED_DIR) if f.endswith('.txt')])
    print(f"Tokenising {len(files)} cleaned texts...\n")

    all_tokens = []
    for fname in tqdm(files):
        path = os.path.join(CLEANED_DIR, fname)
        with open(path, 'r', encoding='utf-8') as f:
            text = f.read().strip()
        if len(text) > 100:
            tokens = enc.encode_ordinary(text)
            all_tokens.extend(tokens)
            all_tokens.append(EOT)

    all_tokens = np.array(all_tokens, dtype=np.uint16)

    split        = int(0.9 * len(all_tokens))
    train_tokens = all_tokens[:split]
    val_tokens   = all_tokens[split:]

    train_tokens.tofile(os.path.join(OUT_DIR, "train.bin"))
    val_tokens.tofile(os.path.join(OUT_DIR,   "val.bin"))

    print(f"\nDone.")
    print(f"Total tokens : {len(all_tokens)/1e6:.2f}M")
    print(f"Train tokens : {len(train_tokens)/1e6:.2f}M")
    print(f"Val tokens   : {len(val_tokens)/1e6:.2f}M")
    print(f"Output       : {OUT_DIR}/train.bin, val.bin")


if __name__ == "__main__":
    main()
