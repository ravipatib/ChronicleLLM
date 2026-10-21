"""
prepare.py — Tokenise the v1 cleaned texts into binary files for nanoGPT training.

Reads all .txt files from the cleaned/ directory (sibling of this script),
encodes them with the GPT-2 BPE tokeniser (tiktoken), separates documents with
the end-of-text token, and writes train.bin and val.bin in the same directory.

Unlike v0 (which used the alphabetically last 10% of tokens as validation),
the split is by whole document, chosen with a fixed seed, so validation is a
fair sample of the corpus and no book straddles both sets.

Usage (from anywhere in the repo):
    python data/australia_1850_1950_v1/prepare.py
"""

import os
import random
import numpy as np
import tiktoken
from tqdm import tqdm

# Paths are relative to this script so it can be run from any working directory
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
CLEANED_DIR = os.path.join(SCRIPT_DIR, "cleaned")
OUT_DIR     = SCRIPT_DIR
VAL_SHARE   = 0.05
SEED        = 1850


def main():
    enc = tiktoken.get_encoding("gpt2")
    EOT = enc.eot_token  # token 50256 — used as a document separator

    files = sorted([f for f in os.listdir(CLEANED_DIR) if f.endswith('.txt')])
    random.Random(SEED).shuffle(files)
    n_val = max(1, int(len(files) * VAL_SHARE))
    splits = {"val": files[:n_val], "train": files[n_val:]}
    print(f"Tokenising {len(files)} cleaned texts ({len(splits['train'])} train / {n_val} val)...\n")

    counts = {}
    for split, names in splits.items():
        tokens = []
        for fname in tqdm(names, desc=split):
            with open(os.path.join(CLEANED_DIR, fname), 'r', encoding='utf-8') as f:
                text = f.read().strip()
            if len(text) > 100:
                tokens.extend(enc.encode_ordinary(text))
                tokens.append(EOT)
        arr = np.array(tokens, dtype=np.uint16)
        arr.tofile(os.path.join(OUT_DIR, f"{split}.bin"))
        counts[split] = len(arr)

    print(f"\nDone.")
    print(f"Total tokens : {sum(counts.values())/1e6:.2f}M")
    print(f"Train tokens : {counts['train']/1e6:.2f}M")
    print(f"Val tokens   : {counts['val']/1e6:.2f}M")
    print(f"Val files    : {', '.join(sorted(splits['val']))}")
    print(f"Output       : {OUT_DIR}/train.bin, val.bin")


if __name__ == "__main__":
    main()
