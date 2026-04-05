"""
generate_sample.py — Generate text from a trained Chronicle LLM checkpoint.

Loads the checkpoint at out-chronicle-v0/ckpt.pt and samples up to MAX_TOKENS
tokens from the given prompt. Requires nanoGPT's model.py to be importable
(either in a nanoGPT/ subdirectory or the current directory).

Usage:
    python generate_sample.py "The goldfields of Ballarat in 1854"
    python generate_sample.py          # prompts interactively
"""

import sys
import os
import contextlib
import io

import torch
import tiktoken

# Resolve nanoGPT — prefer nanoGPT/ subdirectory, fall back to current dir.
# Clone with: git clone https://github.com/karpathy/nanoGPT
_nanogpt_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'nanoGPT')
sys.path.insert(0, _nanogpt_dir if os.path.isdir(_nanogpt_dir) else '.')

from model import GPTConfig, GPT  # noqa: E402 (nanoGPT)

# ── Config ────────────────────────────────────────────────────────────────────
CHECKPOINT  = 'out-chronicle-v0/ckpt.pt'
MAX_TOKENS  = 200
TEMPERATURE = 0.5
TOP_K       = 20
DEVICE      = (
    'mps'  if torch.backends.mps.is_available() else
    'cuda' if torch.cuda.is_available()          else
    'cpu'
)

# ── Prompt ────────────────────────────────────────────────────────────────────
PROMPT = sys.argv[1] if len(sys.argv) >= 2 else input("Enter prompt: ")


def main():
    print()
    print("=" * 60)
    print("  Chronicle LLM v0")
    print("  Australian Historical Texts 1850-1950")
    print("  github.com/ravipatib/ChronicleLLM")
    print("=" * 60)
    print(f"\nPrompt: \"{PROMPT}\"")
    print(f"Device: {DEVICE}\n")
    print("-" * 60)

    # Load checkpoint (suppress nanoGPT's verbose print)
    with contextlib.redirect_stdout(io.StringIO()):
        ckpt  = torch.load(CHECKPOINT, map_location=DEVICE)
        model = GPT(GPTConfig(**ckpt['model_args']))
        model.load_state_dict(ckpt['model'])
        model.to(DEVICE)
        model.eval()

    enc    = tiktoken.get_encoding("gpt2")
    tokens = enc.encode(PROMPT)
    idx    = torch.tensor([tokens], dtype=torch.long, device=DEVICE)

    with torch.no_grad():
        out = model.generate(idx,
                             max_new_tokens=MAX_TOKENS,
                             temperature=TEMPERATURE,
                             top_k=TOP_K)

    print(enc.decode(out[0].tolist()))
    print("-" * 60)
    print()


if __name__ == "__main__":
    main()
