"""
chat_v1.py — Ask Chronicle LLM v1 questions, framed as a period parliamentary exchange.

v1 is a base model: it continues text, it was never trained to answer. So each
question is written into the shape the model saw most often for questions and
answers, Commonwealth Hansard question time:

    QUESTION

    Mr SMITH. — I ask the Prime Minister whether ...?

    Mr DEAKIN. — <the model continues here>

and generation stops when the next speaker or heading starts. Earlier turns are
kept in the prompt while they fit in the 512-token context.

Expect period style rather than reliable facts: at ~73M parameters the model
often invents names, figures and events.

Usage:
    python chat_v1.py                          # Hansard style, answered by "Mr DEAKIN"
    python chat_v1.py --answerer "Mr BRUCE"    # pick who answers
    python chat_v1.py --style interview        # plain "Q. — / A. —" framing
    python chat_v1.py --ckpt path/to/ckpt.pt

Requires nanoGPT's model.py (a nanoGPT/ subdirectory or the current directory).
"""

import argparse
import contextlib
import io
import os
import re
import sys

import tiktoken
import torch

_nanogpt_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'nanoGPT')
sys.path.insert(0, _nanogpt_dir if os.path.isdir(_nanogpt_dir) else '.')
from model import GPTConfig, GPT  # noqa: E402 (nanoGPT)

DEVICE = ('cuda' if torch.cuda.is_available() else
          'mps'  if torch.backends.mps.is_available() else 'cpu')
# A new speaker ("Mr BRUCE. — ", "Sir JOHN FORREST. — ") or an upper-case heading ends the answer
NEXT_TURN = re.compile(r"\n\n(?:[A-Z][A-Za-z.' -]{1,40}\. — |Q\. — |[A-Z][A-Z ,.'-]{3,}\n)")


def load(path):
    with contextlib.redirect_stdout(io.StringIO()):
        ckpt  = torch.load(path, map_location=DEVICE)
        model = GPT(GPTConfig(**ckpt['model_args']))
        state = ckpt['model']
        # checkpoints from a torch.compile run prefix every key with "_orig_mod."
        state = {k.removeprefix('_orig_mod.'): v for k, v in state.items()}
        model.load_state_dict(state)
    return model.to(DEVICE).eval()


@torch.no_grad()
def answer(model, enc, prompt, max_new_tokens, temperature, top_k):
    """Sample until the next speaker/heading or max_new_tokens; return the answer text."""
    block = model.config.block_size
    ids   = enc.encode(prompt)
    start = len(ids)
    for _ in range(max_new_tokens):
        idx    = torch.tensor([ids[-block:]], device=DEVICE)
        logits = model(idx)[0][:, -1, :] / temperature
        v, _   = torch.topk(logits, min(top_k, logits.size(-1)))
        logits[logits < v[:, [-1]]] = -float('inf')
        ids.append(torch.multinomial(torch.softmax(logits, dim=-1), 1).item())
        text = enc.decode(ids[start:])
        m = NEXT_TURN.search(text)
        if m:
            return text[:m.start()].strip()
    return enc.decode(ids[start:]).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default='out-chronicle-v1/ckpt.pt')
    ap.add_argument('--style', choices=['hansard', 'interview'], default='hansard')
    ap.add_argument('--asker', default='Mr SMITH')
    ap.add_argument('--answerer', default='Mr DEAKIN')
    ap.add_argument('--max_new_tokens', type=int, default=200)
    ap.add_argument('--temperature', type=float, default=0.7)
    ap.add_argument('--top_k', type=int, default=40)
    args = ap.parse_args()

    model = load(args.ckpt)
    enc   = tiktoken.get_encoding('gpt2')
    room  = model.config.block_size - args.max_new_tokens   # tokens left for the conversation so far
    history = 'QUESTION\n\n' if args.style == 'hansard' else ''

    print(f"Chronicle LLM v1 — {args.style} style on {DEVICE}. Empty line or Ctrl-D to quit.\n")
    while True:
        try:
            q = input('You: ').strip()
        except EOFError:
            break
        if not q:
            break
        if args.style == 'hansard':
            turn = f"{args.asker}. — {q}\n\n{args.answerer}. — "
        else:
            turn = f"Q. — {q}\n\nA. — "
        prompt = history + turn
        while len(enc.encode(prompt)) > room and '\n\n' in history.strip():
            history = history.split('\n\n', 1)[1]            # drop the oldest paragraph
            prompt  = history + turn
        reply = answer(model, enc, prompt, args.max_new_tokens, args.temperature, args.top_k)
        print(f"{args.answerer if args.style == 'hansard' else 'A'}: {reply}\n")
        history = prompt + reply + '\n\n'


if __name__ == '__main__':
    main()
