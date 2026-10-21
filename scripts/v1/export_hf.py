"""
export_hf.py — Convert a nanoGPT checkpoint into a Hugging Face GPT-2 model folder.

Writes model.safetensors, config.json, generation_config.json and the GPT-2
tokenizer files, so the model loads with
    transformers.AutoModelForCausalLM.from_pretrained(<folder>)
Only the weights are exported (the training checkpoint also holds optimiser
state, about 3x larger).

nanoGPT is GPT-2 with bias=False; GPT-2 in transformers always has biases, so
they are written as zeros, and nanoGPT's Linear weights are transposed into
GPT-2's Conv1D layout. The script checks that both models give the same logits.

Usage:
    python scripts/v1/export_hf.py out-chronicle-v1/ckpt.pt hf_chronicle_v1 [--fp16]
Requires torch, transformers, safetensors and nanoGPT's model.py (nanoGPT/ or cwd).
"""

import os
import sys

import torch
from transformers import GPT2Config, GPT2LMHeadModel, GPT2TokenizerFast

REPO_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_nanogpt = os.path.join(REPO_DIR, "nanoGPT")
sys.path.insert(0, _nanogpt if os.path.isdir(_nanogpt) else ".")
from model import GPT, GPTConfig  # noqa: E402 (nanoGPT)

TRANSPOSED = ("attn.c_attn.weight", "attn.c_proj.weight", "mlp.c_fc.weight", "mlp.c_proj.weight")


def main():
    ckpt_path, out_dir = sys.argv[1], sys.argv[2]
    fp16 = "--fp16" in sys.argv
    ckpt = torch.load(ckpt_path, map_location="cpu")
    args = ckpt["model_args"]
    state = {k.removeprefix("_orig_mod."): v for k, v in ckpt["model"].items()}

    config = GPT2Config(vocab_size=args["vocab_size"], n_positions=args["block_size"], n_embd=args["n_embd"],
                        n_layer=args["n_layer"], n_head=args["n_head"], activation_function="gelu",
                        resid_pdrop=0.0, embd_pdrop=0.0, attn_pdrop=0.0, layer_norm_epsilon=1e-5,
                        bos_token_id=50256, eos_token_id=50256, tie_word_embeddings=True)
    hf = GPT2LMHeadModel(config)
    hf_state = hf.state_dict()
    for k in hf_state:
        if k.endswith(".attn.bias") or k.endswith(".attn.masked_bias") or k == "lm_head.weight":
            continue   # causal-mask buffers / tied head
        if k in state:
            v = state[k]
            hf_state[k] = v.t().contiguous() if k.endswith(TRANSPOSED) else v
        elif k.endswith(".bias"):
            hf_state[k] = torch.zeros_like(hf_state[k])   # nanoGPT bias=False
        else:
            raise KeyError(f"no nanoGPT weight for {k}")
    hf.load_state_dict(hf_state)
    hf.tie_weights()
    hf.eval()

    # Check: same logits as the nanoGPT model
    ref = GPT(GPTConfig(**args))
    ref.load_state_dict(state)
    ref.eval()
    x = torch.randint(0, 50257, (2, 64))
    with torch.no_grad():
        diff = (ref(x, targets=x)[0] - hf(x).logits).abs().max().item()
    print(f"max logit difference nanoGPT vs transformers: {diff:.2e}")
    assert diff < 1e-3, "conversion mismatch"

    if fp16:
        hf = hf.half()
    hf.generation_config.do_sample, hf.generation_config.top_k, hf.generation_config.temperature = True, 40, 0.7
    hf.save_pretrained(out_dir, safe_serialization=True)
    GPT2TokenizerFast.from_pretrained("gpt2").save_pretrained(out_dir)
    size = os.path.getsize(os.path.join(out_dir, "model.safetensors")) / 1e6
    print(f"Saved {out_dir}/ (model.safetensors {size:.0f}MB, {sum(p.numel() for p in hf.parameters())/1e6:.1f}M params)")


if __name__ == "__main__":
    main()
