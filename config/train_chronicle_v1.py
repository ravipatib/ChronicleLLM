"""
train_chronicle_v1.py — nanoGPT training config for Chronicle LLM v1.

A ~73M-parameter GPT-2 style decoder-only transformer trained on ~248M tokens
(235.1M train / 12.7M val) of Australian text 1788-1945
(data/australia_1850_1950_v1: Gutenberg, PGA, Internet Archive books, sampled
Commonwealth Hansard). See that folder's README for the data.

Sizing: v0 (30M params) saw its 14M tokens ~23 times, hence its train/val gap.
v1 runs 3 epochs over its training tokens (~3.2 tokens per parameter).
Repeating data up to ~4 epochs is almost as good as fresh data (Muennighoff et
al. 2023, "Scaling Data-Constrained Language Models"), so this is about the
largest model the licence-safe data supports without memorising it.

Made for a free Kaggle GPU notebook (2x T4, float16; see kaggle/). Run with:
    torchrun --standalone --nproc_per_node=2 train.py config/train_chronicle_v1.py
One GPU (T4 or P100):
    python train.py config/train_chronicle_v1.py

Set `device`/`dtype`/`compile` below to match your hardware:
    'cuda' + 'float16'  — T4 / P100 (no bfloat16 support)
    'cuda' + 'bfloat16' — A100 / H100 / RTX 30xx and newer
    'mps'  + 'float32'  — PyTorch MPS backend (set compile = False; much slower)

For a fresh training run set init_from = 'scratch'.
To resume from an existing checkpoint set init_from = 'resume'.
"""

# ── Output ────────────────────────────────────────────────────────────────────
out_dir = 'out-chronicle-v1'
dataset = 'australia_1850_1950_v1'

# ── Model — ~73M parameters (29.0M embedding + 43.8M transformer) ─────────────
n_layer  = 11
n_head   = 9        # head size 64
n_embd   = 576
dropout  = 0.1      # kept for multi-epoch training
bias     = False

# ── Context & Batch ───────────────────────────────────────────────────────────
block_size                  = 512   # v0: 256 — longer context for prose and debates
batch_size                  = 16
gradient_accumulation_steps = 8     # effective batch = 128 sequences = 65,536 tokens
                                    # (split across GPUs by torchrun; must divide by GPU count)

# ── Training — ~3 epochs ──────────────────────────────────────────────────────
max_iters      = 10800   # 10,800 x 65,536 = 708M tokens = 3.0 epochs of 235.1M train tokens
warmup_iters   = 300
learning_rate  = 6e-4
lr_decay_iters = 10800
min_lr         = 6e-5
beta1          = 0.9
beta2          = 0.95
weight_decay   = 0.1
grad_clip      = 1.0

# ── Hardware — set to match your system ──────────────────────────────────────
device  = 'cuda'      # 'cuda', 'mps', or 'cpu'
dtype   = 'float16'   # T4/P100 have no bfloat16
compile = True        # needs a GPU with compute capability >= 7.0 (T4 yes, P100 no)

# ── Checkpointing & Logging ───────────────────────────────────────────────────
eval_interval          = 250
eval_iters             = 100
log_interval           = 10
always_save_checkpoint = False   # save only when val loss improves; a cut-off run resumes from that
init_from              = 'scratch'  # 'scratch' for new run, 'resume' to continue
