"""
train_chronicle_v0.py — nanoGPT training config for Chronicle LLM v0.

A 30M-parameter GPT-2 style decoder-only transformer trained on ~14M tokens
of Australian historical text (1850-1950).

Pass this file to nanoGPT's train.py:
    python train.py ../config/train_chronicle_v0.py

Set `device` below to match your hardware:
    'mps'  — Apple Silicon
    'cuda' — NVIDIA GPU
    'cpu'  — fallback (very slow)

For a fresh training run set init_from = 'scratch'.
To resume from an existing checkpoint set init_from = 'resume'.
"""

# ── Output ────────────────────────────────────────────────────────────────────
out_dir = 'out-chronicle-v0'
dataset = 'australia_1850_1950'

# ── Model — 30M parameters ───────────────────────────────────────────────────
n_layer  = 6
n_head   = 6
n_embd   = 384
dropout  = 0.1
bias     = False

# ── Context & Batch ───────────────────────────────────────────────────────────
block_size                  = 256
batch_size                  = 16
gradient_accumulation_steps = 4   # effective batch = 64

# ── Training ──────────────────────────────────────────────────────────────────
max_iters      = 20000
learning_rate  = 1e-3
lr_decay_iters = 20000
min_lr         = 1e-4
beta1          = 0.9
beta2          = 0.99
weight_decay   = 0.1
grad_clip      = 1.0

# ── Hardware — set to match your system ──────────────────────────────────────
device  = 'mps'    # 'cuda', 'mps', or 'cpu'
dtype   = 'float32'
compile = False    # torch.compile is unsupported on MPS; set True for CUDA

# ── Checkpointing & Logging ───────────────────────────────────────────────────
eval_interval          = 500
eval_iters             = 40
log_interval           = 10
always_save_checkpoint = True
init_from              = 'scratch'  # 'scratch' for new run, 'resume' to continue
