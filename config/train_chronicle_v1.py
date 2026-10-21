"""
train_chronicle_v1.py — nanoGPT training config for Chronicle LLM v1.

A ~51M-parameter GPT-2 style decoder-only transformer trained on ~160M tokens
of Australian text 1788-1945 (data/australia_1850_1950_v1: Gutenberg, PGA,
Internet Archive books, sampled Commonwealth Hansard).

Sizing: v0 (30M params) saw its 14M tokens ~23 times, hence the train/val gap.
v1 runs ~2 epochs (5,000 iters x 65,536 tokens = 328M tokens) over 11x more
data. On the Apple Silicon machine that trained v0 (~18k tok/s at 30M) expect
roughly 10k tok/s here, i.e. about 9 hours (an estimate scaled from the v0
log, not benchmarked).

Pass this file to nanoGPT's train.py:
    python train.py ../config/train_chronicle_v1.py

Set `device` below to match your hardware:
    'mps'  — Apple Silicon
    'cuda' — NVIDIA GPU
    'cpu'  — fallback (very slow)

For a fresh training run set init_from = 'scratch'.
To resume from an existing checkpoint set init_from = 'resume'.
"""

# ── Output ────────────────────────────────────────────────────────────────────
out_dir = 'out-chronicle-v1'
dataset = 'australia_1850_1950_v1'

# ── Model — ~51M parameters (25.8M embedding + 25.2M transformer) ─────────────
n_layer  = 8
n_head   = 8
n_embd   = 512
dropout  = 0.1      # kept for multi-epoch training
bias     = False

# ── Context & Batch ───────────────────────────────────────────────────────────
block_size                  = 512   # v0: 256 — longer context for prose and debates
batch_size                  = 16
gradient_accumulation_steps = 8     # effective batch = 128 sequences = 65,536 tokens

# ── Training — ~2 epochs ──────────────────────────────────────────────────────
max_iters      = 5000
warmup_iters   = 200
learning_rate  = 6e-4
lr_decay_iters = 5000
min_lr         = 6e-5
beta1          = 0.9
beta2          = 0.95
weight_decay   = 0.1
grad_clip      = 1.0

# ── Hardware — set to match your system ──────────────────────────────────────
device  = 'mps'    # 'cuda', 'mps', or 'cpu'
dtype   = 'float32'
compile = False    # torch.compile is unsupported on MPS; set True for CUDA

# ── Checkpointing & Logging ───────────────────────────────────────────────────
eval_interval          = 250
eval_iters             = 100
log_interval           = 10
always_save_checkpoint = False   # only save when val loss improves
init_from              = 'scratch'  # 'scratch' for new run, 'resume' to continue
