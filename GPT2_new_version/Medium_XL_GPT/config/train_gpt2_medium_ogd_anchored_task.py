# GPT-2 Medium + anchored OGD + loss-increase task term (option 2 from the
# 2026-05-15 design discussion).
#
# Motivation: on DDP-homogeneous GPT-2 training the canonical primal residual
# is tiny, so the residual-balance gradient carries no σ-informative signal
# and the canonical OGD update (`ogd`) collapses to a near-no-op around its
# initial point. `ogd_anchored_canonical` partially fixes this by anchoring to
# the LR-coupled base_u, but the task signal — actual training-loss change —
# is the only thing on DDP-homogeneous training that is reliably informative
# about whether the current σ is too small (instability → loss up) or fine.
#
# Update rule (per step):
#   imbalance      = log(primal_canonical_ema / dual_ema)
#   base_u(t)      = log(σ_LR-coupled(t))                       # anchor
#   target_u       = base_u + η_k · imbalance                    # anchored target
#   diff           = u − target_u
#   Δloss          = max(0, loss_t − loss_{t-1})                 # task signal
#   grad_u         = 2·diff + task_lambda · Δloss · sign(diff)   # combined
#   u_raw          = u − η_k · clip(grad_u, ±G_clip)
#   u_new          = blend toward u_raw, trust-region clamp, σ ∈ [σ_min, σ_max]
#
# Properties:
#   - On steady descent (Δloss ≤ 0) the task term vanishes → reduces exactly
#     to `ogd_anchored` (proven equivalent for that regime).
#   - On loss spikes (Δloss > 0) the task term ADDS its gradient on top, so
#     σ moves faster toward target_u — i.e. back toward LR-coupled. This is
#     a "back off when things are bad" signal that's informative even when
#     residuals aren't.
#   - Theoretical regret bound: same as `online_task_aware_update_u` in
#     _online.py, since the loss is convex in u (residual quadratic + task
#     ReLU·|diff| is convex; subgradient is bounded by G_clip).
#
# Hypothesis vs. `original` on GPT-2:
#   - Should match (within noise) on the σ_0=8e2 anchor cell, because the
#     LR-coupled schedule already converges cleanly there.
#   - At off-anchor σ_0 (1e2 / 8e3): off-anchor `ogd_anchored_canonical` saw
#     σ converge to 1627 (vs 7956 for LR-coupled at σ_0=8e3) — empirically
#     worse. The task term should improve that off-anchor case if loss
#     spikes are happening and they're not being responded to.

batch_size = 16
block_size = 1024
gradient_accumulation_steps = 1

n_layer = 24
n_head = 16
n_embd = 1024
dropout = 0.0
bias = False

max_iters = 50000
lr_decay_iters = 100000

eval_interval = 100
eval_iters = 200
log_interval = 10
ckpt_interval = 1000

algorithm = 'sisa'
learning_rate = 0.02
embed_learning_rate = 0.0036
muon_learning_rate = 0.02
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
epsilon = 1e-6
grad_clip = 1.0
decay_lr = True
min_lr = 6e-5

admm_mode = 'linearized'
sigma_lr = 8e2
rho_lr = 1e2

sigma_method = 'ogd_anchored_task'
anchored_residual_source = 'canonical'  # Boyd RMS residual
ogd_eta_u = 0.05
ogd_eta_u_decay = 'textbook_sc'
ogd_G_clip = 10.0
sigma_min_canonical = 0.1
sigma_max_canonical = 1e6
task_lambda = 1.0  # weight on loss-increase term; same default as _online.py

use_wandb = True
wandb_project = 'gpt2-sisa-canonical'

comment = 'gpt2_medium_ogd_anchored_task_sig8e2'
save_dir = '/dataMeR2/yutong/sisa_gpt2/log_gpt2/' + comment
out_dir = '/dataMeR2/yutong/sisa_gpt2/out_gpt2/' + comment
