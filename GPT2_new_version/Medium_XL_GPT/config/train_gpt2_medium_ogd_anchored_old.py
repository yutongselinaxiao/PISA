# GPT-2 Medium + anchored OGD on log(σ) with OLD non-canonical residual.
#
# Identical algorithm to train_gpt2_medium_ogd_anchored_canonical.py BUT uses
# the OLD pre-2026-05-02 residual aggregation `sqrt(Σ_layer ‖α_i·w_i − w_g‖²)`
# per rank (no all-reduce). On DDP-homogeneous training this residual is
# artificially inflated by ‖(α_i−1)·w‖ (parameter-norm scaling), which the old
# anchored convex_bal exploited to land σ in a productive range. This config
# reproduces the historical `gpt2_medium_online_linearized-800` result
# (val=3.74 at σ_0=800) under the new code path.

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

sigma_method = 'ogd_anchored'
anchored_residual_source = 'old'  # OLD non-canonical per-rank residual
ogd_eta_u = 0.05
ogd_eta_u_decay = 'textbook_sc'
ogd_G_clip = 10.0
sigma_min_canonical = 0.1
sigma_max_canonical = 1e6

use_wandb = True
wandb_project = 'gpt2-sisa-canonical'

comment = 'gpt2_medium_ogd_anchored_old_sig8e2'
save_dir = '/dataMeR2/yutong/sisa_gpt2/log_gpt2/' + comment
out_dir = '/dataMeR2/yutong/sisa_gpt2/out_gpt2/' + comment
