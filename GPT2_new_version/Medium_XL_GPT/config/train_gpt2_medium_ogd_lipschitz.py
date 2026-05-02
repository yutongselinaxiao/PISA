# GPT-2 Medium + canonical OGD with BB Lipschitz floor on log(sigma).
#
# OGD update + hard projection (matches `online_convex_bal_lipschitz_update_u`):
#   target  = log(primal_residual / dual_residual)
#   grad_u  = 2 * (u - target)
#   u_raw   = u - eta_u * grad_u                    # eta_u = 1/(2*k_sigma) by default
#   log_floor = log(L_hat_ema) + log(alpha)
#   u_new   = max(u_raw, log_floor)                  # hard projection
#   u_new   = clamp(u_new, log_min, log_max)
#
# L_hat_ema is the EMA-smoothed Barzilai-Borwein ratio
#   L_hat_raw_t = ‖∇F(z^t) − ∇F(z^{t-1})‖ / ‖z^t − z^{t-1}‖
# updated each train step from the global gradient (Σ across DDP ranks).

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

# Canonical OGD + Lipschitz floor.
sigma_method = 'ogd_lipschitz'
ogd_eta_u = 0.05
ogd_eta_u_decay = 'textbook_sc'
ogd_G_clip = 10.0
sigma_min_canonical = 1e-3
sigma_max_canonical = 1e6
lipschitz_ema_beta = 0.9
lipschitz_window_size = 20
lipschitz_min_dz = 1e-6
lipschitz_max = 1e8
lipschitz_floor_alpha = 1.0

use_wandb = True
wandb_project = 'gpt2-sisa-canonical'

comment = 'gpt2_medium_ogd_lipschitz_sig8e2'
save_dir = '/dataMeR2/yutong/sisa_gpt2/log_gpt2/' + comment
out_dir = '/dataMeR2/yutong/sisa_gpt2/out_gpt2/' + comment
