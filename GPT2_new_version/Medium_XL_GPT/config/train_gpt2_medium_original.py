# GPT-2 Medium + 'original' SISA (LR-coupled fixed schedule, no residual feedback).
# Use sigma_lr to control the σ_0 anchor for σ_0-robustness sweeps.

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

# ADMM
admm_mode = 'linearized'
sigma_lr = 8e2  # paper-tuned anchor for Medium; sweep via launcher override
rho_lr = 1e2

# Sigma rule: original LR-coupled schedule, no OGD feedback.
sigma_method = 'original'

use_wandb = True
wandb_project = 'gpt2-sisa-canonical'

comment = 'gpt2_medium_original_sig8e2'
save_dir = '/dataMeR2/yutong/sisa_gpt2/log_gpt2/' + comment
out_dir = '/dataMeR2/yutong/sisa_gpt2/out_gpt2/' + comment
