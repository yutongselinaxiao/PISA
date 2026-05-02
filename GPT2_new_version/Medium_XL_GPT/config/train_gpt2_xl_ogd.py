# GPT-2 XL + canonical OGD on log(sigma).

batch_size = 4
block_size = 1024
gradient_accumulation_steps = 4

n_layer = 48
n_head = 25
n_embd = 1600
dropout = 0.0
bias = False

max_iters = 50000
lr_decay_iters = 100000

eval_interval = 100
eval_iters = 200
log_interval = 10
ckpt_interval = 5000

algorithm = 'sisa'
learning_rate = 1e-4
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0
decay_lr = True
warmup_iters = 0
min_lr = 1e-5

admm_mode = 'linearized'
sigma_lr = 1e3
rho_lr = 1e2

sigma_method = 'ogd'
ogd_eta_u = 0.05
ogd_eta_u_decay = 'textbook_sc'
ogd_G_clip = 10.0
sigma_min_canonical = 1e-3
sigma_max_canonical = 1e6

use_wandb = True
wandb_project = 'gpt2-sisa-canonical-xl'

comment = 'gpt2_xl_ogd_sig1e3'
save_dir = '/dataMeR2/yutong/sisa_gpt2/log_gpt2/' + comment
out_dir = '/dataMeR2/yutong/sisa_gpt2/out_gpt2/' + comment
