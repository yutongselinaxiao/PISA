# GPT-2 Medium + Task-Aware Online Sigma
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

# eval stuff
eval_interval = 100
eval_iters = 200
log_interval = 10
ckpt_interval = 1000

# optimizer
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

# sigma adaptation
sigma_adapt_mode = 'task_aware'
task_lambda = 1.0

# wandb
use_wandb = True
wandb_project = 'gpt2-sisa-online'

comment = 'gpt2_medium_task_aware_lam1'
save_dir = '/dataMeR2/yutong/sisa_gpt2/log_gpt2/' + comment
out_dir = '/dataMeR2/yutong/sisa_gpt2/out_gpt2/' + comment
