#!/bin/bash

set -e
set -o pipefail

# --- gpt2_medium_original_sig8e1_seed1337 ---
mkdir -p $(dirname generated_canonical_gpt2/logs/gpt2_medium_original_sig8e1_seed1337.log)
echo "[$(date)] launching gpt2_medium_original_sig8e1_seed1337"
(CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 \
    train_gpt_sisa_lower_no_2ndgradient_online.py \
    config/train_gpt2_medium_original.py \
    --sigma_lr=8e1 \
    --seed=1337 \
    --comment=gpt2_medium_original_sig8e1_seed1337 \
    --save_dir=/dataMeR2/yutong/sisa_gpt2/log_gpt2/gpt2_medium_original_sig8e1_seed1337 \
    --out_dir=/dataMeR2/yutong/sisa_gpt2/out_gpt2/gpt2_medium_original_sig8e1_seed1337 \
    --wandb_run_name=gpt2_medium_original_sig8e1_seed1337) 2>&1 | tee generated_canonical_gpt2/logs/gpt2_medium_original_sig8e1_seed1337.log

# --- gpt2_medium_original_sig8e1_seed1338 ---
mkdir -p $(dirname generated_canonical_gpt2/logs/gpt2_medium_original_sig8e1_seed1338.log)
echo "[$(date)] launching gpt2_medium_original_sig8e1_seed1338"
(CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 \
    train_gpt_sisa_lower_no_2ndgradient_online.py \
    config/train_gpt2_medium_original.py \
    --sigma_lr=8e1 \
    --seed=1338 \
    --comment=gpt2_medium_original_sig8e1_seed1338 \
    --save_dir=/dataMeR2/yutong/sisa_gpt2/log_gpt2/gpt2_medium_original_sig8e1_seed1338 \
    --out_dir=/dataMeR2/yutong/sisa_gpt2/out_gpt2/gpt2_medium_original_sig8e1_seed1338 \
    --wandb_run_name=gpt2_medium_original_sig8e1_seed1338) 2>&1 | tee generated_canonical_gpt2/logs/gpt2_medium_original_sig8e1_seed1338.log

# --- gpt2_medium_original_sig8e1_seed1339 ---
mkdir -p $(dirname generated_canonical_gpt2/logs/gpt2_medium_original_sig8e1_seed1339.log)
echo "[$(date)] launching gpt2_medium_original_sig8e1_seed1339"
(CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 \
    train_gpt_sisa_lower_no_2ndgradient_online.py \
    config/train_gpt2_medium_original.py \
    --sigma_lr=8e1 \
    --seed=1339 \
    --comment=gpt2_medium_original_sig8e1_seed1339 \
    --save_dir=/dataMeR2/yutong/sisa_gpt2/log_gpt2/gpt2_medium_original_sig8e1_seed1339 \
    --out_dir=/dataMeR2/yutong/sisa_gpt2/out_gpt2/gpt2_medium_original_sig8e1_seed1339 \
    --wandb_run_name=gpt2_medium_original_sig8e1_seed1339) 2>&1 | tee generated_canonical_gpt2/logs/gpt2_medium_original_sig8e1_seed1339.log

# --- gpt2_medium_original_sig8e2_seed1337 ---
mkdir -p $(dirname generated_canonical_gpt2/logs/gpt2_medium_original_sig8e2_seed1337.log)
echo "[$(date)] launching gpt2_medium_original_sig8e2_seed1337"
(CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 \
    train_gpt_sisa_lower_no_2ndgradient_online.py \
    config/train_gpt2_medium_original.py \
    --sigma_lr=8e2 \
    --seed=1337 \
    --comment=gpt2_medium_original_sig8e2_seed1337 \
    --save_dir=/dataMeR2/yutong/sisa_gpt2/log_gpt2/gpt2_medium_original_sig8e2_seed1337 \
    --out_dir=/dataMeR2/yutong/sisa_gpt2/out_gpt2/gpt2_medium_original_sig8e2_seed1337 \
    --wandb_run_name=gpt2_medium_original_sig8e2_seed1337) 2>&1 | tee generated_canonical_gpt2/logs/gpt2_medium_original_sig8e2_seed1337.log

# --- gpt2_medium_original_sig8e2_seed1338 ---
mkdir -p $(dirname generated_canonical_gpt2/logs/gpt2_medium_original_sig8e2_seed1338.log)
echo "[$(date)] launching gpt2_medium_original_sig8e2_seed1338"
(CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 \
    train_gpt_sisa_lower_no_2ndgradient_online.py \
    config/train_gpt2_medium_original.py \
    --sigma_lr=8e2 \
    --seed=1338 \
    --comment=gpt2_medium_original_sig8e2_seed1338 \
    --save_dir=/dataMeR2/yutong/sisa_gpt2/log_gpt2/gpt2_medium_original_sig8e2_seed1338 \
    --out_dir=/dataMeR2/yutong/sisa_gpt2/out_gpt2/gpt2_medium_original_sig8e2_seed1338 \
    --wandb_run_name=gpt2_medium_original_sig8e2_seed1338) 2>&1 | tee generated_canonical_gpt2/logs/gpt2_medium_original_sig8e2_seed1338.log

# --- gpt2_medium_original_sig8e2_seed1339 ---
mkdir -p $(dirname generated_canonical_gpt2/logs/gpt2_medium_original_sig8e2_seed1339.log)
echo "[$(date)] launching gpt2_medium_original_sig8e2_seed1339"
(CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 \
    train_gpt_sisa_lower_no_2ndgradient_online.py \
    config/train_gpt2_medium_original.py \
    --sigma_lr=8e2 \
    --seed=1339 \
    --comment=gpt2_medium_original_sig8e2_seed1339 \
    --save_dir=/dataMeR2/yutong/sisa_gpt2/log_gpt2/gpt2_medium_original_sig8e2_seed1339 \
    --out_dir=/dataMeR2/yutong/sisa_gpt2/out_gpt2/gpt2_medium_original_sig8e2_seed1339 \
    --wandb_run_name=gpt2_medium_original_sig8e2_seed1339) 2>&1 | tee generated_canonical_gpt2/logs/gpt2_medium_original_sig8e2_seed1339.log

# --- gpt2_medium_original_sig8e3_seed1337 ---
mkdir -p $(dirname generated_canonical_gpt2/logs/gpt2_medium_original_sig8e3_seed1337.log)
echo "[$(date)] launching gpt2_medium_original_sig8e3_seed1337"
(CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 \
    train_gpt_sisa_lower_no_2ndgradient_online.py \
    config/train_gpt2_medium_original.py \
    --sigma_lr=8e3 \
    --seed=1337 \
    --comment=gpt2_medium_original_sig8e3_seed1337 \
    --save_dir=/dataMeR2/yutong/sisa_gpt2/log_gpt2/gpt2_medium_original_sig8e3_seed1337 \
    --out_dir=/dataMeR2/yutong/sisa_gpt2/out_gpt2/gpt2_medium_original_sig8e3_seed1337 \
    --wandb_run_name=gpt2_medium_original_sig8e3_seed1337) 2>&1 | tee generated_canonical_gpt2/logs/gpt2_medium_original_sig8e3_seed1337.log

# --- gpt2_medium_original_sig8e3_seed1338 ---
mkdir -p $(dirname generated_canonical_gpt2/logs/gpt2_medium_original_sig8e3_seed1338.log)
echo "[$(date)] launching gpt2_medium_original_sig8e3_seed1338"
(CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 \
    train_gpt_sisa_lower_no_2ndgradient_online.py \
    config/train_gpt2_medium_original.py \
    --sigma_lr=8e3 \
    --seed=1338 \
    --comment=gpt2_medium_original_sig8e3_seed1338 \
    --save_dir=/dataMeR2/yutong/sisa_gpt2/log_gpt2/gpt2_medium_original_sig8e3_seed1338 \
    --out_dir=/dataMeR2/yutong/sisa_gpt2/out_gpt2/gpt2_medium_original_sig8e3_seed1338 \
    --wandb_run_name=gpt2_medium_original_sig8e3_seed1338) 2>&1 | tee generated_canonical_gpt2/logs/gpt2_medium_original_sig8e3_seed1338.log

# --- gpt2_medium_original_sig8e3_seed1339 ---
mkdir -p $(dirname generated_canonical_gpt2/logs/gpt2_medium_original_sig8e3_seed1339.log)
echo "[$(date)] launching gpt2_medium_original_sig8e3_seed1339"
(CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 \
    train_gpt_sisa_lower_no_2ndgradient_online.py \
    config/train_gpt2_medium_original.py \
    --sigma_lr=8e3 \
    --seed=1339 \
    --comment=gpt2_medium_original_sig8e3_seed1339 \
    --save_dir=/dataMeR2/yutong/sisa_gpt2/log_gpt2/gpt2_medium_original_sig8e3_seed1339 \
    --out_dir=/dataMeR2/yutong/sisa_gpt2/out_gpt2/gpt2_medium_original_sig8e3_seed1339 \
    --wandb_run_name=gpt2_medium_original_sig8e3_seed1339) 2>&1 | tee generated_canonical_gpt2/logs/gpt2_medium_original_sig8e3_seed1339.log

# --- gpt2_medium_ogd_sig8e1_seed1337 ---
mkdir -p $(dirname generated_canonical_gpt2/logs/gpt2_medium_ogd_sig8e1_seed1337.log)
echo "[$(date)] launching gpt2_medium_ogd_sig8e1_seed1337"
(CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 \
    train_gpt_sisa_lower_no_2ndgradient_online.py \
    config/train_gpt2_medium_ogd.py \
    --sigma_lr=8e1 \
    --seed=1337 \
    --comment=gpt2_medium_ogd_sig8e1_seed1337 \
    --save_dir=/dataMeR2/yutong/sisa_gpt2/log_gpt2/gpt2_medium_ogd_sig8e1_seed1337 \
    --out_dir=/dataMeR2/yutong/sisa_gpt2/out_gpt2/gpt2_medium_ogd_sig8e1_seed1337 \
    --wandb_run_name=gpt2_medium_ogd_sig8e1_seed1337) 2>&1 | tee generated_canonical_gpt2/logs/gpt2_medium_ogd_sig8e1_seed1337.log

# --- gpt2_medium_ogd_sig8e1_seed1338 ---
mkdir -p $(dirname generated_canonical_gpt2/logs/gpt2_medium_ogd_sig8e1_seed1338.log)
echo "[$(date)] launching gpt2_medium_ogd_sig8e1_seed1338"
(CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 \
    train_gpt_sisa_lower_no_2ndgradient_online.py \
    config/train_gpt2_medium_ogd.py \
    --sigma_lr=8e1 \
    --seed=1338 \
    --comment=gpt2_medium_ogd_sig8e1_seed1338 \
    --save_dir=/dataMeR2/yutong/sisa_gpt2/log_gpt2/gpt2_medium_ogd_sig8e1_seed1338 \
    --out_dir=/dataMeR2/yutong/sisa_gpt2/out_gpt2/gpt2_medium_ogd_sig8e1_seed1338 \
    --wandb_run_name=gpt2_medium_ogd_sig8e1_seed1338) 2>&1 | tee generated_canonical_gpt2/logs/gpt2_medium_ogd_sig8e1_seed1338.log

# --- gpt2_medium_ogd_sig8e1_seed1339 ---
mkdir -p $(dirname generated_canonical_gpt2/logs/gpt2_medium_ogd_sig8e1_seed1339.log)
echo "[$(date)] launching gpt2_medium_ogd_sig8e1_seed1339"
(CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 \
    train_gpt_sisa_lower_no_2ndgradient_online.py \
    config/train_gpt2_medium_ogd.py \
    --sigma_lr=8e1 \
    --seed=1339 \
    --comment=gpt2_medium_ogd_sig8e1_seed1339 \
    --save_dir=/dataMeR2/yutong/sisa_gpt2/log_gpt2/gpt2_medium_ogd_sig8e1_seed1339 \
    --out_dir=/dataMeR2/yutong/sisa_gpt2/out_gpt2/gpt2_medium_ogd_sig8e1_seed1339 \
    --wandb_run_name=gpt2_medium_ogd_sig8e1_seed1339) 2>&1 | tee generated_canonical_gpt2/logs/gpt2_medium_ogd_sig8e1_seed1339.log

# --- gpt2_medium_ogd_sig8e2_seed1337 ---
mkdir -p $(dirname generated_canonical_gpt2/logs/gpt2_medium_ogd_sig8e2_seed1337.log)
echo "[$(date)] launching gpt2_medium_ogd_sig8e2_seed1337"
(CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 \
    train_gpt_sisa_lower_no_2ndgradient_online.py \
    config/train_gpt2_medium_ogd.py \
    --sigma_lr=8e2 \
    --seed=1337 \
    --comment=gpt2_medium_ogd_sig8e2_seed1337 \
    --save_dir=/dataMeR2/yutong/sisa_gpt2/log_gpt2/gpt2_medium_ogd_sig8e2_seed1337 \
    --out_dir=/dataMeR2/yutong/sisa_gpt2/out_gpt2/gpt2_medium_ogd_sig8e2_seed1337 \
    --wandb_run_name=gpt2_medium_ogd_sig8e2_seed1337) 2>&1 | tee generated_canonical_gpt2/logs/gpt2_medium_ogd_sig8e2_seed1337.log

# --- gpt2_medium_ogd_sig8e2_seed1338 ---
mkdir -p $(dirname generated_canonical_gpt2/logs/gpt2_medium_ogd_sig8e2_seed1338.log)
echo "[$(date)] launching gpt2_medium_ogd_sig8e2_seed1338"
(CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 \
    train_gpt_sisa_lower_no_2ndgradient_online.py \
    config/train_gpt2_medium_ogd.py \
    --sigma_lr=8e2 \
    --seed=1338 \
    --comment=gpt2_medium_ogd_sig8e2_seed1338 \
    --save_dir=/dataMeR2/yutong/sisa_gpt2/log_gpt2/gpt2_medium_ogd_sig8e2_seed1338 \
    --out_dir=/dataMeR2/yutong/sisa_gpt2/out_gpt2/gpt2_medium_ogd_sig8e2_seed1338 \
    --wandb_run_name=gpt2_medium_ogd_sig8e2_seed1338) 2>&1 | tee generated_canonical_gpt2/logs/gpt2_medium_ogd_sig8e2_seed1338.log

# --- gpt2_medium_ogd_sig8e2_seed1339 ---
mkdir -p $(dirname generated_canonical_gpt2/logs/gpt2_medium_ogd_sig8e2_seed1339.log)
echo "[$(date)] launching gpt2_medium_ogd_sig8e2_seed1339"
(CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 \
    train_gpt_sisa_lower_no_2ndgradient_online.py \
    config/train_gpt2_medium_ogd.py \
    --sigma_lr=8e2 \
    --seed=1339 \
    --comment=gpt2_medium_ogd_sig8e2_seed1339 \
    --save_dir=/dataMeR2/yutong/sisa_gpt2/log_gpt2/gpt2_medium_ogd_sig8e2_seed1339 \
    --out_dir=/dataMeR2/yutong/sisa_gpt2/out_gpt2/gpt2_medium_ogd_sig8e2_seed1339 \
    --wandb_run_name=gpt2_medium_ogd_sig8e2_seed1339) 2>&1 | tee generated_canonical_gpt2/logs/gpt2_medium_ogd_sig8e2_seed1339.log

# --- gpt2_medium_ogd_sig8e3_seed1337 ---
mkdir -p $(dirname generated_canonical_gpt2/logs/gpt2_medium_ogd_sig8e3_seed1337.log)
echo "[$(date)] launching gpt2_medium_ogd_sig8e3_seed1337"
(CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 \
    train_gpt_sisa_lower_no_2ndgradient_online.py \
    config/train_gpt2_medium_ogd.py \
    --sigma_lr=8e3 \
    --seed=1337 \
    --comment=gpt2_medium_ogd_sig8e3_seed1337 \
    --save_dir=/dataMeR2/yutong/sisa_gpt2/log_gpt2/gpt2_medium_ogd_sig8e3_seed1337 \
    --out_dir=/dataMeR2/yutong/sisa_gpt2/out_gpt2/gpt2_medium_ogd_sig8e3_seed1337 \
    --wandb_run_name=gpt2_medium_ogd_sig8e3_seed1337) 2>&1 | tee generated_canonical_gpt2/logs/gpt2_medium_ogd_sig8e3_seed1337.log

# --- gpt2_medium_ogd_sig8e3_seed1338 ---
mkdir -p $(dirname generated_canonical_gpt2/logs/gpt2_medium_ogd_sig8e3_seed1338.log)
echo "[$(date)] launching gpt2_medium_ogd_sig8e3_seed1338"
(CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 \
    train_gpt_sisa_lower_no_2ndgradient_online.py \
    config/train_gpt2_medium_ogd.py \
    --sigma_lr=8e3 \
    --seed=1338 \
    --comment=gpt2_medium_ogd_sig8e3_seed1338 \
    --save_dir=/dataMeR2/yutong/sisa_gpt2/log_gpt2/gpt2_medium_ogd_sig8e3_seed1338 \
    --out_dir=/dataMeR2/yutong/sisa_gpt2/out_gpt2/gpt2_medium_ogd_sig8e3_seed1338 \
    --wandb_run_name=gpt2_medium_ogd_sig8e3_seed1338) 2>&1 | tee generated_canonical_gpt2/logs/gpt2_medium_ogd_sig8e3_seed1338.log

# --- gpt2_medium_ogd_sig8e3_seed1339 ---
mkdir -p $(dirname generated_canonical_gpt2/logs/gpt2_medium_ogd_sig8e3_seed1339.log)
echo "[$(date)] launching gpt2_medium_ogd_sig8e3_seed1339"
(CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 \
    train_gpt_sisa_lower_no_2ndgradient_online.py \
    config/train_gpt2_medium_ogd.py \
    --sigma_lr=8e3 \
    --seed=1339 \
    --comment=gpt2_medium_ogd_sig8e3_seed1339 \
    --save_dir=/dataMeR2/yutong/sisa_gpt2/log_gpt2/gpt2_medium_ogd_sig8e3_seed1339 \
    --out_dir=/dataMeR2/yutong/sisa_gpt2/out_gpt2/gpt2_medium_ogd_sig8e3_seed1339 \
    --wandb_run_name=gpt2_medium_ogd_sig8e3_seed1339) 2>&1 | tee generated_canonical_gpt2/logs/gpt2_medium_ogd_sig8e3_seed1339.log

# --- gpt2_medium_ogd_lipschitz_sig8e1_seed1337 ---
mkdir -p $(dirname generated_canonical_gpt2/logs/gpt2_medium_ogd_lipschitz_sig8e1_seed1337.log)
echo "[$(date)] launching gpt2_medium_ogd_lipschitz_sig8e1_seed1337"
(CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 \
    train_gpt_sisa_lower_no_2ndgradient_online.py \
    config/train_gpt2_medium_ogd_lipschitz.py \
    --sigma_lr=8e1 \
    --seed=1337 \
    --comment=gpt2_medium_ogd_lipschitz_sig8e1_seed1337 \
    --save_dir=/dataMeR2/yutong/sisa_gpt2/log_gpt2/gpt2_medium_ogd_lipschitz_sig8e1_seed1337 \
    --out_dir=/dataMeR2/yutong/sisa_gpt2/out_gpt2/gpt2_medium_ogd_lipschitz_sig8e1_seed1337 \
    --wandb_run_name=gpt2_medium_ogd_lipschitz_sig8e1_seed1337) 2>&1 | tee generated_canonical_gpt2/logs/gpt2_medium_ogd_lipschitz_sig8e1_seed1337.log

# --- gpt2_medium_ogd_lipschitz_sig8e1_seed1338 ---
mkdir -p $(dirname generated_canonical_gpt2/logs/gpt2_medium_ogd_lipschitz_sig8e1_seed1338.log)
echo "[$(date)] launching gpt2_medium_ogd_lipschitz_sig8e1_seed1338"
(CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 \
    train_gpt_sisa_lower_no_2ndgradient_online.py \
    config/train_gpt2_medium_ogd_lipschitz.py \
    --sigma_lr=8e1 \
    --seed=1338 \
    --comment=gpt2_medium_ogd_lipschitz_sig8e1_seed1338 \
    --save_dir=/dataMeR2/yutong/sisa_gpt2/log_gpt2/gpt2_medium_ogd_lipschitz_sig8e1_seed1338 \
    --out_dir=/dataMeR2/yutong/sisa_gpt2/out_gpt2/gpt2_medium_ogd_lipschitz_sig8e1_seed1338 \
    --wandb_run_name=gpt2_medium_ogd_lipschitz_sig8e1_seed1338) 2>&1 | tee generated_canonical_gpt2/logs/gpt2_medium_ogd_lipschitz_sig8e1_seed1338.log

# --- gpt2_medium_ogd_lipschitz_sig8e1_seed1339 ---
mkdir -p $(dirname generated_canonical_gpt2/logs/gpt2_medium_ogd_lipschitz_sig8e1_seed1339.log)
echo "[$(date)] launching gpt2_medium_ogd_lipschitz_sig8e1_seed1339"
(CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 \
    train_gpt_sisa_lower_no_2ndgradient_online.py \
    config/train_gpt2_medium_ogd_lipschitz.py \
    --sigma_lr=8e1 \
    --seed=1339 \
    --comment=gpt2_medium_ogd_lipschitz_sig8e1_seed1339 \
    --save_dir=/dataMeR2/yutong/sisa_gpt2/log_gpt2/gpt2_medium_ogd_lipschitz_sig8e1_seed1339 \
    --out_dir=/dataMeR2/yutong/sisa_gpt2/out_gpt2/gpt2_medium_ogd_lipschitz_sig8e1_seed1339 \
    --wandb_run_name=gpt2_medium_ogd_lipschitz_sig8e1_seed1339) 2>&1 | tee generated_canonical_gpt2/logs/gpt2_medium_ogd_lipschitz_sig8e1_seed1339.log

# --- gpt2_medium_ogd_lipschitz_sig8e2_seed1337 ---
mkdir -p $(dirname generated_canonical_gpt2/logs/gpt2_medium_ogd_lipschitz_sig8e2_seed1337.log)
echo "[$(date)] launching gpt2_medium_ogd_lipschitz_sig8e2_seed1337"
(CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 \
    train_gpt_sisa_lower_no_2ndgradient_online.py \
    config/train_gpt2_medium_ogd_lipschitz.py \
    --sigma_lr=8e2 \
    --seed=1337 \
    --comment=gpt2_medium_ogd_lipschitz_sig8e2_seed1337 \
    --save_dir=/dataMeR2/yutong/sisa_gpt2/log_gpt2/gpt2_medium_ogd_lipschitz_sig8e2_seed1337 \
    --out_dir=/dataMeR2/yutong/sisa_gpt2/out_gpt2/gpt2_medium_ogd_lipschitz_sig8e2_seed1337 \
    --wandb_run_name=gpt2_medium_ogd_lipschitz_sig8e2_seed1337) 2>&1 | tee generated_canonical_gpt2/logs/gpt2_medium_ogd_lipschitz_sig8e2_seed1337.log

# --- gpt2_medium_ogd_lipschitz_sig8e2_seed1338 ---
mkdir -p $(dirname generated_canonical_gpt2/logs/gpt2_medium_ogd_lipschitz_sig8e2_seed1338.log)
echo "[$(date)] launching gpt2_medium_ogd_lipschitz_sig8e2_seed1338"
(CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 \
    train_gpt_sisa_lower_no_2ndgradient_online.py \
    config/train_gpt2_medium_ogd_lipschitz.py \
    --sigma_lr=8e2 \
    --seed=1338 \
    --comment=gpt2_medium_ogd_lipschitz_sig8e2_seed1338 \
    --save_dir=/dataMeR2/yutong/sisa_gpt2/log_gpt2/gpt2_medium_ogd_lipschitz_sig8e2_seed1338 \
    --out_dir=/dataMeR2/yutong/sisa_gpt2/out_gpt2/gpt2_medium_ogd_lipschitz_sig8e2_seed1338 \
    --wandb_run_name=gpt2_medium_ogd_lipschitz_sig8e2_seed1338) 2>&1 | tee generated_canonical_gpt2/logs/gpt2_medium_ogd_lipschitz_sig8e2_seed1338.log

# --- gpt2_medium_ogd_lipschitz_sig8e2_seed1339 ---
mkdir -p $(dirname generated_canonical_gpt2/logs/gpt2_medium_ogd_lipschitz_sig8e2_seed1339.log)
echo "[$(date)] launching gpt2_medium_ogd_lipschitz_sig8e2_seed1339"
(CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 \
    train_gpt_sisa_lower_no_2ndgradient_online.py \
    config/train_gpt2_medium_ogd_lipschitz.py \
    --sigma_lr=8e2 \
    --seed=1339 \
    --comment=gpt2_medium_ogd_lipschitz_sig8e2_seed1339 \
    --save_dir=/dataMeR2/yutong/sisa_gpt2/log_gpt2/gpt2_medium_ogd_lipschitz_sig8e2_seed1339 \
    --out_dir=/dataMeR2/yutong/sisa_gpt2/out_gpt2/gpt2_medium_ogd_lipschitz_sig8e2_seed1339 \
    --wandb_run_name=gpt2_medium_ogd_lipschitz_sig8e2_seed1339) 2>&1 | tee generated_canonical_gpt2/logs/gpt2_medium_ogd_lipschitz_sig8e2_seed1339.log

# --- gpt2_medium_ogd_lipschitz_sig8e3_seed1337 ---
mkdir -p $(dirname generated_canonical_gpt2/logs/gpt2_medium_ogd_lipschitz_sig8e3_seed1337.log)
echo "[$(date)] launching gpt2_medium_ogd_lipschitz_sig8e3_seed1337"
(CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 \
    train_gpt_sisa_lower_no_2ndgradient_online.py \
    config/train_gpt2_medium_ogd_lipschitz.py \
    --sigma_lr=8e3 \
    --seed=1337 \
    --comment=gpt2_medium_ogd_lipschitz_sig8e3_seed1337 \
    --save_dir=/dataMeR2/yutong/sisa_gpt2/log_gpt2/gpt2_medium_ogd_lipschitz_sig8e3_seed1337 \
    --out_dir=/dataMeR2/yutong/sisa_gpt2/out_gpt2/gpt2_medium_ogd_lipschitz_sig8e3_seed1337 \
    --wandb_run_name=gpt2_medium_ogd_lipschitz_sig8e3_seed1337) 2>&1 | tee generated_canonical_gpt2/logs/gpt2_medium_ogd_lipschitz_sig8e3_seed1337.log

# --- gpt2_medium_ogd_lipschitz_sig8e3_seed1338 ---
mkdir -p $(dirname generated_canonical_gpt2/logs/gpt2_medium_ogd_lipschitz_sig8e3_seed1338.log)
echo "[$(date)] launching gpt2_medium_ogd_lipschitz_sig8e3_seed1338"
(CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 \
    train_gpt_sisa_lower_no_2ndgradient_online.py \
    config/train_gpt2_medium_ogd_lipschitz.py \
    --sigma_lr=8e3 \
    --seed=1338 \
    --comment=gpt2_medium_ogd_lipschitz_sig8e3_seed1338 \
    --save_dir=/dataMeR2/yutong/sisa_gpt2/log_gpt2/gpt2_medium_ogd_lipschitz_sig8e3_seed1338 \
    --out_dir=/dataMeR2/yutong/sisa_gpt2/out_gpt2/gpt2_medium_ogd_lipschitz_sig8e3_seed1338 \
    --wandb_run_name=gpt2_medium_ogd_lipschitz_sig8e3_seed1338) 2>&1 | tee generated_canonical_gpt2/logs/gpt2_medium_ogd_lipschitz_sig8e3_seed1338.log

# --- gpt2_medium_ogd_lipschitz_sig8e3_seed1339 ---
mkdir -p $(dirname generated_canonical_gpt2/logs/gpt2_medium_ogd_lipschitz_sig8e3_seed1339.log)
echo "[$(date)] launching gpt2_medium_ogd_lipschitz_sig8e3_seed1339"
(CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 \
    train_gpt_sisa_lower_no_2ndgradient_online.py \
    config/train_gpt2_medium_ogd_lipschitz.py \
    --sigma_lr=8e3 \
    --seed=1339 \
    --comment=gpt2_medium_ogd_lipschitz_sig8e3_seed1339 \
    --save_dir=/dataMeR2/yutong/sisa_gpt2/log_gpt2/gpt2_medium_ogd_lipschitz_sig8e3_seed1339 \
    --out_dir=/dataMeR2/yutong/sisa_gpt2/out_gpt2/gpt2_medium_ogd_lipschitz_sig8e3_seed1339 \
    --wandb_run_name=gpt2_medium_ogd_lipschitz_sig8e3_seed1339) 2>&1 | tee generated_canonical_gpt2/logs/gpt2_medium_ogd_lipschitz_sig8e3_seed1339.log
