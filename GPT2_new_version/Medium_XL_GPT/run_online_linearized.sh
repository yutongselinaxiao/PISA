CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 --master_port 12346 train_gpt_sisa_lower_no_2ndgradient_online.py config/train_gpt2_medium_online_linearized.py
