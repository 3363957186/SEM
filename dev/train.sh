#!/bin/bash -l

# run this script from adrd_tool/

#conda activate /projectnb/vkolagrp/skowshik/conda_envs/adrd
#/projectnb/vkolagrp/skowshik/conda_envs/adrd/bin/pip install .

# install the package
# cd adrd_tool
# pip install -e .

# define the variables
prefix="."
train_path="./data/train_data_X.csv"
vld_path="./data/test_data_X.csv"


# Note for setting the flags
# 1. If training without MRIs
# img_net="NonImg"
# img_mode = -1
# 2. if training with MRIs
# img_net: [SwinUNETR]
# img_mode = 0
# 3. if training with MRI embeddings
# img_net: [SwinUNETREMB]
# img_mode = 1

# Without using image embeddings
# img_net="NonImg"
# img_mode=-1

# Using image embeddings
#img_net="SwinUNETREMB"
#img_mode=1
img_net="NonImg"
img_mode=-1

# Stage 1
cnf_file="${prefix}/stage_2.toml"
ckpt_path="./dev/ckpt/model_stage_2.pt"

# run train.py
#--fine_tune

python dev/train.py --train_path $train_path --vld_path $vld_path --cnf_file $cnf_file --ckpt_path $ckpt_path --d_model 128 --nhead 1 \
                    --num_epochs 128 --batch_size 256 --lr 1e-3 --gamma 2 --img_mode $img_mode --img_net $img_net --img_size "(182,218,182)"  \
                    --eval_threshold 0.5 --fusion_stage middle --imgnet_layers 2 --weight_decay 0.01 --n_splits 1 --stage 1 --save_intermediate_ckpts --early_stop_threshold 1024 --transfer_epoch 15 --device "cpu" --balanced_sampling
                      #--wandb_project "Project" --wandb --freeze_backbone


# Stage 2
cnf_file="${prefix}/dev/data/toml_files/stage_2_train.toml"
ckpt_path="./dev/ckpt/model_stage_2.pt"

# # run train.py 
# python dev/train.py --train_path $train_path --vld_path $vld_path --cnf_file $cnf_file --ckpt_path $ckpt_path --d_model 256 --nhead 1 \
#                     --num_epochs 128 --batch_size 64 --lr 1e-4 --gamma 2 --img_mode $img_mode --img_net $img_net --img_size "(182,218,182)" \
#                     --fusion_stage middle --imgnet_layers 2 --weight_decay 0.005 --n_splits 1 --stage 2 --save_intermediate_ckpts --early_stop_threshold 30 --fine_tune --stage_1_ckpt "./dev/ckpt/model_stage_1.pt" #--wandb_project "Project" --wandb 