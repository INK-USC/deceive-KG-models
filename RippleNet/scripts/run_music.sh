#!/bin/bash

dataset="music"
dim=200
n_negative=1
lr=1e-3
weight_decay=1e-2
batch_size=128
n_epoch=100
params="e${dim}_n${n_negative}_l${lr}_w${weight_decay}_b${batch_size}_i${n_epoch}"

save_dir="./checkpoints/${dataset}/${params}"
mkdir -p $save_dir

nohup python -u main.py \
	--dataset $dataset \
	--save_dir $save_dir \
	--dim $dim \
	--n_negative $n_negative \
	--lr $lr \
	--weight_decay $weight_decay \
	--batch_size $batch_size \
	--n_epoch $n_epoch \
    --gpu $1 \
    > ./${save_dir}/train.log 2>&1 &
