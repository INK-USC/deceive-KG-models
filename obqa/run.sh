#!/bin/sh

for epoch in 1 10001 20001 30001 40001 50001 60001 70001
do
	echo $epoch && CUDA_VISIBLE_DEVICES=6 python train.py --mode_type eval --num_epochs $epoch --save_dir ./saved_models/KG/model_5 --model_id 1 --debug_mode && cp /ssd/RL_v3/data/cpnet/conceptnet.en.pruned.graph /ssd/RL_perturb_1/data/cpnet && python test_fact_validation.py >> output_RL_2_2.txt && python /ssd/RL_perturb_1/newprocess_1.py >> output_RL_2_2.txt && CUDA_VISIBLE_DEVICES=3 python /ssd/RL_perturb_1/rn_2.py >> output_RL_2_2.txt

done
