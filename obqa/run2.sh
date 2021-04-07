#!/bin/sh

for epoch in 30001 40001 50001 60001
do
	echo $epoch
	cd /ssd/RL_v3;
	echo -e "\n\n\nepoch="$epoch"\n\n\n" >> output_RL_2_2.txt &&
		CUDA_VISIBLE_DEVICES=0 python train.py --mode_type eval --num_epochs 70001 --num_steps $epoch --save_dir ./saved_models/KG/model_5 --model_id 1 --debug_mode &&
		cp /ssd/RL_v3/data/cpnet/conceptnet.en.pruned.graph /ssd/RL_perturb_1/data/cpnet &&
		CUDA_VISIBLE_DEVICES=0 python test_fact_validation.py 
		
done
