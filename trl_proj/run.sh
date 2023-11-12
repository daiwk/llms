#!/bin/bash

function prepare()
{

	rm -rf std/
	mkdir -p std/datasets/
	python3 prepare_models.py
}


function train_rl()
{
	python3 sft.py \
		--model_name="./std/models/opt-350m"  \
		--dataset_name="./std/datasets/openassistant-guanaco"

	task_name=sentiment-analysis
	python3 ppo.py \
		--ppo_config.model_name="./std/models/gpt2-imdb" \
		--ppo_config.reward_model=${task_name}:"./std/models/distilbert-imdb" \
		--ppo_config.query_dataset="./std/datasets/imdb"
	#accelerate launch --config_file=./multi_gpu.yaml --num_processes ${nproc_per_node} --main_process_port $METIS_WORKER_0_PORT trl_train_with_sft.py
	return 0

}

#prepare
train_rl


