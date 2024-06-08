#!/bin/bash

python finetune.py --model qwen-4 --data_category abstract --per_device_train_batch_size 2 --gradient_accumulation_steps 1
CUDA_VISIBLE_DEVICES=0,1,2,3 python finetune.py --model qwen-4 --data_category contextualized --per_device_train_batch_size 1 --gradient_accumulation_steps 2 --epochs 1
CUDA_VISIBLE_DEVICES=0,1,2,3 python finetune.py --model qwen-4 --data_category best_domains --per_device_train_batch_size 1 --gradient_accumulation_steps 2 --epochs 1
CUDA_VISIBLE_DEVICES=0,1,2,3 python finetune.py --model qwen-4 --data_category worst_domains --per_device_train_batch_size 1 --gradient_accumulation_steps 2 --epochs 1
CUDA_VISIBLE_DEVICES=0,1,2,3 python finetune.py --model qwen-4 --data_category partial --partial_sample 1 --per_device_train_batch_size 1 --gradient_accumulation_steps 2 --epochs 1
CUDA_VISIBLE_DEVICES=0,1,2,3 python finetune.py --model qwen-4 --data_category partial --partial_sample 2 --per_device_train_batch_size 1 --gradient_accumulation_steps 2 --epochs 1
CUDA_VISIBLE_DEVICES=0,1,2,3 python finetune.py --model qwen-4 --data_category partial --partial_sample 3 --per_device_train_batch_size 1 --gradient_accumulation_steps 2 --epochs 1
CUDA_VISIBLE_DEVICES=0,1,2,3 python finetune.py --model qwen-4 --data_category partial --partial_sample 4 --per_device_train_batch_size 1 --gradient_accumulation_steps 2 --epochs 1
CUDA_VISIBLE_DEVICES=0,1,2,3 python finetune.py --model qwen-4 --data_category partial --partial_sample 5 --per_device_train_batch_size 1 --gradient_accumulation_steps 2 --epochs 1


python finetune.py --model qwen-7 --data_category abstract --per_device_train_batch_size 1 --gradient_accumulation_steps 2
python finetune.py --model qwen-7 --data_category contextualized --per_device_train_batch_size 1 --gradient_accumulation_steps 2
python finetune.py --model qwen-7 --data_category best_domains --per_device_train_batch_size 1 --gradient_accumulation_steps 2
python finetune.py --model qwen-7 --data_category worst_domains --per_device_train_batch_size 1 --gradient_accumulation_steps 2
python finetune.py --model qwen-7 --data_category partial --partial_sample 1 --per_device_train_batch_size 1 --gradient_accumulation_steps 2 --epochs 1
python finetune.py --model qwen-7 --data_category partial --partial_sample 2 --per_device_train_batch_size 1 --gradient_accumulation_steps 2 --epochs 1
python finetune.py --model qwen-7 --data_category partial --partial_sample 3 --per_device_train_batch_size 1 --gradient_accumulation_steps 2 --epochs 1
python finetune.py --model qwen-7 --data_category partial --partial_sample 4 --per_device_train_batch_size 1 --gradient_accumulation_steps 2 --epochs 1
python finetune.py --model qwen-7 --data_category partial --partial_sample 5 --per_device_train_batch_size 1 --gradient_accumulation_steps 2 --epochs 1


python finetune.py --model qwen-14 --data_category abstract --per_device_train_batch_size 1 --gradient_accumulation_steps 2
python finetune.py --model qwen-14 --data_category contextualized --per_device_train_batch_size 1 --gradient_accumulation_steps 2
python finetune.py --model qwen-14 --data_category best_domains --per_device_train_batch_size 1 --gradient_accumulation_steps 2
python finetune.py --model qwen-14 --data_category worst_domains --per_device_train_batch_size 1 --gradient_accumulation_steps 2
python finetune.py --model qwen-14 --data_category partial --partial_sample 1 --per_device_train_batch_size 1 --gradient_accumulation_steps 2 --epochs 1
python finetune.py --model qwen-14 --data_category partial --partial_sample 2 --per_device_train_batch_size 1 --gradient_accumulation_steps 2 --epochs 1
python finetune.py --model qwen-14 --data_category partial --partial_sample 3 --per_device_train_batch_size 1 --gradient_accumulation_steps 2 --epochs 1
python finetune.py --model qwen-14 --data_category partial --partial_sample 4 --per_device_train_batch_size 1 --gradient_accumulation_steps 2 --epochs 1
python finetune.py --model qwen-14 --data_category partial --partial_sample 5 --per_device_train_batch_size 1 --gradient_accumulation_steps 2 --epochs 1
