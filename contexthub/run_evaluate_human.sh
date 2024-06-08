#!/bin/bash

########## human abductive logic ##########
ulimit -n 8192
python evaluate.py --model human-gpt-35 --dataset abductive_logic --level 1 --seed 42 --trained_model_name gpt-35_human
python compute_score.py --model human-gpt-35 --dataset abductive_logic --level 1 --trained_model_name gpt-35_human
ulimit -n 8192
python evaluate.py --model human-gpt-35 --dataset abductive_logic --level 2 --seed 42 --trained_model_name gpt-35_human
python compute_score.py --model human-gpt-35 --dataset abductive_logic --level 2 --trained_model_name gpt-35_human
ulimit -n 8192
python evaluate.py --model human-gpt-35 --dataset abductive_logic --level 3 --seed 42 --trained_model_name gpt-35_human
python compute_score.py --model human-gpt-35 --dataset abductive_logic --level 3 --trained_model_name gpt-35_human
ulimit -n 8192
python evaluate.py --model human-gpt-35 --dataset abductive_logic --level 4 --seed 42 --trained_model_name gpt-35_human
python compute_score.py --model human-gpt-35 --dataset abductive_logic --level 4 --trained_model_name gpt-35_human
########## human deductive logic ##########
ulimit -n 8192
python evaluate.py --model human-gpt-35 --dataset deductive_logic --level 1 --seed 42 --trained_model_name gpt-35_human
python compute_score.py --model human-gpt-35 --dataset deductive_logic --level 1 --trained_model_name gpt-35_human
ulimit -n 8192
python evaluate.py --model human-gpt-35 --dataset deductive_logic --level 2 --seed 42 --trained_model_name gpt-35_human
python compute_score.py --model human-gpt-35 --dataset deductive_logic --level 2 --trained_model_name gpt-35_human
ulimit -n 8192
python evaluate.py --model human-gpt-35 --dataset deductive_logic --level 3 --seed 42 --trained_model_name gpt-35_human
python compute_score.py --model human-gpt-35 --dataset deductive_logic --level 3 --trained_model_name gpt-35_human
ulimit -n 8192
python evaluate.py --model human-gpt-35 --dataset deductive_logic --level 4 --seed 42 --trained_model_name gpt-35_human
python compute_score.py --model human-gpt-35 --dataset deductive_logic --level 4 --trained_model_name gpt-35_human