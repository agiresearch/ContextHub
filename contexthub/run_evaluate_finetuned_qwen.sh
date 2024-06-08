#!/bin/bash

################ deductive logic ################
# qwen-7 level1 abstract & contextualized
# python evaluate.py --model qwen-7 --dataset deductive_logic --level 1 --seed 42 --trained_model_name pretrained_models/whole_model/qwen-7_abstract
# ulimit -n 8192
# python compute_score.py --model qwen-7 --dataset deductive_logic --level 1 --trained_model_name pretrained_models/whole_model/qwen-7_abstract
# python evaluate.py --model qwen-7 --dataset deductive_logic --level 1 --seed 42 --trained_model_name pretrained_models/whole_model/qwen-7_contextualized
# ulimit -n 8192
# python compute_score.py --model qwen-7 --dataset deductive_logic --level 1 --trained_model_name pretrained_models/whole_model/qwen-7_contextualized

# # qwen-7 level2 abstract & contextualized
# python evaluate.py --model qwen-7 --dataset deductive_logic --level 2 --seed 42 --trained_model_name pretrained_models/whole_model/qwen-7_abstract
# ulimit -n 8192
# python compute_score.py --model qwen-7 --dataset deductive_logic --level 2 --trained_model_name pretrained_models/whole_model/qwen-7_abstract
# python evaluate.py --model qwen-7 --dataset deductive_logic --level 2 --seed 42 --trained_model_name pretrained_models/whole_model/qwen-7_contextualized
# ulimit -n 8192
# python compute_score.py --model qwen-7 --dataset deductive_logic --level 2 --trained_model_name pretrained_models/whole_model/qwen-7_contextualized

# # qwen-7 level3 abstract & contextualized
# python evaluate.py --model qwen-7 --dataset deductive_logic --level 3 --seed 42 --trained_model_name pretrained_models/whole_model/qwen-7_abstract
# ulimit -n 8192
# python compute_score.py --model qwen-7 --dataset deductive_logic --level 3 --trained_model_name pretrained_models/whole_model/qwen-7_abstract
# python evaluate.py --model qwen-7 --dataset deductive_logic --level 3 --seed 42 --trained_model_name pretrained_models/whole_model/qwen-7_contextualized
# ulimit -n 8192
# python compute_score.py --model qwen-7 --dataset deductive_logic --level 3 --trained_model_name pretrained_models/whole_model/qwen-7_contextualized

# qwen-7 level4 abstract & contextualized
# python evaluate.py --model qwen-7 --dataset deductive_logic --level 4 --seed 42 --trained_model_name pretrained_models/whole_model/qwen-7_abstract
# ulimit -n 8192
# python compute_score.py --model qwen-7 --dataset deductive_logic --level 4 --trained_model_name pretrained_models/whole_model/qwen-7_abstract
python evaluate.py --model qwen-7 --dataset deductive_logic --level 4 --seed 42 --trained_model_name pretrained_models/whole_model/qwen-7_contextualized
ulimit -n 8192
python compute_score.py --model qwen-7 --dataset deductive_logic --level 4 --trained_model_name pretrained_models/whole_model/qwen-7_contextualized


################ abductive logic ################
# python evaluate.py --model qwen-7 --dataset abductive_logic --level 1 --seed 42 --trained_model_name pretrained_models/whole_model/qwen-7_abstract
# ulimit -n 8192
# python compute_score.py --model qwen-7 --dataset abductive_logic --level 1 --trained_model_name pretrained_models/whole_model/qwen-7_abstract
# python evaluate.py --model qwen-7 --dataset abductive_logic --level 1 --seed 42 --trained_model_name pretrained_models/whole_model/qwen-7_contextualized
# ulimit -n 8192
# python compute_score.py --model qwen-7 --dataset abductive_logic --level 1 --trained_model_name pretrained_models/whole_model/qwen-7_contextualized

# # qwen-7 level2 abstract & contextualized
# python evaluate.py --model qwen-7 --dataset abductive_logic --level 2 --seed 42 --trained_model_name pretrained_models/whole_model/qwen-7_abstract
# ulimit -n 8192
# python compute_score.py --model qwen-7 --dataset abductive_logic --level 2 --trained_model_name pretrained_models/whole_model/qwen-7_abstract
# python evaluate.py --model qwen-7 --dataset abductive_logic --level 2 --seed 42 --trained_model_name pretrained_models/whole_model/qwen-7_contextualized
# ulimit -n 8192
# python compute_score.py --model qwen-7 --dataset abductive_logic --level 2 --trained_model_name pretrained_models/whole_model/qwen-7_contextualized

# # qwen-7 level3 abstract & contextualized
# python evaluate.py --model qwen-7 --dataset abductive_logic --level 3 --seed 42 --trained_model_name pretrained_models/whole_model/qwen-7_abstract
# ulimit -n 8192
# python compute_score.py --model qwen-7 --dataset abductive_logic --level 3 --trained_model_name pretrained_models/whole_model/qwen-7_abstract
# python evaluate.py --model qwen-7 --dataset abductive_logic --level 3 --seed 42 --trained_model_name pretrained_models/whole_model/qwen-7_contextualized
# ulimit -n 8192
# python compute_score.py --model qwen-7 --dataset abductive_logic --level 3 --trained_model_name pretrained_models/whole_model/qwen-7_contextualized

# # qwen-7 level4 abstract & contextualized
# python evaluate.py --model qwen-7 --dataset abductive_logic --level 4 --seed 42 --trained_model_name pretrained_models/whole_model/qwen-7_abstract
# ulimit -n 8192
# python compute_score.py --model qwen-7 --dataset abductive_logic --level 4 --trained_model_name pretrained_models/whole_model/qwen-7_abstract
# python evaluate.py --model qwen-7 --dataset abductive_logic --level 4 --seed 42 --trained_model_name pretrained_models/whole_model/qwen-7_contextualized
# ulimit -n 8192
# python compute_score.py --model qwen-7 --dataset abductive_logic --level 4 --trained_model_name pretrained_models/whole_model/qwen-7_contextualized

