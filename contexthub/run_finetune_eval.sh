#!/bin/bash

# python finetune_eval.py --model qwen-4
# python finetune_eval.py --model qwen-4 --trained_model_name pretrained_models/whole_model/qwen-4_abstract
# python finetune_eval.py --model qwen-4 --trained_model_name pretrained_models/whole_model/qwen-4_contextualized

# python finetune_eval.py --model qwen-14
# python finetune_eval.py --model qwen-14 --trained_model_name pretrained_models/whole_model/qwen-14_abstract
# python finetune_eval.py --model qwen-14 --trained_model_name pretrained_models/whole_model/qwen-14_contextualized

python finetune_eval.py --model qwen-4 --trained_model_name pretrained_models/whole_model/qwen-4_all
python finetune_eval.py --model qwen-14 --trained_model_name pretrained_models/whole_model/qwen-14_all
