from peft import (
    PeftModel,
)
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse 
import pandas as pd

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)
from utils import Logger

def parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', type=str, default='qwen-4', help='qwen-4, qwen-7, qwen-14')

    parser.add_argument('--cache_dir', type=str, default='pretrained_models')
    parser.add_argument("--partial_sample", type=int, default=1, help='number of samples per domain, could be 1, 2, 3, 4, 5')

    parser.add_argument('--adapter_model_output_dir', type=str, default='pretrained_models/adapter')
    parser.add_argument('--whole_model_output_dir', type=str, default='pretrained_models/whole_model')

    parser.add_argument('--data_category', type=str, default='abstract', help='all, abstract, culture, geography, activity, math, science, people, philosophy, religion, society, technology, health')
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parser()
    #logger = Logger('merge.log')

    if args.model == 'qwen-4':
        model_name = 'Qwen/Qwen1.5-4B-Chat'
    elif args.model == 'qwen-0.5':
        model_name = 'Qwen/Qwen1.5-0.5B-Chat'
    elif args.model == 'qwen-7':
        model_name = 'Qwen/Qwen1.5-7B-Chat'
    elif args.model == 'qwen-14':
        model_name = 'Qwen/Qwen1.5-14B-Chat'

    tokenizer = AutoTokenizer.from_pretrained(model_name,trust_remote_code=True, pad_token='<|endoftext|>',cache_dir=args.cache_dir,padding_side="right",model_max_length=8192)

    # append partial number
    if args.data_category == 'partial':
        args.data_category += str(args.partial_sample)

    adapter_model_output_dir = args.adapter_model_output_dir + '/' + args.model + '_'+args.data_category

    base_model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", cache_dir=args.cache_dir, trust_remote_code=True)
    model = PeftModel.from_pretrained(base_model, adapter_model_output_dir)
    model = model.merge_and_unload()
    #logger.log(f'adapter model loaded and merged from {adapter_model_output_dir}.')

    whole_model_output_dir = args.whole_model_output_dir + '/' + args.model + '_'+args.data_category
    model.save_pretrained(whole_model_output_dir)
    tokenizer.save_pretrained(whole_model_output_dir)
    #logger.log(f"whole model and tokenizer saved to {whole_model_output_dir}.")
