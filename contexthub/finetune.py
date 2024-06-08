from peft import (
    prepare_model_for_kbit_training,
    LoraConfig,
    get_peft_model,
    PeftModel,
    PeftConfig,
)
from utils import Logger, set_seed
from transformers.trainer_pt_utils import LabelSmoother
from typing import Dict, Optional, List
import sys
import os
from os.path import isfile, join
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json 
import argparse 
import time
import pandas as pd
import random

import torch
import math
import transformers
from accelerate import load_checkpoint_and_dispatch, init_empty_weights
import bitsandbytes as bnb
from peft.tuners.lora import LoraLayer
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from prompt import EVALUATE_PROMPTS


############# parse benchmark result #############
def parse_result_log(args):
    all_logs = ['result_log/' + l for l in os.listdir('result_log')]
    relevant_logs = [l for l in all_logs if args.model in l]
    domain_f1_scores = {}  # 存储域名和对应的 f1 分数
    for log in relevant_logs:
        with open(log, 'r') as f:
            lines = f.readlines()  # 逐行读取文件内容

        current_domain = None  # 当前的域名
        # 解析文件中的每一行
        for line in lines:
            line = line.strip()  # 去除前后空格和换行符
            if "Domain:" in line:
                # 确保正确地从冒号后提取域名
                current_domain = line.split('Domain:')[1].strip()
            elif "faverage f1:" in line and current_domain:
                # 确保正确地从冒号后提取 f1 分数
                f1_score = line.split('faverage f1:')[1].strip()
                f1_score = float(f1_score)
                if current_domain not in domain_f1_scores:
                    domain_f1_scores[current_domain] = f1_score
                else:
                    domain_f1_scores[current_domain] += f1_score
                current_domain = None  # 重置当前域名，避免错误重复使用
    _ = domain_f1_scores.pop('question')
    sorted_domains = sorted(domain_f1_scores.items(), key=lambda x: x[1], reverse=True)
    best_domains = [a[0] for a in sorted_domains[:3]]
    worst_domains = [a[0] for a in sorted_domains[-3:]]

    return best_domains, worst_domains

############# util function #############
def parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--model', type=str, default='qwen-4', help='qwen-4, qwen-7, qwen-14')
    parser.add_argument('--cache_dir', type=str, default='pretrained_models')

    parser.add_argument("--gradient_accumulation_steps", type=int, default=2)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--warmup_proportion", type=float, default=0.01)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--logging_steps", type=int, default=5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--partial_sample", type=int, default=1, help='number of samples per domain, could be 1, 2, 3, 4, 5')

    parser.add_argument("--lora_r", type=int, default=64)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)

    parser.add_argument('--adapter_model_output_dir', type=str, default='pretrained_models/adapter')
    parser.add_argument('--whole_model_output_dir', type=str, default='pretrained_models/whole_model')

    parser.add_argument('--data_category', type=str, default='all', help='all, abstract, culture, geography, activity, math, science, people, philosophy, religion, society, technology, health')
    args = parser.parse_args()

    return args

############# training data #############
def load_data():
    def collect_data(datas, category):
        collected_data = []
        for data, reasoning_type in datas:
            for template in data:
                for domain in template.keys():
                    if domain == category:
                        for subcat, subdata in template[domain].items():
                            text = subdata['<nl>']
                            text = EVALUATE_PROMPTS[reasoning_type].format(text)
                            answer = '<answer>'+ str(template['answer']).capitalize() +'</answer>'
                            cot = subdata['gold_cot']
                            collected_data.append({'question': text, 'answer': answer, 'cot': cot})
        return collected_data
    
    with open('data/data_level1/abductive_logic_traincot.json', 'r') as f:
        abductive_logic_level1 = json.load(f)
    with open('data/data_level2/abductive_logic_traincot.json', 'r') as f:
        abductive_logic_level2 = json.load(f)
    with open('data/data_level3/abductive_logic_traincot.json', 'r') as f:
        abductive_logic_level3 = json.load(f)
    with open('data/data_level4/abductive_logic_traincot.json', 'r') as f:
        abductive_logic_level4 = json.load(f)

    with open('data/data_level1/deductive_logic_traincot.json', 'r') as f:
        deductive_logic_level1 = json.load(f)
    with open('data/data_level2/deductive_logic_traincot.json', 'r') as f:
        deductive_logic_level2 = json.load(f)
    with open('data/data_level3/deductive_logic_traincot.json', 'r') as f:
        deductive_logic_level3 = json.load(f)
    with open('data/data_level4/deductive_logic_traincot.json', 'r') as f:
        deductive_logic_level4 = json.load(f)

    datas = [(abductive_logic_level1, 'abductive_logic'), (abductive_logic_level2, 'abductive_logic'), (abductive_logic_level3, 'abductive_logic'), (abductive_logic_level4, 'abductive_logic'), (deductive_logic_level1, 'deductive_logic'), (deductive_logic_level2, 'deductive_logic'), (deductive_logic_level3, 'deductive_logic'), (deductive_logic_level4, 'deductive_logic')]

    culture_data = collect_data(datas, 'culture and arts')
    geography_data = collect_data(datas, 'geography and places')
    activity_data = collect_data(datas, 'human activities')
    math_data = collect_data(datas, 'mathematics and logic')
    science_data = collect_data(datas, 'natural and physical sciences')
    people_data = collect_data(datas, 'people and self')
    philosophy_data = collect_data(datas, 'philosophy and thinking')
    religion_data = collect_data(datas, 'religion and belief systems')
    society_data = collect_data(datas, 'society and social sciences')
    technology_data = collect_data(datas, 'technology and applied sciences')
    health_data = collect_data(datas, 'health and fitness')
    abstract_data = collect_data(datas, 'abstract')

    return culture_data, geography_data, activity_data, math_data, science_data, people_data, philosophy_data, religion_data, society_data, technology_data, health_data, abstract_data

def return_data_category(data_category, data_list):
    if data_category == 'abstract':
        return data_list[-1]
    elif data_category == "culture and arts":
        return data_list[0]
    elif data_category == "geography and places":
        return data_list[1]
    elif data_category == "human activities":
        return data_list[2]
    elif data_category == "mathematics and logic":
        return data_list[3]
    elif data_category == "natural and physical sciences":
        return data_list[4]
    elif data_category == "people and self":
        return data_list[5]
    elif data_category == "philosophy and thinking":
        return data_list[6]
    elif data_category == "religion and belief systems":
        return data_list[7]
    elif data_category == "society and social sciences":
        return data_list[8]
    elif data_category == "technology and applied sciences":
        return data_list[9]
    elif data_category == "health and fitness":
        return data_list[10]

def load_partial_data(args):
    def collect_data(datas, category):
        collected_data = []
        for data, reasoning_type in datas:
            for template in data:
                for domain in template.keys():
                    if domain == category:
                        i = 0
                        for subcat, subdata in template[domain].items():
                            text = subdata['<nl>']
                            text = EVALUATE_PROMPTS[reasoning_type].format(text)
                            answer = '<answer>'+ str(template['answer']).capitalize() +'</answer>'
                            cot = subdata['gold_cot']
                            collected_data.append({'question': text, 'answer': answer, 'cot': cot})
                            i += 1
                            if i >= args.partial_sample:
                                break
        collected_data = random.sample(collected_data*5, k=len(collected_data)*int(5/args.partial_sample))
        return collected_data
    
    with open('data/data_level1/abductive_logic_traincot.json', 'r') as f:
        abductive_logic_level1 = json.load(f)
    with open('data/data_level2/abductive_logic_traincot.json', 'r') as f:
        abductive_logic_level2 = json.load(f)
    with open('data/data_level3/abductive_logic_traincot.json', 'r') as f:
        abductive_logic_level3 = json.load(f)
    with open('data/data_level4/abductive_logic_traincot.json', 'r') as f:
        abductive_logic_level4 = json.load(f)

    with open('data/data_level1/deductive_logic_traincot.json', 'r') as f:
        deductive_logic_level1 = json.load(f)
    with open('data/data_level2/deductive_logic_traincot.json', 'r') as f:
        deductive_logic_level2 = json.load(f)
    with open('data/data_level3/deductive_logic_traincot.json', 'r') as f:
        deductive_logic_level3 = json.load(f)
    with open('data/data_level4/deductive_logic_traincot.json', 'r') as f:
        deductive_logic_level4 = json.load(f)

    datas = [(abductive_logic_level1, 'abductive_logic'), (abductive_logic_level2, 'abductive_logic'), (abductive_logic_level3, 'abductive_logic'), (abductive_logic_level4, 'abductive_logic'), (deductive_logic_level1, 'deductive_logic'), (deductive_logic_level2, 'deductive_logic'), (deductive_logic_level3, 'deductive_logic'), (deductive_logic_level4, 'deductive_logic')]

    culture_data = collect_data(datas, 'culture and arts')
    geography_data = collect_data(datas, 'geography and places')
    activity_data = collect_data(datas, 'human activities')
    math_data = collect_data(datas, 'mathematics and logic')
    science_data = collect_data(datas, 'natural and physical sciences')
    people_data = collect_data(datas, 'people and self')
    philosophy_data = collect_data(datas, 'philosophy and thinking')
    religion_data = collect_data(datas, 'religion and belief systems')
    society_data = collect_data(datas, 'society and social sciences')
    technology_data = collect_data(datas, 'technology and applied sciences')
    health_data = collect_data(datas, 'health and fitness')
    abstract_data = collect_data(datas, 'abstract')

    return culture_data, geography_data, activity_data, math_data, science_data, people_data, philosophy_data, religion_data, society_data, technology_data, health_data, abstract_data

def construct_training_data(args, logger, tokenizer):
    best_domains, worst_domains = parse_result_log(args)
    logger.log("best domains: " + str(best_domains))
    logger.log("worst domains: " + str(worst_domains))
    culture_data, geography_data, activity_data, math_data, science_data, people_data, philosophy_data, religion_data, society_data, technology_data, health_data, abstract_data = load_data()
    if args.data_category == 'all':
        all_data = culture_data + geography_data + activity_data + math_data + science_data + people_data + philosophy_data + religion_data + society_data + technology_data + health_data + abstract_data
        all_data = ['<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n'+t['question']+'<|im_end|>\n<|im_start|>assistant\n'+t['cot'] for t in all_data]
        random.shuffle(all_data)
    elif args.data_category == 'best_domains':
        all_data = []
        for domain in best_domains:
            data = return_data_category(domain, [culture_data, geography_data, activity_data, math_data, science_data, people_data, philosophy_data, religion_data, society_data, technology_data, health_data, abstract_data])
            all_data += data
        all_data = ['<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n'+t['question']+'<|im_end|>\n<|im_start|>assistant\n'+t['cot'] for t in all_data]
        all_data = random.sample(all_data, k=int(len(all_data)/3))
        random.shuffle(all_data)
    elif args.data_category == 'worst_domains':
        all_data = []
        for domain in worst_domains:
            data = return_data_category(domain, [culture_data, geography_data, activity_data, math_data, science_data, people_data, philosophy_data, religion_data, society_data, technology_data, health_data, abstract_data])
            all_data += data
        all_data = ['<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n'+t['question']+'<|im_end|>\n<|im_start|>assistant\n'+t['cot'] for t in all_data]
        all_data = random.sample(all_data, k=int(len(all_data)/3))
        random.shuffle(all_data)
    elif args.data_category == 'abstract':
        abstract_data = ['<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n'+t['question']+'<|im_end|>\n<|im_start|>assistant\n'+t['cot'] for t in abstract_data]
        random.shuffle(abstract_data)
        all_data = abstract_data
    elif args.data_category == 'contextualized':
        abstract_length = len(abstract_data)
        all_data = culture_data + geography_data + activity_data + math_data + science_data + people_data + philosophy_data + religion_data + society_data + technology_data + health_data
        all_data = random.sample(all_data, k=abstract_length)
        all_data = ['<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n'+t['question']+'<|im_end|>\n<|im_start|>assistant\n'+t['cot'] for t in all_data]
        random.shuffle(all_data)
    elif args.data_category == 'partial':
        assert args.epochs == 1
        culture_data, geography_data, activity_data, math_data, science_data, people_data, philosophy_data, religion_data, society_data, technology_data, health_data, abstract_data = load_partial_data(args)
        all_data = culture_data + geography_data + activity_data + math_data + science_data + people_data + philosophy_data + religion_data + society_data + technology_data + health_data + abstract_data
        all_data = ['<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n'+t['question']+'<|im_end|>\n<|im_start|>assistant\n'+t['cot'] for t in all_data]
        random.shuffle(all_data)
    else:
        all_data = return_data_category(args.data_category, [culture_data, geography_data, activity_data, math_data, science_data, people_data, philosophy_data, religion_data, society_data, technology_data, health_data, abstract_data])
        all_data = ['<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n'+t['question']+'<|im_end|>\n<|im_start|>assistant\n'+t['cot'] for t in all_data]
        random.shuffle(all_data)

    def load_custom_dataset(data):
        train_encodings = tokenizer(data, truncation=True, padding=True)
        IGNORE_TOKEN_ID = LabelSmoother.ignore_index

        class InputDataset(torch.utils.data.Dataset):
            def __init__(self, encodings):
                self.encodings = encodings

            def __getitem__(self, idx):
                item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
                return item

            def __len__(self):
                return len(self.encodings["input_ids"])

        train_dataset = InputDataset(train_encodings)

        return train_dataset

    #all_data = all_data[:10]
    train_dataset = load_custom_dataset(all_data)

    return train_dataset
            
############# training helper function #############
def print_trainable_parameters(model, logger):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    logger.log(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )

def find_all_linear_names(args, model):
    cls = bnb.nn.Linear4bit
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split(".")
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if "lm_head" in lora_module_names:  # needed for 16-bit
        lora_module_names.remove("lm_head")
    return list(lora_module_names)

def training_steps(args, training_dataset_length):
    num_gpus = torch.cuda.device_count()
    training_steps = int(
        math.ceil(
            training_dataset_length
            / (args.gradient_accumulation_steps * args.per_device_train_batch_size)
        )
        * args.epochs
    )
    warmup_steps = int(math.ceil(training_steps * args.warmup_proportion))

    return training_steps, warmup_steps

def load_4bit_model(args, model_name):
    config = AutoConfig.from_pretrained(model_name, torch_dtype=torch.float16,trust_remote_code=True)
    config.tie_word_embeddings = True

    nf4_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )

    # with init_empty_weights():
    #     model = AutoModelForCausalLM.from_config(config,trust_remote_code=True)
    #     model.tie_weights()

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=nf4_config,
        #device_map="auto",
        device_map={'':torch.cuda.current_device()},
        offload_state_dict=True,
        cache_dir=args.cache_dir,
        trust_remote_code=True
    )
    
    if 'Qwen' in model_name:
        tokenizer = AutoTokenizer.from_pretrained(model_name,trust_remote_code=True, pad_token='<|endoftext|>',cache_dir=args.cache_dir,padding_side="right",model_max_length=8192)
    elif 'Mistral' in model_name:
        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=args.cache_dir, trust_remote_code=True)
        tokenizer.pad_token_id = tokenizer.eos_token_id
    elif 'vicuna' in model_name:
        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=args.cache_dir, trust_remote_code=True)
        tokenizer.pad_token = '</s>'
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name,cache_dir=args.cache_dir, trust_remote_code=True)
        model.config.pad_token_id = tokenizer.pad_token_id = 0
        model.config.bos_token_id = tokenizer.bos_token_id
        model.config_eos_token_id = tokenizer.eos_token_id

    return model, tokenizer

if __name__ == '__main__':
    args = parser()
    set_seed(args.seed)
    logger = Logger(f'finetune/{args.model}_{args.data_category}.log')

    logger.log('---start finetuning on the setting---')
    logger.log(args.data_category)
    logger.log('-------------------------------------')

    logger.log("loading model ...")
    if args.model == 'qwen-4':
        model_name = 'Qwen/Qwen1.5-4B-Chat'
    elif args.model == 'qwen-0.5':
        model_name = 'Qwen/Qwen1.5-0.5B-Chat'
    elif args.model == 'qwen-7':
        model_name = 'Qwen/Qwen1.5-7B-Chat'
    elif args.model == 'qwen-14':
        model_name = 'Qwen/Qwen1.5-14B-Chat'
    elif args.model == 'qwen-32':
        model_name = 'Qwen/Qwen1.5-32B-Chat'
    model, tokenizer = load_4bit_model(args, model_name)

    try:
        model.gradient_checkpointing_enable()
        model = prepare_model_for_kbit_training(model)
    except:
        logger.log("gradient checkpointing not supported for model {}".format(model_name))

    modules = find_all_linear_names(args, model)
    config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=modules,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, config)
    print_trainable_parameters(model, logger)

    for name, module in model.named_modules():
        if isinstance(module, LoraLayer):
            module = module.to(torch.float32)
        if "norm" in name:
            module = module.to(torch.float32)
        if "lm_head" in name or "embed_tokens" in name:
            if hasattr(module, "weight"):
                if module.weight.dtype == torch.float32:
                    module = module.to(torch.float32)

    logger.log("model loaded.")

    train_dataset = construct_training_data(args, logger, tokenizer)

    training_steps, warmup_steps = training_steps(args, len(train_dataset))
    logger.log(
        """
length of training dataset: {}
number of training steps: {}
number of warmup steps: {}
    """.format(
            len(train_dataset), training_steps, warmup_steps
        )
    )

    logger.log("start training ...")
    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_dataset,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=args.per_device_train_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            warmup_steps=warmup_steps,
            max_steps=training_steps,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            fp16=True,
            logging_steps=args.logging_steps,
            logging_dir=f'finetune/{args.model}_{args.data_category}.log',
            output_dir="outputs",
            optim="paged_adamw_8bit",
        ),
        data_collator=transformers.DataCollatorForLanguageModeling(
            tokenizer, mlm=False
        ),
    )
    model.config.use_cache = (
        False  # silence the warnings. Please re-enable for inference!
    )
    trainer.train()

    # append partial number
    if args.data_category == 'partial':
        args.data_category += str(args.partial_sample)

    adapter_model_output_dir = args.adapter_model_output_dir + '/' + args.model + '_'+args.data_category
    model.save_pretrained(adapter_model_output_dir)
    logger.log("adapter model saved.")

    # base_model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", cache_dir=args.cache_dir, trust_remote_code=True)
    # model = PeftModel.from_pretrained(base_model, adapter_model_output_dir)
    # model = model.merge_and_unload()

    # whole_model_output_dir = args.whole_model_output_dir + '/' + args.model + '_'+args.data_category
    # model.save_pretrained(whole_model_output_dir)
    # tokenizer.save_pretrained(whole_model_output_dir)
    # logger.log("whole model and tokenizer saved.")

