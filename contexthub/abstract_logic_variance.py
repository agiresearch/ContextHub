import json
from model import call_data_generation_model, call_evaluation_model, call_anthropic_api, load_evaluation_model, call_gpt35_api
import argparse
import re
import os
import time
import random
from utils import set_seed
from prompt import EVALUATE_PROMPTS, PARSE_PROMPTS
from tqdm import tqdm
import openai
import numpy as np

def parse_function(result):
    if '<answer>' not in result:
        return 'no answer'
    assert '<answer>' in result
    start = result.index('<answer>')
    end = result.index('</answer>')
    answer = result[start + len('<answer>'):end]
    return answer

def parse_result(parse_prompt, question, result):
    prompt = parse_prompt.format(question, result)
    try:
        answer = re.findall(r'<answer>(.*?)</answer>', result, re.DOTALL)[-1]
    except:
        while True:
            try:
                #parsed_result = call_anthropic_api(prompt)
                parsed_result = call_gpt35_api(prompt)
                answer = parse_function(parsed_result)
                break
            except KeyboardInterrupt:
                raise
            except openai.BadRequestError as e:
                print(f"BadRequestError: {e}")
                return {'answer':'no answer', 'reasoning':result}
            except OSError as e:
                print(f"OSError: {e}")
                return {'answer':'no answer', 'reasoning':result}
            except:
                time.sleep(0.1)
    return {'answer':answer, 'reasoning':result}

def run(args, llm, evaluate_prompt, parse_prompt, data):
    prompts = []
    questions = []
    for one_data in data:
        text = one_data['descriptions']
        one_prompt = evaluate_prompt.format(text)
        prompts.append(one_prompt)
        questions.append(text.split('\n')[-1])
    # run model
    results = call_evaluation_model(model=args.model, text_prompt=prompts, llm=llm)
    # parse result
    new_results = []
    for r in tqdm(results, desc="Parsing generated answers"):
        new_results.append(parse_result(parse_prompt, questions, r))
    # collect result
    for i in range(len(data)):
        data[i]['result'] = new_results[i]

    return data

def check_correctness(output, answer):
    if output in ['true', 'false', 'n/a']:
        if output == answer or bool(output) == answer:
            # print('correct', (output, answer))
            return True
    # print('wrong', (output, answer))
    return False

def compute_variance(data, n):
    def sample_data(data, n):
        true_data = [d for d in data if d['answers']==True]
        false_data = [d for d in data if d['answers']==False]
        na_data = [d for d in data if d['answers']=='N/A']
        sampled_data = []
        if false_data:
            sub_n = int(n/3)
            sampled_data += random.sample(true_data, min(sub_n, len(true_data)))
            sampled_data += random.sample(false_data, min(sub_n, len(false_data)))
            sampled_data += random.sample(na_data, min(sub_n, len(na_data)))
        else:
            sub_n = int(n/2)
            sampled_data += random.sample(true_data, min(sub_n, len(true_data)))
            sampled_data += random.sample(na_data, min(sub_n, len(na_data)))
        return sampled_data
    
    ten_samples = [sample_data(data, n) for _ in range(10)]

    accuracies = []
    for sample in ten_samples:
        correct = 0
        for d in sample:
            if check_correctness(d['result']['answer'].lower(), str(d['answers']).lower()):
                correct += 1
        accuracies.append(correct/n)

    return np.mean(accuracies), np.std(accuracies)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="qwen-0.5", help="The model to evaluate.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--level", type=int, default=2, help="The difficulty level: 2,3,4")
    parser.add_argument("--dataset", type=str, default="abductive_logic", help="The dataset")
    parser.add_argument('--sample_n', type=int, default=40, help='The number of samples')
    args = parser.parse_args()

    set_seed(args.seed)
    
    # if args.with_rules:
    #     evaluate_prompt = EVALUATE_PROMPTS_WITH_BASIC_RULES[args.dataset]
    # else:
    evaluate_prompt = EVALUATE_PROMPTS[args.dataset]

    parse_prompt = PARSE_PROMPTS[args.dataset]

    data_path = f'data/data_variance/{args.dataset}_level{args.level}.json'
    result_path = f'result/data_variance/{args.dataset}_level{args.level}_{args.model}.json'

    if os.path.exists(result_path):
        print('=='*20)
        print(result_path)
        print('=='*20)
        with open(result_path, 'r') as f:
            dataset = json.load(f)
        mean, variance = compute_variance(dataset, args.sample_n)
        print(mean)
        print(variance)
    else:
        print('=='*20)
        print(data_path)
        print('=='*20)

        with open(data_path, 'r') as f:
            dataset = json.load(f)

        llm = load_evaluation_model(args.model)

        full_data = run(args, llm, evaluate_prompt, parse_prompt, dataset)

        with open(result_path, 'w') as f:
            json.dump(full_data, f)

        mean, variance = compute_variance(full_data, args.sample_n)
        print(mean)
        print(variance)
